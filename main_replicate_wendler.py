# This script does the exact same experiments as the orignal paper https://arxiv.org/pdf/2402.10588
# but faster, in batched mode.

# %%
try:
    %load_ext autoreload
    %autoreload 2
except:
    pass
# %%
from torch._tensor import Tensor
from imports import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
from src.prompt import gen_prompt, gen_common_suffixes, get_answer_tensor2, get_valid_answer
from src.kv_cache import gen_kv_cache, run_with_kv_cache
from src.intervention import Intervention
from src.constants import LANGS, WORD_LIST
from src.llm import safe_tokenize, num_correct, loss_on_answers
from utils.data import gen_lang_ids, results_dict_to_csv

from utils.plot import plot_ci_simple
from utils.config_argparse import try_parse_args
from utils.data import parse_word_list, gen_lang_ids, gen_ids
from utils.misc import ci, wilson_ci
from utils.tokenizer import AutoTLTokenizer

import src.wendler as wendler

from eindex import eindex
from collections import namedtuple
import warnings
import re
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache
from src.kv_cache import broadcast_kv_cache
from transformer_lens.utils import test_prompt
from types import SimpleNamespace
from tqdm import tqdm
from typing import Literal
# Import GPT-2 tokenizer
#disable gradients
torch.set_grad_enabled(False)

from ast import literal_eval
from tabulate import tabulate
import warnings
# %%
@dataclass
class Config:
    seed: int = 42
    #model_name: str = "meta-llama/Llama-3.1-8B"
    model_name: str = "meta-llama/Llama-2-7b-hf"
    #model_name: str = "mistralai/Mistral-7B-v0.1"
    # single_token_only: bool = False
    # multi_token_only: bool = False
    # out_dir: str = './out_iclr'
    dataset_path: str = "data_wendler/word_list.csv"
    task : Literal["translate", "copy", "cloze"] = "translate"
    debug: bool = True
    num_multi_shot : int = 5
    token_add_spaces: bool = True
    token_add_leading_byte: bool = True 
    token_add_prefixes : bool = False
    token_add_capitalization : bool = True
    quantize: Optional[str] = None
    word_list_key : str = 'claude'
    src_lang : str = None
    dest_lang : str = None
    latent_lang : str = 'en'
    devices : Optional[str] = "1,2"
    out : Optional[str] = "out_icml_2025"
    token_add_capitalization : bool = True
    token_add_prefixes : bool = True
    token_add_spaces : bool = True
    token_utf8_byte : bool = True

cfg = Config()
cfg = try_parse_args(cfg)
cfg_dict = asdict(cfg)
print(cfg_dict)


if cfg.devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.devices  # Makes only GPUs 0 and 1 visible
    n_devices = torch.cuda.device_count()
    assert n_devices == len(cfg.devices.split(',')), f"Expected {n_devices} devices, got {cfg.devices}"
else:
    n_devices = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#os.makedirs(cfg.out_dir, exist_ok=True)


tokenizer = AutoTLTokenizer.from_pretrained(cfg.model_name)

vocab = tokenizer.get_vocab()
space_token = tokenizer.convert_ids_to_tokens(safe_tokenize(" ", tokenizer).input_ids.item())

#loaded_model = False
if 'loaded_model' not in globals():
    model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                        device=device,
                                                        n_devices = n_devices,
                                                        dtype = torch.bfloat16)
    loaded_model = True
print("Devices:", cfg.devices)

# %%
from itertools import combinations, product, permutations


lang_pairs = list(permutations(LANGS, 2))

output_results = pd.DataFrame(columns=['src_lang', 'dest_lang', 'latent_lang', 'avg_prob', 'sem95_error', 'acc'])

def probs_on_answer(logits, answers, padding_id = -1, log_probs=False):
    softmax_fn = torch.log_softmax if log_probs else torch.softmax
    sum_fn = torch.logsumexp if log_probs else torch.sum
    zero = float('-inf') if log_probs else 0

    probs_est = softmax_fn(logits, dim=-1) # (batch, vocab)
    on_answer = eindex(probs_est, answers, "batch [batch seq] -> batch seq")
    on_answer[answers == tokenizer.pad_token_id] = zero
    on_answer = sum_fn(on_answer, dim=-1) # (batch,)
    return on_answer

def loss_on_answer(logits, answers, padding_id = -1) -> Float[Tensor, "batch"]:
    return -probs_on_answer(logits, answers, padding_id, log_probs=True)



print("Computing translation probabilities for each language pair")
runner = tqdm(lang_pairs)
for (src, dest) in tqdm(lang_pairs):
    df = wendler.load_data("data_wendler/langs", SimpleNamespace(src_lang=src, dest_lang=dest))
    mask = df['en'].isin(WORD_LIST)
    prompt_df, suffix_df = df[mask], df[~mask]
    
    prompt = gen_prompt(prompt_df, src, dest)
    common_suffixes = gen_common_suffixes(suffix_df[src], src, dest)
    kv_cache = gen_kv_cache(prompt, model)
    suffix_toks = safe_tokenize(common_suffixes, model)

    answers = get_answer_tensor2(suffix_df[dest], tokenizer, vocab, cfg, padding_value=tokenizer.pad_token_id) # (batch, num_answers)
    # answers_en = get_answer_tensor(suffix_df['en'], vocab, space_token, cfg, padding_value=tokenizer.pad_token_id) # (batch, num_answers)

    logits, cache = run_with_kv_cache(suffix_toks, 
                                  kv_cache, 
                                  model, 
                                  mini_batch_size = 32,
                                  last_seq = True)

    preds = logits.argmax(dim=-1, keepdim=True).to(answers.device)
    correct = torch.any((preds == answers) & (preds != tokenizer.pad_token_id), dim=-1).sum().item()

    probs = probs_on_answer(logits, answers, padding_id = tokenizer.pad_token_id, log_probs=False)
    mean_loss = -torch.log(probs).mean().item()
    
    mean_probs = probs.mean().item()
    ci95 = 1.96 * probs.std().item() / np.sqrt(probs.shape[0])


    acc = correct / answers.shape[0]
    print(f"{src} -> {dest} Translated {correct}/{answers.shape[0]} correctly. Accuracy: {acc:.2%} Loss: {mean_loss:.2f} Probs {mean_prosb:.2f} ± {ci95:.2f}")
    output_results = output_results.append({'src_lang': src, 'dest_lang': dest, 'latent_lang': , 'avg_prob': mean_probs, 'sem95_error': ci95, 'acc': acc}, ignore_index=True)
# %%
print(acc_dict)

# Plotting bar chart for each translation pair with error bars
pairs = list(acc_dict.keys())
accuracies = [acc_dict[pair][2] for pair in pairs]
mean_loss = [acc_dict[pair][3] for pair in pairs]
ci95 = [acc_dict[pair][4] for pair in pairs]

plt.figure(figsize=(12, 8))
plt.bar(range(len(pairs)), mean_loss, yerr=ci95, tick_label=[f"{src}->{dest}" for src, dest in pairs], capsize=5)
plt.xlabel('Translation Pairs')
plt.ylabel('Loss (-log likelihood)')
plt.title('Translation Accuracy for Each Language Pair')
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()




# %%
#names_filter = ["hook_embed"] + [f"block.{x}.hook_resid_post" for x in range(model.cfg.n_layers)]
src, dest = 'fr', 'de'
df = wendler.load_data("data_wendler/langs", SimpleNamespace(src_lang=src, dest_lang=dest))
mask = df['en'].isin(WORD_LIST)
prompt_df, suffix_df = df[mask], df[~mask]

prompt = gen_prompt(prompt_df, src, dest)
common_suffixes = gen_common_suffixes(suffix_df[src], src, dest)
kv_cache = gen_kv_cache(prompt, model)
suffix_toks = safe_tokenize(common_suffixes, model)

logits, cache = run_with_kv_cache(suffix_toks, 
                                  kv_cache, 
                                  model, 
                                  keep_resid_cache= True, 
                                  mini_batch_size = 64,
                                  last_seq = True)

answers = get_answer_tensor2(suffix_df[dest], tokenizer, vocab, cfg, padding_value=tokenizer.pad_token_id) # 
answers_en = get_answer_tensor2(suffix_df['en'], tokenizer, vocab, cfg, padding_value=tokenizer.pad_token_id) #


# %%



def logit_lens_layer(resid, answers, log_probs = False):
    logit_est = model.unembed(model.ln_final(resid.to(model.W_U.device))) # (batch, vocab)
    return probs_on_answer(logit_est, answers, log_probs = log_probs)
   

def logit_lens(cache, answers):
    probs_per_layer = [logit_lens_layer(resid, answers) for resid in cache.values()]
    probs_per_layer = torch.stack(probs_per_layer, dim=1)
    return probs_per_layer



def layer_prob_plot(prob_dict : dict[str, Float[Tensor, "batch layers"]], **kwargs):
# Create x-axis values (layer numbers)
    title = kwargs.get("title", "Layer-wise probability")
    # Create the plot
    plt.figure(figsize=(10, 6))

    for (name, probs) in prob_dict.items():
        probs = probs.float().cpu()
        probs_mean = probs.mean(dim=0)
        ci_95 = 1.96 * probs.std(dim=0) / np.sqrt(probs.shape[0])  
        layers = np.arange(len(probs_mean))

    
        # Plot English probabilities with confidence interval
        plt.plot(layers, probs_mean, linewidth=2, label=name)
        plt.fill_between(layers, 
                        probs_mean - ci_95,
                        probs_mean + ci_95,
                        alpha=0.2)
    plt.title(title)

    # Customize the plot
    plt.xlabel('Layer')
    plt.ylabel('Probability')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()

    # Show the plot
    plt.show()

# %%

layer_prob_plot({dest: logit_lens(cache, answers), "en": logit_lens(cache, answers_en)}, title = f"{src} -> {dest} model: {cfg.model_name}")

#  %%
# def all_t():
#     for name, tensor in globals().items():
#         try:
#             print(name, tensor.shape)
#         except:
#             continue


# # %%
# if 'logits' in locals() or 'logits' in globals():
#     del logits

# # Create a list of keys first, then iterate over that list
# if 'cache' in locals() or 'cache' in globals():
#     cache_keys = list(cache.keys())  # Create a static list of keys
#     for key in cache_keys:  # Iterate over the static list instead of the dict
#         del cache[key]
#     del cache
# torch.cuda.empty_cache()
# import gc
# gc.collect()
