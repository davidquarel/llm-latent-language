# This script does the exact same experiments as the orignal paper https://arxiv.org/pdf/2402.10588
# but faster, in batched mode.

# %%
try:
    %load_ext autoreload
    %autoreload 2
except:
    pass
# %%
from imports import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
from src.prompt import gen_prompt, gen_common_suffixes, get_answer_tensor2, get_valid_answer
from src.kv_cache import gen_kv_cache, run_with_kv_cache
from src.intervention import Intervention
from src.constants import LANG_TO_NAME, LANG_BANK, LANGS_NO_SPACE, LANGS
from src.llm import safe_tokenize, proj, proj_batched
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
    try:
        print(f"Loaded {model.name_or_path}") 
        loaded_model = True
    except:
        print(f"Failed to load {cfg.model_name}")
print("Devices:", cfg.devices)

# %%

def num_correct(logits, answers):
    preds = logits.argmax(dim=-1, keepdim=True).to(answers.device)
    correct = torch.any(preds == answers, dim=-1).sum()
    return correct.item()

def loss_on_answer(logits, answers, padding_id = -1):
    probs = torch.softmax(logits, dim=-1).float().cpu()
    probs_on_answer = eindex(probs, answers.cpu(), "batch [batch seq] -> batch seq")
    probs_on_answer[answers == padding_id] = 0
    probs_on_answer = probs_on_answer.sum(dim=-1) # (batch,) sum over valid answers
    return -torch.log(probs_on_answer)


from itertools import combinations, product
from transformer_lens.utils import test_prompt


lang_pairs = list(product(LANGS, repeat=2))


acc_dict = {}

src, dest = 'zh', 'fr'
src_i, dest_i = 'de', 'ru'


df = pd.read_csv("data_wendler/merge_inner.csv")
df = df.rename(columns={'word_original': 'en'})
word_list = ['cloud', 'mountain', 'moon', 'flower']
mask = df['en'].isin(word_list)
prompt_df, suffix_df = df[mask], df[~mask]
clean_prefix = gen_prompt(prompt_df, src, dest)
interv_prefix = gen_prompt(prompt_df, src_i, dest_i)

def suffix_format(src_word, src_lang,dest_lang):
    src_space = "" if src_lang in LANGS_NO_SPACE else " "
    return f'{src_space}{src_word}" - {LANG_TO_NAME[dest_lang]}: "'

clean_row = suffix_df.loc[df['en'] == 'eight']
interv_row = suffix_df.loc[df['en'] == 'hand']

clean_src_word = clean_row[src].item()
clean_dest_word = clean_row[dest].item()

interv_src_word = interv_row[src_i].item()
interv_dest_word = interv_row[dest_i].item()

clean_latent_word = clean_row['en'].item()
interv_latent_word = interv_row['en'].item()

clean_suffix = suffix_format(clean_src_word , src, dest)
interv_suffix = suffix_format(interv_src_word , src_i, dest_i)

clean_prompt = clean_prefix + clean_suffix
interv_prompt = interv_prefix + interv_suffix

print(clean_prompt)
print(interv_prompt)
print("========")

test_prompt(clean_prompt, clean_dest_word, model, prepend_space_to_answer=False)
print("===")
test_prompt(interv_prompt, interv_dest_word, model, prepend_space_to_answer=False)

def logit_lens(prompt, dest_word, latent_word):


    logit, cache = model.run_with_cache(prompt, names_filter = [f"blocks.{x}.hook_resid_post" for x in range(model.cfg.n_layers)])
    pre_logits = [model.unembed(model.ln_final(x[0, -1].to(model.W_U.device))) for x in cache.values()] 
    dest_id = tokenizer.encode(dest_word, return_tensors="pt", add_special_tokens=False).item()
    clean_id = tokenizer.encode(latent_word, return_tensors="pt", add_special_tokens=False).item()
    pre_logits = torch.stack(pre_logits, dim=0)
    dest_logprobs = torch.log_softmax(pre_logits, dim=-1)[:, dest_id]
    latent_logprobs = torch.log_softmax(pre_logits, dim=-1)[:, clean_id]

    plt.plot(dest_logprobs.float().cpu(), label="dest")
    plt.plot(latent_logprobs.float().cpu(), label="latent")
    plt.legend()
    plt.show()


# %%

#logits, cache = model.run_with_cache(prompt_fr_zh)
#test_prompt("The cat sat on the", "mat", model, prepend_space_to_answer=False)
#test_prompt("The cat sat on the", "mat", model, prepend_space_to_answer=False)

start_layer, end_layer = 13, 18
logit, cache = model.run_with_cache(clean_prompt, names_filter = ["block.{x}.hook_resid_post" for x in range(start_layer, end_layer+1)])

def ablate(
    resid: Float[Tensor, "batch seq d_model"],
    hook: HookPoint
) -> Float[Tensor, "batch seq d_model"]:
    subspace = model.W_U
    resid_proj = proj(resid, hook)
    

    # modify resid (can be inplace)
    return resid

# %%
    print(f"Translating {src} -> {dest}")
    
    
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
mask = df['en'].isin(word_list)
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

    softmax_fn = torch.log_softmax if log_probs else torch.softmax
    sum_fn = torch.logsumexp if log_probs else torch.sum
    zero = float('-inf') if log_probs else 0

    probs_est = softmax_fn(logit_est, dim=-1) # (batch, vocab)
    on_answer = eindex(probs_est, answers, "batch [batch seq] -> batch seq")
    on_answer[answers == tokenizer.pad_token_id] = zero
    on_answer = sum_fn(on_answer, dim=-1) # (batch,)
    return on_answer

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
