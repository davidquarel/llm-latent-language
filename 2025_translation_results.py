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
from src.llm import safe_tokenize
from utils.data import gen_lang_ids, results_dict_to_csv

from utils.plot import plot_ci_simple
from utils.config_argparse import try_parse_args
from utils.data import parse_word_list, gen_lang_ids, gen_ids
from utils.misc import ci, wilson_ci
from utils.tokenizer import AutoTLTokenizer

import src.wendler as wendler


from itertools import combinations, product, permutations
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
    devices : Optional[str] = "0,3"
    out : Optional[str] = "out_icml_2025"
    token_add_capitalization : bool = True
    token_add_prefixes : bool = True
    token_add_spaces : bool = True
    token_utf8_byte : bool = True

cfg = Config()
#cfg = try_parse_args(cfg)
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

def run_translations():
    LANGS = ['fr', 'de', 'ru', 'zh', 'es','en']
    output_results = []

    print("Computing translation probabilities for each language pair")

    df = pd.read_csv(cfg.dataset_path)

    mask = df['en'].isin(WORD_LIST)
    prompt_df, suffix_df = df[mask], df[~mask]

    for lang in LANGS:
        all_answers = dict([(lang, get_answer_tensor2(suffix_df[lang], tokenizer, vocab, cfg, padding_value=tokenizer.pad_token_id)) for lang in LANGS])
    
    runner = tqdm(permutations(LANGS, 2))

    for (src, dest) in runner:
        #df = wendler.load_data("data_wendler/langs", SimpleNamespace(src_lang=src, dest_lang=dest))
        prompt = gen_prompt(prompt_df, src, dest)
        common_suffixes = gen_common_suffixes(suffix_df[src], src, dest)
        kv_cache = gen_kv_cache(prompt, model)
        suffix_toks = safe_tokenize(common_suffixes, model)
    

        logits, cache = run_with_kv_cache(suffix_toks, 
                                    kv_cache, 
                                    model, 
                                    mini_batch_size = 32,
                                    last_seq = True)

        answers = all_answers[dest]
        preds = logits.argmax(dim=-1, keepdim=True).to(answers.device)
        correct = torch.any(preds == answers, dim=-1).sum().item()

        probs = probs_on_answer(logits, answers, padding_id = tokenizer.pad_token_id, log_probs=False)
        mean_loss = -torch.log(probs).mean().item()
    
        mean_probs = probs.mean().item()
        ci95 = 1.96 * probs.std().item() / np.sqrt(probs.shape[0])


        acc = correct / answers.shape[0]
        print(f"{src} -> {dest} Translated {correct}/{answers.shape[0]} correctly. Accuracy: {acc:.2%} Loss: {mean_loss:.2f} Probs {mean_probs:.2f} ± {ci95:.2f}")
        output_results.append({'src_lang': src, 
                                'dest_lang': dest, 
                                'latent_lang': None, 
                                'avg_prob': mean_probs, 
                                'sem95_error': ci95, 
                                'acc': acc})
    output_results = pd.DataFrame(output_results)
    return output_results
model_basename = cfg.model_name.split('/')[-1]
out_path = os.path.join(cfg.out, model_basename, f"translation_no_interv_latent.csv")

if __name__ == "__main__":

    output_results = run_translations()

    out_path = os.path.join(cfg.out, model_basename, f"translation_no_interv_latent.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    output_results.to_csv(out_path, index=False)
# %%
