# %%
%load_ext autoreload
%autoreload 2
# %%
from imports import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))


# =========================

from eindex import eindex
from collections import namedtuple
import warnings
import re
import math
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache
from transformer_lens.utils import test_prompt, to_numpy
from itertools import combinations

# ==== Custom Libraries ====
from src.prompt import gen_prompt, gen_common_suffixes, raw_tokenize, TokenizedSuffixesResult, find_all_tokens
from src.kv_cache import gen_kv_cache, run_with_kv_cache
from src.llm import suffix_preamble, run, measure_performance, proj
from src.intervention import Intervention
from src.tuned_lens_wrap import TunedLens, BatchTunedLens, load_tuned_lens
from src.constants import LANG2NAME, LANG_BANK
from src.logit_lens import logit_lens, logit_lens_batched, LangIdx

from utils.plot import plot_ci_simple, plot_logit_lens, plot_latents
from utils.font import cjk, noto_sans, dejavu_mono
from utils.config_argparse import try_parse_args
from utils.io import parse_word_list, create_lang_idx


# Import GPT-2 tokenizer
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#disable gradients
torch.set_grad_enabled(False)

from ast import literal_eval
from tabulate import tabulate
MAIN = __name__ == '__main__'
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
# %%
@dataclass
class Config:
    seed: int = 42
    src_lang: str = 'fr'
    dest_lang: str = 'de'
    latent_lang: str = 'en'
    model_name: str = "meta-llama/Llama-2-7b-hf" #'gpt2' #'meta-llama/Meta-Llama-3-8B'
    # single_token_only: bool = False
    # multi_token_only: bool = False
    # out_dir: str = './visuals'
    dataset_path: str = "data/butanium_v3.tsv"
    debug: bool = True
    num_multi_shot : int = 5
    token_add_spaces: bool = True
    token_add_leading_byte: bool = True
    token_add_prefixes : bool = False
    token_add_capitalization : bool = True
    # dataset_filter_correct : bool = True
    intervention_func : str = 'hook_reject_subspace'
    # log_file : str = 'DUMMY_NAME'
    # metric : str = 'p_alt'
    # metric_goal : str = 'max'
    # use_reverse_lens : bool = False
    # rev_lens_scale : bool = 1
    # only_compute_stats : bool = False
    word_list_key : str = 'claude'
    cache_prefix : bool = True

cfg = Config()
cfg = try_parse_args(cfg)
cfg_dict = asdict(cfg)
    


# %%
#LOAD_MODEL = False
if 'LOAD_MODEL' not in globals():
    model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device, 
                                                            dtype = torch.float16)
    tokenizer = model.tokenizer
    tokenizer_vocab = model.tokenizer.get_vocab() # type: ignore
    LOAD_MODEL = False
    # tuned_lens = load_tuned_lens(model)
    # batched_tuned_lens = BatchTunedLens(tuned_lens).to(device)
# %%
df = pd.read_csv(cfg.dataset_path, delimiter = '\t') 


def build_lang_idx(df, lang, vocab, **kwargs):
    array = []
    word_list_key = kwargs.get('word_list_key', 'claude')
    for primary, word_list in df[[lang, f'{word_list_key}_{lang}']].values:
        row_list = parse_word_list(word_list)
        row_list.append(primary)
        tokens = [find_all_tokens(x, vocab, **kwargs) for x in row_list]
        try:
            idx = torch.unique(torch.cat(tokens))
        except:
            print(f"Could not concatenate tokens for row: {row_list}")
            print(f'{row_list=}')
            print(f'{tokens=}')
            
        array.append(idx)
    return array
# %%

lang_idx = create_lang_idx(df[1:2], model.tokenizer.vocab, **cfg_dict)


# %%

src_words = LANG_BANK[cfg.src_lang]
dest_words = LANG_BANK[cfg.dest_lang]
suffix_words = df[cfg.src_lang]

# %%
def lang_reject(prompt, latent_ids,target_ids):
    
    intervention = Intervention("hook_reject_subspace", range(model.cfg.n_layers))
    all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
    fwd_hooks = intervention.fwd_hooks(model, latent_ids = latent_ids)
    
    logits_clean = model(prompt)[0, -1]
    with model.hooks(fwd_hooks = fwd_hooks):
        logits_reject = model(prompt, names_filter=all_post_resid)[0, -1]
        
    logit_diff = (logits_clean - logits_reject)[target_ids].mean()
    return logit_diff

def compute_mean_and_ci_gaussian(tensor):
    mean = torch.mean(tensor)
    std_error = torch.std(tensor, unbiased=True) / torch.sqrt(torch.tensor(tensor.numel()))
    
    # For 95% confidence interval in a Gaussian distribution, we use 1.96
    # (approximately 2 standard deviations)
    z_score = 1.96
    margin_of_error = z_score * std_error
    
    return mean.item(), margin_of_error
# %%

# def gen_lang_reject_heatmap(model, languages):
#     results = {}
#     combos = list(combinations(languages, 3))
#     for src_lang, latent_lang, dest_lang in combos:
        
#         prompt_prefix = gen_prompt(src_words = LANG_BANK[src_lang],
#                             dest_words = LANG_BANK[dest_lang],
#                             src_lang = src_lang, 
#                             dest_lang = dest_lang,
#                             num_examples=5)
#         lang_idx = create_lang_idx(df, model.tokenizer.vocab, **cfg_dict)
#         suffixes = gen_common_suffixes(suffix_words,
#                                         src_lang = src_lang,
#                                         dest_lang = dest_lang)

#         intervention = Intervention(cfg.intervention_func, range(model.cfg.n_layers))
        
        
#         logit_diffs = []
#         for i in range(len(suffixes)):
#             suffix = suffixes[i]
#             prompt = model.tokenizer.encode(prompt_prefix + suffix, return_tensors="pt")
#             latent_ids = lang_idx.latent_ids[i]
#             logit_diff = lang_reject(prompt, latent_ids, lang_idx.dest_ids[i])
#             logit_diffs.append(logit_diff)
#         logit_diff_mean, logit_diff_err = compute_mean_and_ci_gaussian(torch.Tensor(logit_diffs))
        
#         results[(src_lang, latent_lang, dest_lang)] = (logit_diff_mean, logit_diff_err)
#     return results




