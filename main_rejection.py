# %%
%load_ext autoreload
%autoreload 2
# %%
from imports import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
from src.prompt import gen_prompt, gen_common_suffixes, raw_tokenize, TokenizedSuffixesResult, find_all_tokens
from src.kv_cache import gen_kv_cache, run_with_kv_cache
from src.llm import suffix_preamble, run, measure_performance, proj
from src.intervention import Intervention
from src.tuned_lens_wrap import TunedLens, BatchTunedLens, load_tuned_lens
from src.constants import LANG2NAME, LANG_BANK
from src.logit_lens import logit_lens, logit_lens_batched, LangIdx
from utils.io import parse_word_list

from utils.plot import plot_ci_simple, plot_logit_lens, plot_latents
from utils.font import cjk, noto_sans, dejavu_mono
from utils.config_argparse import try_parse_args


from eindex import eindex
from collections import namedtuple
import warnings
import re
import math
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache
from transformer_lens.utils import test_prompt, to_numpy
from tuned_lens import TunedLens
from src.tuned_lens_wrap import BatchTunedLens
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
    dataset_path: str = "data/butanium_v2.tsv"
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

# %%

src_words = LANG_BANK[cfg.src_lang]
dest_words = LANG_BANK[cfg.dest_lang]
suffix_words = df[cfg.src_lang]


# %%
find_all_tokens('nuée', model.tokenizer.vocab, **cfg_dict)
# %%
if MAIN:
    prompt = gen_prompt(src_words = src_words,
                    dest_words = dest_words,
                    src_lang = cfg.src_lang, 
                    dest_lang = cfg.dest_lang,
                    num_examples=cfg.num_multi_shot)
    kv_cache = gen_kv_cache(prompt, model)
    suffixes = gen_common_suffixes(suffix_words,
                                    src_lang = cfg.src_lang,
                                    dest_lang = cfg.dest_lang)
    suffix_toks = raw_tokenize(suffixes, model)
    result = run_with_kv_cache(suffix_toks.input_ids, kv_cache, model)
    logits = eindex(result.logits, suffix_toks.indices, "batch [batch] vocab")
    translated = list(zip(df[cfg.src_lang][:10], model.tokenizer.convert_ids_to_tokens(logits.argmax(dim=-1))))
    
    print(tabulate(translated, headers=[f'{cfg.src_lang=}', f'{cfg.dest_lang=}']))
# %%

# %%
from torch.utils.data import DataLoader, TensorDataset, IterableDataset, Dataset

# %%

def build_lang_idx(df_col, lang, vocab, **kwargs):
    array = []
    word_list_key = kwargs.get('word_list_key', f'claude')
    for primary, word_list in df[[lang, f'{word_list_key}_{lang}']].values:
        row_list = parse_word_list(word_list)
        row_list.append(primary)
        tokens = [find_all_tokens(x, vocab, **cfg_dict) for x in row_list]
        try:
            idx = torch.unique(torch.cat(tokens))
        except:
            print(f'{i=}')
            print(f'{row=}')
            print(f'{row_list=}')
            print(f'{tokens=}')
            
        array.append(idx)
    return array





# Usage


# %%



    
# %%


# %%


# %%
def run(cfg):
    prompt = gen_prompt(src_words = src_words,
                        dest_words = dest_words,
                        src_lang = cfg.src_lang, 
                        dest_lang = cfg.dest_lang,
                        num_examples=cfg.num_multi_shot)
    kv_cache = gen_kv_cache(prompt, model)
    lang_idx = create_lang_idx(df, model.tokenizer.vocab, **cfg_dict)

    suffixes = gen_common_suffixes(suffix_words,
                                    src_lang = cfg.src_lang,
                                    dest_lang = cfg.dest_lang)
    suffix_toks = raw_tokenize(suffixes, model)
    probs =  logit_lens_batched(kv_cache, suffix_toks, model, lang_idx, batch_size = 8)
    return probs

#cfg = Config(token_add_prefixes=True, word_list_key="claude")
logprobs_lang = run(cfg).cpu()
# %%
def run_plot(probs):
    fig, ax = plt.subplots()
    plot_ci_simple(probs[0].cpu(), ax, dim=1, label=f'src={cfg.src_lang}')
    plot_ci_simple(probs[1].cpu(), ax, dim=1, label=f'latent={cfg.latent_lang}')
    plot_ci_simple(probs[2].cpu(), ax, dim=1, label=f'dest={cfg.dest_lang}')
    ax.legend()
    plt.show()
    

    
minimax = torch.topk(logprobs_lang[1].max(dim=-1).values, largest=False, k =6).indices
labels = list(df[["word_original", cfg.src_lang, cfg.dest_lang]].iloc[minimax].apply(
    lambda row: f"{row['word_original']} : {row[cfg.src_lang]} -> {row[cfg.dest_lang]}",
    axis=1
))
plot_latents(logprobs_lang[1][minimax], labels = labels)
    
# %%
#test_prompt(prompt + suffixes[86], "<0xE5>", model)






intervention = Intervention(cfg.intervention_func, range(model.cfg.n_layers))
# only_delete_book = torch.Tensor([3143, 8277,3769,18167, 7977,19773, 3971,3489,7232,14415,9554,12219]).int()
# to_delete = torch.Tensor([3143, 8277,       3769, 18167,7977,19773, 5183, 3489,9538,14415,2471,29871]).int()
# #to_delete = torch.Tensor([3143, 8277,3769,7977,19773,3971,3489,7232,14415,9554,12219,232,27981]).int()
only_delete_book = torch.Tensor([3143]).int()
to_delete = torch.Tensor([26163,623]).int()
logit_lens_clean =  logit_lens(prompt + suffixes[0], model)
logit_lens_no_book = logit_lens(prompt + suffixes[0], model, intervention = intervention, latent_ids = only_delete_book)
logit_lens_delete = logit_lens(prompt + suffixes[0], model, intervention=intervention, latent_ids=to_delete)

clean_logprobs = torch.log_softmax(logit_lens_clean, dim=-1)
delete_logprobs = torch.log_softmax(logit_lens_delete, dim=-1)
middle_logprobs = torch.log_softmax(logit_lens_no_book, dim=-1)

plot_logit_lens(clean_logprobs, model.tokenizer)
plot_logit_lens(middle_logprobs, model.tokenizer)
plot_logit_lens(delete_logprobs, model.tokenizer)

book_tgt = 10586
tokens_ids = torch.Tensor([3143, book_tgt]).int()
plot_latents(torch.stack([clean_logprobs[:, book_tgt].cpu(),
                          middle_logprobs[:, book_tgt].cpu(), 
              delete_logprobs[:, book_tgt].cpu()]),
             ["zh", "zh_no_english", "zh_no_english_or_space"])
# %%
print(f"zh {torch.exp(clean_logprobs[:, book_tgt].cpu()[-1])}")
print(f"zh_no_english {torch.exp(middle_logprobs[:, book_tgt].cpu())[-1]}")
print(f"zh_no_english_or_space {torch.exp(delete_logprobs[:, book_tgt].cpu()[-1])}")
# %%

# %%
