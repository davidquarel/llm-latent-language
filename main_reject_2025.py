# This script does the exact same experiments as the orignal paper https://arxiv.org/pdf/2402.10588
# but faster, in batched mode.

# %%
%load_ext autoreload
%autoreload 2
# %%
from imports import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
from src.prompt import gen_prompt, gen_common_suffixes, find_all_tokens
from src.kv_cache import gen_kv_cache, run_with_kv_cache
from src.intervention import Intervention
from src.constants import LANG_TO_NAME, LANG_BANK, LANGS_NO_SPACE
from src.llm import safe_tokenize
from utils.data import gen_lang_ids, results_dict_to_csv

from utils.plot import plot_ci_simple
from utils.config_argparse import try_parse_args
from utils.data import parse_word_list, gen_lang_ids, gen_ids
from utils.misc import ci
from utils.tokenizer import AutoTLTokenizer

from eindex import eindex
from collections import namedtuple
import warnings
import re
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache
from src.kv_cache import broadcast_kv_cache
from transformer_lens.utils import test_prompt
from types import SimpleNamespace
# Import GPT-2 tokenizer
#disable gradients
torch.set_grad_enabled(False)

from ast import literal_eval
from tabulate import tabulate
import warnings
from src.prompt import token_prefixes
# %%
@dataclass
class Config:
    seed: int = 42
    model_name: str = "meta-llama/Llama-2-7b-hf"
    # single_token_only: bool = False
    # multi_token_only: bool = False
    out_dir: str = './out_iclr'
    dataset_path: str = "data/butanium_v2.tsv"
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
    devices : Optional[str] = None # "0,1"

os.environ["CUDA_VISIBLE_DEVICES"] = "1,2"  # Makes only GPUs 0 and 1 visible

cfg = Config()
cfg = try_parse_args(cfg)
cfg_dict = asdict(cfg)
print(cfg_dict)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.makedirs(cfg.out_dir, exist_ok=True)


tokenizer = AutoTLTokenizer.from_pretrained(cfg.model_name)

@dataclass
class TokenizedSuffixesResult:
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None

    def to(self, *args, **kwargs):
        """Move all tensor attributes to the specified device/dtype."""
        return TokenizedSuffixesResult(
            input_ids=self.input_ids.to(*args, **kwargs) if self.input_ids is not None else None,
            attention_mask=self.attention_mask.to(*args, **kwargs) if self.attention_mask is not None else None,
            indices=self.indices.to(*args, **kwargs) if self.indices is not None else None
        )

# if tokenizer.unk_token_id is None:
#     tokenizer.unk_token_id = tokenizer.all_special_ids[-1]
# if tokenizer.pad_token_id is None:
#     tokenizer.pad_token_id = tokenizer.unk_token_id

# tokenizer.sep_token_id = safe_tokenize(" ", tokenizer).input_ids.item()
# tokenizer.sep_token = tokenizer.convert_ids_to_tokens(safe_tokenize(" ", tokenizer).input_ids.item())



vocab = tokenizer.get_vocab()
space_token = tokenizer.convert_ids_to_tokens(safe_tokenize(" ", tokenizer).input_ids.item())
# def load_model():
model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                        device=device,
                                                        n_devices = 2,
                                                        dtype = torch.bfloat16)


# %%
LANGS = ['en', 'zh', 'fr', 'es', 'de', 'ru']
keys = ['word_original'] + LANGS + [f'claude_{lang}' for lang in LANGS]

df = pd.read_csv(cfg.dataset_path, delimiter = '\t') 
df = df[keys]

prompt_df, suffix_df = df[:cfg.num_multi_shot], df[cfg.num_multi_shot:]
# %%
def gen_prompt(df, src_lang, dest_lang):

    # Chinese does not have spaces between words
    # and chinese tokens do not have a leading space
    src_space = "" if src_lang in LANGS_NO_SPACE else " "
    dest_space = "" if dest_lang in LANGS_NO_SPACE else " "

    src_words = df[src_lang].to_list()
    dest_words = df[dest_lang].to_list()

    prompt = ""
    for (src, dest) in list(zip(src_words, dest_words))[:-1]:
        prompt += f'{LANG_TO_NAME[src_lang]}: "{src_space}{src}" - {LANG_TO_NAME[dest_lang]}: "{dest_space}{dest}"\n'
    # Add the last example without the destination translation
    # dont need a space, as if present, will be included in the next word
    src = src_words[-1]
    prompt += f'{LANG_TO_NAME[src_lang]}: "{src_space}{src}" - {LANG_TO_NAME[dest_lang]}: "'

    return prompt
# %%
def gen_common_suffixes(src_words, 
                        src_lang = None, 
                        dest_lang = None):
    assert src_lang is not None, "Source language must be provided"
    assert dest_lang is not None, "Destination language must be provided"
    common_suffixes = []
    src_space = "" if src_lang in LANGS_NO_SPACE else " "
    
    for src_word in src_words:
        src_word = src_word.split('▁')[-1] # Remove leading space token if present
        suffix = f'{src_space}{src_word}" {LANG_TO_NAME[dest_lang]}: "'
        common_suffixes.append(suffix)
    return common_suffixes


def gen_answer_ids(df, dest_lang, tokenizer):
    #answer_list = []
    answer_list_ids = []
    for (word, synonyms) in df[[dest_lang, f"claude_{dest_lang}"]].values:
        answers = [word] + parse_word_list(synonyms)
        answers = answers + [f" {x}" for x in answers]
        #answer_list.append(answers)
        prefixes = set()
        for answer in answers:
            prefixes.update(set(token_prefixes(answer)))


        answer_id = safe_tokenize(list(prefixes), tokenizer).input_ids[:, 0]
        answer_id = torch.unique(answer_id)
        answer_list_ids.append(answer_id)

    all_answer_ids = torch.nn.utils.rnn.pad_sequence(answer_list_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    return all_answer_ids



test_cfg = SimpleNamespace(src_lang = 'en', dest_lang = 'de')

prompt = gen_prompt(prompt_df, test_cfg.src_lang, test_cfg.dest_lang)
common_suffixes = gen_common_suffixes(suffix_df[test_cfg.src_lang], test_cfg.src_lang, test_cfg.dest_lang)

kv_cache = gen_kv_cache(prompt, model)
suffix_toks = safe_tokenize(common_suffixes, model)
logits = run_with_kv_cache(suffix_toks.input_ids, kv_cache, model, attention_mask = suffix_toks.attention_mask, names_filter = []).logits
suffix_toks = suffix_toks.to(logits.device)
logits_last_seq = eindex(logits, suffix_toks.indices, "batch [batch] dvocab -> batch dvocab")
#prediction = torch.argmax(logits_last_seq, dim=-1)
#list(zip(tokenizer.convert_ids_to_tokens(prediction), suffix_df[test_cfg.dest_lang].to_list()))
all_answer_ids = gen_answer_ids(suffix_df, test_cfg.dest_lang, tokenizer)

probs = torch.softmax(logits_last_seq, dim=-1)
#probs_on_answer[batch][i] = probs[batch][all_answer_ids[batch][i]]

probs_on_answer = eindex(probs, all_answer_ids, "batch [batch seq]") #(batch,), gives per translation prompt probability mass on any valid answer token
assert probs_on_answer.shape == all_answer_ids.shape
mask = all_answer_ids == tokenizer.pad_token_id
probs_on_answer[mask] = 0  
probs_on_answer = probs_on_answer.sum(dim=-1) #average over all tokens
print(probs_on_answer)


# %%




# %%
subspace = model.W_U.T[all_answer_ids]
subspace[all_answer_ids == tokenizer.pad_token_id] = 0

# %%
