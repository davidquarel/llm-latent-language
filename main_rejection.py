# %%
%load_ext autoreload
%autoreload 2
# %%
from imports import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
from src.prompt import gen_prompt, gen_common_suffixes, safe_tokenize, TokenizedSuffixesResult, find_all_tokens
from src.kv_cache import gen_kv_cache, run_with_kv_cache
from src.llm import suffix_preamble, run, measure_performance
from src.intervention import Intervention
from utils.plot import plot_ci_simple
from utils.config_argparse import try_parse_args
from src.constants import LANG_TO_NAME, LANG_BANK
from eindex import eindex
from collections import namedtuple
import warnings
import re
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache

from transformer_lens.utils import test_prompt
# Import GPT-2 tokenizer
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
#disable gradients
torch.set_grad_enabled(False)

from ast import literal_eval
from tabulate import tabulate
# %%
@dataclass
class Config:
    seed: int = 42
    src_lang: str = 'fr'
    dest_lang: str = 'zh'
    latent_lang: str = 'en'
    model_name: str = "meta-llama/Llama-2-7b-hf" # 'meta-llama/Meta-Llama-3-8B'
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
    # intervention_func : str = 'hook_reject_subspace'
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
    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# %%
#LOAD_MODEL = False
if 'LOAD_MODEL' not in globals():
    model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device,
                                                            dtype = torch.float16)
    tokenizer = model.tokenizer
    tokenizer_vocab = model.tokenizer.get_vocab() # type: ignore
    LOAD_MODEL = False    
# %%
df = pd.read_csv(cfg.dataset_path, delimiter = '\t') 

src_multishot = LANG_BANK[cfg.src_lang]
dest_multishot = LANG_BANK[cfg.dest_lang]
dest_words = df[cfg.src_lang]

prompt = gen_prompt(src_words = src_multishot,
                dest_words = dest_multishot,
                src_lang = cfg.src_lang, 
                dest_lang = cfg.dest_lang,
                num_examples=cfg.num_multi_shot)

# %%
def parse_word_list(s):
    # Remove the outer brackets and split by commas
    try:
        result = literal_eval(s)
        return result
    except:
        warnings.warn(f"Could not parse row: {s}")
        s = s.strip()[1:-1]
        items = re.split(r',\s*', s)
        
        result = []
        for item in items:
            # Remove surrounding quotes if present
            if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
                item = item[1:-1]
            # Handle apostrophes within words
            item = item.replace("'", "'")
            result.append(item)
    
        return result
# %%
def padded_tensor(tensor_list):
    # Find the maximum length and d_model
    max_length = max(tensor.size(0) for tensor in tensor_list)
    d_model = tensor_list[0].size(-1)
    K = len(tensor_list)
    
    # Create the output tensor directly
    result = torch.zeros((K, max_length, d_model), dtype=tensor_list[0].dtype, device=tensor_list[0].device)
    
    # Fill the output tensor
    for i, tensor in enumerate(tensor_list):
        result[i, :tensor.size(0)] = tensor
    
    return result

# %%
def sanity_check(df):
    
    src_multishot = LANG_BANK[cfg.src_lang]
    dest_multishot = LANG_BANK[cfg.dest_lang]
    dest_words = df[cfg.src_lang]
    
    prompt = gen_prompt(src_words = src_multishot,
                    dest_words = dest_multishot,
                    src_lang = cfg.src_lang, 
                    dest_lang = cfg.dest_lang,
                    num_examples=cfg.num_multi_shot)
    kv_cache = gen_kv_cache(prompt, model)
    suffixes = gen_common_suffixes(dest_words,
                                    src_lang = cfg.src_lang,
                                    dest_lang = cfg.dest_lang)
    suffix_toks = safe_tokenize(suffixes, model)
    result = run_with_kv_cache(suffix_toks.input_ids, kv_cache, model)
    
    intervention = Intervention("hook_batch_reject", range(model.cfg.n_layers))
    
    logits = eindex(result.logits, suffix_toks.indices, "batch [batch] vocab")
    translated = list(zip(df[cfg.src_lang][:10], model.tokenizer.convert_ids_to_tokens(logits.argmax(dim=-1))))
    
    print(tabulate(translated, headers=[f'{cfg.src_lang=}', f'{cfg.dest_lang=}']))

    print(list(zip(model.tokenizer.convert_ids_to_tokens(logits.argmax(dim=-1)), 
                   [x.item() for x in torch.softmax(logits, dim=-1).max(dim=-1).values])))
sanity_check(df)
# %%

from torch.utils.data import DataLoader, TensorDataset, IterableDataset, Dataset

# %%

def build_lang_idx(df, lang, model, cfg):
    array = []
    for primary, word_list in df[[lang, f'{cfg.word_list_key}_{lang}']].values:
        row_list = parse_word_list(word_list)
        row_list.append(primary)
        if lang not in ["zh", "ko", "jp"]:
            padded_list = [" " + x for x in row_list] + row_list
        else:
            padded_list = row_list
            
        all_tokens = safe_tokenize(padded_list, model).input_ids[:, 0]
        
        if lang in ["zh", "ko", "jp"]:
            leading_bytes = torch.Tensor(model.tokenizer.convert_tokens_to_ids([f'<0x{(x.encode("utf-8")[0]):X}>' for x in row_list]))
            all_tokens = torch.cat([leading_bytes.to(torch.long).to(device), all_tokens])
        
        all_tokens = torch.unique(all_tokens)
        array.append(all_tokens)
    return array

LangIdx = namedtuple('LangIdx', ['src_idx', 'latent_idx', 'dest_idx'],
                     defaults=[None, None, None])

def create_lang_idx(df, model, cfg):
    src_idx = build_lang_idx(df, cfg.src_lang, model, cfg)
    dest_idx = build_lang_idx(df, cfg.dest_lang,model, cfg)
    latent_idx = build_lang_idx(df, cfg.latent_lang,model,cfg)
    
    assert len(src_idx) == len(dest_idx), "Mismatch between src_idx and dest_idx lengths"
    assert len(src_idx) == len(latent_idx), "Mismatch between src_idx and latent_idx lengths"
    
    return LangIdx(src_idx, latent_idx, dest_idx)

# Usage


# %%

class MixedDataset(Dataset):
    def __init__(self, tensors: Tuple[torch.Tensor, ...], tensor_lists: Tuple[List[torch.Tensor], ...]):
        if not tensors and not tensor_lists:
            raise ValueError("Both tuples are empty")
        
        self.tensors = tensors
        self.tensor_lists = tensor_lists
        
        self.length = len(self.tensors[0]) if self.tensors else len(self.tensor_lists[0])
        
        # Ensure all data have the same length
        if not all(len(t) == self.length for t in self.tensors):
            raise ValueError("All tensors must have the same length")
        if not all(len(l) == self.length for l in self.tensor_lists):
            raise ValueError("All tensor lists must have the same length as tensors")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return (
            tuple(tensor[idx] for tensor in self.tensors),
            tuple(tensor_list[idx] for tensor_list in self.tensor_lists)
        )

    @staticmethod
    def collate_fn(batch):
        # Separate fixed-length tensors and variable-length tensor lists
        fixed_tensors, var_tensor_lists = zip(*batch)
        
        # Handle fixed-length tensors
        collated_fixed = tuple(torch.stack(items) for items in zip(*fixed_tensors))
        
        # Handle variable-length tensor lists
        collated_var = tuple(list(items) for items in zip(*var_tensor_lists))
        
        return collated_fixed + collated_var


    
# %%


# %%
@torch.no_grad
def logit_lens_with_cache(kv_cache : HookedTransformerKeyValueCache, 
                       suffix_toks : TokenizedSuffixesResult,
                       model,
                       lang_idx : LangIdx,
                       intervention=None,
                       **kwargs):
    
    device = next(model.parameters()).device
    
    batch_size = kwargs.get('batch_size', 1)
    use_tuned_lens = kwargs.get('use_tuned_lens', False)
    
    #return_float = kwargs.get('return_float', False)
    
    
    ALL_POST_RESID = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
    
    dataset = MixedDataset(suffix_toks, lang_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=MixedDataset.collate_fn)

    runner = tqdm(dataloader, total=len(suffix_toks.input_ids), desc="Computing logits", position=0, leave=True)
    
    all_src_probs = []
    all_latent_probs = []
    all_dest_probs = []    
    
    for i, (toks, toks_mask, toks_idx, src_idx, latent_idx, dest_idx) in enumerate(runner):
        
        latent_subspace = padded_tensor([model.unembed.W_U.T[lat] for lat in latent_idx])
        
        fwd_hooks = [] if intervention is None else intervention.fwd_hooks(model, rejection_subspaces = latent_subspace)
        _, cache = run_with_kv_cache(toks, kv_cache, model, fwd_hooks = fwd_hooks, names_filter = ALL_POST_RESID)
        
        latents = torch.stack(tuple(cache.values()), dim=1) # (batch, num_layers, seq, d_model)
        latents = eindex(latents, toks_idx, "batch num_layers [batch] d_model") # (batch, num_layers, d_model)
        
        if use_tuned_lens:
            approx_logits = model.tuned_lens(latents)
        else:
            approx_logits = model.unembed(latents)
            
        probs = torch.softmax(approx_logits, dim=-1) # (batch, num_layers, vocab)
        
        src_probs = eindex(probs, src_idx, "batch num_layers vocab") # (batch, num_layers)
        for batch, (src, latent, dest) in enumerate(zip(src_idx, latent_idx, dest_idx)):
            src_probs = probs[batch, :, src].sum(dim=-1)  # (n_layers)
            latent_probs = probs[batch, :, latent].sum(dim=-1)  # (n_layers)
            dest_probs = probs[batch, :, dest].sum(dim=-1)  # (n_layers)
            all_src_probs.append(src_probs)
            all_latent_probs.append(latent_probs)
            all_dest_probs.append(dest_probs)
        
        runner.update(batch_size)
        
    all_src_probs = torch.stack(all_src_probs, dim=-1) #(bs, n)
    all_latent_probs = torch.stack(all_latent_probs, dim=-1) #(bs, n)
    all_dest_probs = torch.stack(all_dest_probs, dim=-1) #(bs, n)
    all_probs = torch.stack([all_src_probs, all_latent_probs, all_dest_probs], dim=0) #(lang, bs, n)
    
    return all_probs

# %%
def plot_prob_flow(probs):
    fig, ax = plt.subplots()
    plot_ci_simple(probs[0].cpu(), ax, dim=1, label=f'src={cfg.src_lang}')
    plot_ci_simple(probs[1].cpu(), ax, dim=1, label=f'latent={cfg.latent_lang}')
    plot_ci_simple(probs[2].cpu(), ax, dim=1, label=f'dest={cfg.dest_lang}')
    ax.legend()
    plt.show()
# %%

def run_rejection(df, cfg):
    src_multishot = LANG_BANK[cfg.src_lang]
    dest_multishot = LANG_BANK[cfg.dest_lang]
    dest_words = df[cfg.src_lang]
    
    prompt = gen_prompt(src_words = src_multishot,
                    dest_words = dest_multishot,
                    src_lang = cfg.src_lang, 
                    dest_lang = cfg.dest_lang,
                    num_examples=cfg.num_multi_shot)
    kv_cache = gen_kv_cache(prompt, model)
    kv_cache.freeze()
    suffixes = gen_common_suffixes(dest_words,
                                    src_lang = cfg.src_lang,
                                    dest_lang = cfg.dest_lang)
    suffix_toks = safe_tokenize(suffixes, model)
    lang_idx = create_lang_idx(df, model, cfg)
    intervention = Intervention("hook_batch_reject_slow", range(model.cfg.n_layers))
    probs_clean = logit_lens_with_cache(kv_cache, suffix_toks, model, lang_idx, intervention=None, batch_size=cfg.batch_size)
    
    
    kv_cache = gen_kv_cache(prompt, model)
    kv_cache.freeze()
    probs_reject = logit_lens_with_cache(kv_cache, suffix_toks, model, lang_idx, intervention=intervention, batch_size=cfg.batch_size)
    
    plot_prob_flow(probs_clean)
    plot_prob_flow(probs_reject)
    
cfg.batch_size = 8
run_rejection(df, cfg)
# %%

# def run(df, cfg):
#     prompt = gen_prompt(src_words = src_words,
#                         dest_words = dest_words,
#                         src_lang = cfg.src_lang, 
#                         dest_lang = cfg.dest_lang,
#                         num_examples=cfg.num_multi_shot)
#     kv_cache = gen_kv_cache(prompt, model)
#     lang_idx = create_lang_idx(df, model.tokenizer.vocab, **cfg_dict)

#     suffixes = gen_common_suffixes(suffix_words,
#                                     src_lang = cfg.src_lang,
#                                     dest_lang = cfg.dest_lang)
#     suffix_toks = safe_tokenize(suffixes, model)
#     probs =  logit_lens_batched(kv_cache, suffix_toks, model, lang_idx, batch_size = 8)
#     return probs

# cfg = Config(token_add_prefixes=True, word_list_key="claude")
# probs = run(cfg)
# %%

#plot_prob_flow(probs)
# %%
#test_prompt(prompt + suffixes[86], "<0xE5>", model)