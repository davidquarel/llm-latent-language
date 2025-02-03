# This script does the exact same experiments as the orignal paper https://arxiv.org/pdf/2402.10588
# but faster, in batched mode.

# %%
try:
    %load_ext autoreload
    %autoreload 2
except:
    pass
# %%
from pandas import DataFrame, Series
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
    devices : Optional[str] = "0"

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


def gen_df(word_list = ['cloud', 'mountain', 'moon', 'two']):
    df = pd.read_csv("data_wendler/merge_inner.csv")
    df = df.rename(columns={'word_original': 'en'})
    mask= df['en'].isin(word_list)
    prefix_df, suffix_df = df[mask], df[~mask]
    return prefix_df, suffix_df


def suffix_format(src_word, src_lang,dest_lang) -> str:
    src_space = "" if src_lang in LANGS_NO_SPACE else " "
    return f'{src_space}{src_word}" - {LANG_TO_NAME[dest_lang]}: "'



# %%




# %%
def ablate(
    resid: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    ablate_word: str,
) -> Float[Tensor, "batch seq d_model"]:
    clean_id = tokenizer.encode(ablate_word, return_tensors="pt", add_special_tokens=False).to(model.W_U.device)
    subspace = model.W_U.T[clean_id]
    resid_proj = proj(resid[:, -1], subspace)
    resid[:, -1] = resid[:, -1] - resid_proj

    # modify resid (can be inplace)
    return resid


def replace(
    resid: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    word_delete: str,
    resid_splice : Float[Tensor, "batch d_model"],
    scale: float = 1.0,
) -> Float[Tensor, "batch seq d_model"]:
    id_old = tokenizer.encode(word_delete, return_tensors="pt", add_special_tokens=False)
    S_old = model.W_U.T[id_old]
    old = proj(resid[:, -1], S_old)
    resid[:, -1] = resid[:, -1] - old + scale * resid_splice
    # modify resid (can be inplace)
    return resid


# %%
def logit_lens(logits, cache, answers, ax, **kwargs) -> None:
    pre_logits: list[Any] = [model.unembed(model.ln_final(x[0, -1].to(model.W_U.device))) for x in cache.values()]
    pre_logits = torch.stack(pre_logits, dim=0)
    dest_logprobs = torch.log_softmax(pre_logits, dim=-1)
    probs = torch.softmax(dest_logprobs[-1], dim=-1)
    for name, word in answers.items():
        dest_id = tokenizer.encode(word, return_tensors="pt", add_special_tokens=False)[0][0]
        #print(f"{name}: {dest_logprobs}")
        ax.plot(dest_logprobs[:, dest_id].float().cpu(), label=name)
        ax.annotate(f"{probs[dest_id]:.3f},{dest_logprobs[-1, dest_id]:.3f}", xy=(len(cache)-1, dest_logprobs[-1, dest_id].float().cpu()), 
            xytext=(5, 2), textcoords='offset points', fontsize=12, color='red')
        # for i, logprob in enumerate(dest_logprobs[:, dest_id].float().cpu()):
        #     ax.annotate(f"{logprob:.2f}", xy=(i, logprob), xytext=(5, 2), textcoords='offset points', fontsize=8, color='blue')
    ax.set_title(kwargs.get("title", "Logits for each word"))
    ax.legend()

prefix_df, suffix_df = gen_df()
def test_word_list(prefix_df, suffix_df, trans_list = ['house', 'foot']):
    dc = {}
    fig, axes = plt.subplots(2, 2, figsize=(20, 12))

    answers = {'en_house' : 'house',
               'fr_house' : 'maison',
                'de_house' : 'Haus',
                'en_foot' : 'foot',
                'fr_foot' : 'pied',
                'de_foot' : 'Fuß'}

    for i, (src, dest) in enumerate([('zh', 'fr'), ('ru', 'de')]):
        for j,en_word in enumerate(trans_list):
            row   = suffix_df.loc[suffix_df['en'] == en_word]
            suffix = suffix_format(row[src].item() , src, dest)
            answer = row[dest].item()
            prefix = gen_prompt(prefix_df, src, dest)
            prompt = prefix + suffix
            test_prompt(prompt, answer, model, prepend_space_to_answer=False)

            logits, cache = model.run_with_cache(prompt, names_filter=[f"blocks.{x}.hook_resid_post" for x in range(model.cfg.n_layers)])

            hooks = [(f"blocks.{x}.hook_resid_post", lambda resid, hook : ablate(resid, hook, en_word)) for x in range(model.cfg.n_layers)]
            with model.hooks(fwd_hooks=hooks):
                logits, cache = model.run_with_cache(prompt, names_filter=[f"blocks.{x}.hook_resid_post" for x in range(model.cfg.n_layers)])
    
            logit_lens(logits, cache, answers, axes[i,j], title=f"reject {en_word} {src} -> {dest}, {en_word}, prob ")
            axes[i,j].grid()
    plt.tight_layout()
    
test_word_list(prefix_df, suffix_df)

# # %%
# fwd_hooks_clean = [(f"blocks.{x}.hook_resid_post", lambda resid, hook : ablate(resid, hook, dc['lat'])) for x in range(model.cfg.n_layers)]
# fwd_hooks_alt = [(f"blocks.{x}.hook_resid_post", lambda resid, hook : ablate(resid, hook, dc['lat_alt'])) for x in range(model.cfg.n_layers)]

# Create the subplots


# # Run the original sequence with subplots

# logit_lens(cache, dc, axes[0, 0], title="clean")

# logits, cache = model.run_with_cache(interv_prompt, names_filter=[f"blocks.{x}.hook_resid_post" for x in range(model.cfg.n_layers)])
# logit_lens(cache, dc, axes[0, 1], title='alt')

# # Run the ablation sequence
# with model.hooks(fwd_hooks=fwd_hooks_clean):
#     logits, cache = model.run_with_cache(clean_prompt, names_filter=[f"blocks.{x}.hook_resid_post" for x in range(model.cfg.n_layers)])
# logit_lens(cache, dc, axes[1, 0], title='clean_ablate')

# with model.hooks(fwd_hooks=fwd_hooks_alt):
#     logits, cache = model.run_with_cache(interv_prompt, names_filter=[f"blocks.{x}.hook_resid_post" for x in range(model.cfg.n_layers)])
# logit_lens(cache, dc, axes[1, 1], title='alt_ablate')

# plt.tight_layout()
# plt.show()


# %%
trans_list = ['house', 'foot']
(src, dest), (src_s, dest_s) = ('de', 'fr'), ('de', 'fr')

def init(src, dest, en_word):
    row   = suffix_df.loc[suffix_df['en'] == en_word]
    suffix = suffix_format(row[src].item() , src, dest)
    answer = row[dest].item()
    prefix = gen_prompt(prefix_df, src, dest)
    prompt = prefix + suffix
    return prompt, answer

def get_subspace(word):
    word_id = tokenizer.encode(word, return_tensors="pt", add_special_tokens=False).item()
    S_old = model.W_U.T[word_id]
    assert S_old.shape == torch.Size([model.cfg.d_model]), f"Expected shape {model.cfg.d_model}, got {S_old.shape}"
    return S_old


# def proj(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
#     """
#     Projects vector x onto vector y
    
#     Args:
#         x: Input vector to be projected
#         y: Vector to project onto
        
#     Returns:
#         Projection of x onto y
#     """
#     # Compute dot product of x and y
#     dot_product = torch.sum(x * y, dim=-1, keepdim=True)
    
#     # Compute squared magnitude of y
#     y_magnitude_squared = torch.sum(y * y, dim=-1, keepdim=True)
    
#     # Compute the projection
#     projection = (dot_product / y_magnitude_squared) * y
    
#     return projection




def zero(
    resid: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
) -> Float[Tensor, "batch seq d_model"]:
    resid[:, -1] = torch.zeros_like(resid[:, -1])
    return resid

answers = {'en_house': 'house',
          'fr_house': 'maison',
          'de_house': 'Haus',
          'en_foot': 'foot',
          'fr_foot': 'pied',
          'de_foot': 'Fuß'}

en_word, en_word_alt = answers['en_house'], answers['en_foot']
clean_prompt, _ = init(src, dest, en_word)
interv_prompt, _ = init(src_s, dest_s, en_word_alt)

test_prompt(clean_prompt, answers['fr_house'], model, prepend_space_to_answer=False)
print("===")
test_prompt(interv_prompt, answers['fr_foot'], model, prepend_space_to_answer=False)



# zero_hook = [(f"blocks.{x}.hook_resid_post", lambda resid, hook : zero(resid, hook)) for x in range(30,model.cfg.n_layers)]
# with model.hooks(fwd_hooks=zero_hook):
#     zero_logits, zero_cache = model.run_with_cache(clean_prompt, names_filter=all_resid)
# logit_lens(zero_logits, zero_cache, answers, axes[1,1], title=f"zero {src} -> {dest}, {en_word}")

def replace(
    resid: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    word_delete: str,
    resid_splice : Float[Tensor, "batch d_model"],
    scale: float = 1.0,
) -> Float[Tensor, "batch seq d_model"]:
    S_old = get_subspace(word_delete)
    proj_S_old = proj(resid[:, -1], S_old)
    new_resid = resid[:, -1] - proj_S_old + scale * resid_splice
    print(resid[:, -1].shape, proj_S_old.shape, resid_splice.shape, new_resid.shape)
    resid[:, -1] = new_resid
    # modify resid (can be inplace)
    return resid

shift_hooks = []
for n in range(11, 29):
    name = f"blocks.{n}.hook_resid_post"
    S_del = get_subspace(en_word_alt)
    _, c = proj(interv_cache[name][:, -1], S_del, return_coeff=True)
    splice = c * S_del
    hook = lambda resid, hook: replace(resid,hook, en_word, splice, scale=0.11)
    shift_hooks.append((name, hook))

with model.hooks(fwd_hooks=shift_hooks):
    shift_logits, shift_cache = model.run_with_cache(clean_prompt, names_filter=all_resid)
logit_lens(shift_logits, shift_cache, answers, axes[1,0], title=f"clean shift {en_word} {src} -> {dest}, {en_word}")



            # start, end = 13, 18
            # #clean run
            

            # steer_hook = []
            # en_word_id = tokenizer.encode(en_word, return_tensors="pt", add_special_tokens=False)
            # en_word_subspace = model.W_U.T[en_word_id]
            # for layer in range(13, 18):
            #     clean_resid = clean_cache[f"blocks.{layer}.hook_resid_post"][:, -1]
            #     resid_splice = proj(clean_resid, en_word_subspace)
            #     steer_hook.append((f"blocks.{layer}.hook_resid_post", lambda resid, hook : ablate(resid, hook, en_word, resid_splice)))


            # # hooks = [(f"blocks.{x}.hook_resid_post", lambda resid, hook : ablate(resid, hook, en_word)) for x in range(model.cfg.n_layers)]
            # with model.hooks(fwd_hooks=steer_hook):
            #     steered_logits, steered_cache = model.run_with_cache(prompt, names_filter=[f"blocks.{x}.hook_resid_post" for x in range(model.cfg.n_layers)])
    
            # logit_lens(steered_logits, steered_cache, answers, axes[i,j], title=f"reject {en_word} {src} -> {dest}, {en_word}, prob ")
            # axes[i,j].grid()
# %%
def get_concept_strength(prompt_clean, prompt_target, en_word_target, layers):
    # Get clean prompt projections
    _, cache_clean = model.run_with_cache(prompt_clean)
    clean_projections = {
        layer: proj(cache_clean[f"blocks.{layer}.hook_resid_post"][:, -1], 
                   get_subspace(en_word_target))
        for layer in layers
    }
    
    # Get target concept projections
    _, cache_target = model.run_with_cache(prompt_target) 
    target_projections = {
        layer: proj(cache_target[f"blocks.{layer}.hook_resid_post"][:, -1],
                   get_subspace(en_word_target))
        for layer in layers
    }
    
    # Calculate layer-wise scaling factors
    strength_factors = {
        layer: (target_projections[layer] - clean_projections[layer]).norm(p=2, dim=-1)
                / clean_projections[layer].norm(p=2, dim=-1)
        for layer in layers
    }
    
    return strength_factors

strength_factors = get_concept_strength(clean_prompt, interv_prompt, en_word_alt, range(model.cfg.n_layers))
# %%

def print_topk_tokens(
    logits: torch.Tensor, 
    tokenizer: AutoTLTokenizer,
    top_k: int = 10,
    title: str = "Top tokens"
) -> None:
    """
    Prints the top-k tokens with their logits, probabilities, and log probabilities.
    
    Args:
        logits: Tensor of shape (vocab_size,) or (1, vocab_size)
        tokenizer: Tokenizer with to_string() method
        top_k: Number of top tokens to display
        title: Header for the output table
    """
    # Ensure proper tensor shape
    logits = logits.squeeze().detach().cpu()
    if logits.dim() != 1:
        raise ValueError("Logits must be 1D or squeezable to 1D")
    
    # Calculate probabilities
    probs = logits.softmax(dim=-1)
    log_probs = logits.log_softmax(dim=-1)
    
    # Get top-k tokens
    top_probs, top_indices = torch.topk(probs, k=top_k)
    top_logits = logits[top_indices]
    top_log_probs = log_probs[top_indices]
    
    # Prepare table data
    headers = ["Rank", "Token", "Logit", "Prob", "Log Prob"]
    rows = []
    
    for rank, (idx, prob, logit, log_prob) in enumerate(zip(
        top_indices, top_probs, top_logits, top_log_probs
    )):
        token_str = tokenizer.convert_ids_to_tokens(idx.item()).strip()
        rows.append([
            rank,
            repr(token_str),
            f"{logit.item():.2f}",
            f"{prob.item():.2%}",
            f"{log_prob.item():.2f}"
        ])
    
    # Print formatted output
    print(f"\n=== {title} ===")
    print(tabulate(rows, headers=headers, tablefmt="pretty"))

# %%
def concept_steering_hook(
    resid: Float[Tensor, "batch seq d_model"],
    hook: HookPoint,
    en_word_target: str,
    strength_factors: dict,
    layer: int
):
    concept_vec = get_subspace(en_word_target)
    scale = strength_factors[layer].to(resid.device)
    
    # Add scaled concept vector to final token's residual
    resid[:, -1] += scale * concept_vec * 0.2
    return resid

from functools import partial

# Calculate optimal strength
layers = range(12, 20)
strength_factors = get_concept_strength(clean_prompt, interv_prompt, "foot", layers)

# Create intervention hooks
strength_hooks = [
    (f"blocks.{layer}.hook_resid_post",
     partial(concept_steering_hook, 
             en_word_target="foot",
             strength_factors=strength_factors,
             layer=layer))
    for layer in layers
]
# Run model with intervention


fig, axes = plt.subplots(2, 2, figsize=(20, 12))

all_resid = [f"blocks.{x}.hook_resid_post" for x in range(model.cfg.n_layers)]

clean_logits, clean_cache = model.run_with_cache(clean_prompt, names_filter=all_resid)
logit_lens(clean_logits, clean_cache, answers, axes[0,0], title=f"clean {src} -> {dest}, {en_word}")

interv_logits, interv_cache = model.run_with_cache(interv_prompt, names_filter=all_resid)
logit_lens(interv_logits, interv_cache, answers, axes[0,1], title=f"clean {src_s} -> {dest_s}, {en_word_alt}")

with model.hooks(fwd_hooks=strength_hooks):
    logits_steered, cache_steered = model.run_with_cache(clean_prompt, names_filter=all_resid)
logit_lens(logits_steered, cache_steered, answers, axes[1,0], title=f"steer {src} -> {dest}, {en_word}")

#print_topk_tokens(clean_logits[:, -1], tokenizer, title="Clean top tokens")
#print_topk_tokens(interv_logits[:, -1], tokenizer, title="Intervention top tokens")
print_topk_tokens(logits_steered[:, -1], tokenizer, title="Steered top tokens")
# %%
