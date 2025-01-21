# %%
%load_ext autoreload
%autoreload 2
# %%
from imports import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
from src.prompt import gen_prompt, gen_common_suffixes, find_all_tokens
from src.kv_cache import gen_kv_cache, run_with_kv_cache
from src.llm import safe_tokenize, TokenizedSuffixesResult
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
from src.kv_cache import broadcast_kv_cache
from transformer_lens.utils import test_prompt
# Import GPT-2 tokenizer
#gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
#llama_tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")
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
    model_name: str = 'meta-llama/Llama-2-7b-hf'
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
    if model.tokenizer.unk_token_id is None:
        model.tokenizer.unk_token_id = model.tokenizer.bos_token_id
        
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

# %%



src_multishot = LANG_BANK[cfg.src_lang]
dest_multishot = LANG_BANK[cfg.dest_lang]
src_words = df[cfg.src_lang]

prompt = gen_prompt(src_words = src_multishot,
                dest_words = dest_multishot,
                src_lang = cfg.src_lang, 
                dest_lang = cfg.dest_lang,
                num_examples=cfg.num_multi_shot)
kv_cache = gen_kv_cache(prompt, model)

suffixes = gen_common_suffixes(src_words,
                                src_lang = cfg.src_lang,
                                dest_lang = cfg.dest_lang)
suffix_toks = safe_tokenize(suffixes, model)

# test detokenization
print(src_words[:20])
test_suffix_tokens = safe_tokenize(suffixes, model)
recover_src_words = model.tokenizer.convert_ids_to_tokens(test_suffix_tokens.input_ids[:, 0])
print(recover_src_words[:20])

all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
logits, cache = run_with_kv_cache(suffix_toks.input_ids, kv_cache, model, names_filter=all_post_resid)

# intervention = Intervention("hook_batch_reject", range(model.cfg.n_layers))

logits = eindex(logits, suffix_toks.indices, "batch [batch] vocab")
cache_stacked = torch.stack(tuple(cache.values()), dim=1) #(batch, num_layer, seq, d_model)
cache_last = eindex(cache_stacked, suffix_toks.indices, "batch layer [batch] dmodel")

logits_per_layer = model.unembed(model.ln_final(cache_last)) # (batch, num_layer, vocab)
dest_idx = safe_tokenize(list(df[cfg.dest_lang]), model).input_ids[:,0]
latent_idx = safe_tokenize(list(df[cfg.latent_lang]), model).input_ids[:,0]

logprobs_internal = torch.log_softmax(logits_per_layer, dim=-1)


logprobs_dest = eindex(logprobs_internal, dest_idx, "batch layer [batch]").cpu()
logprobs_latent = eindex(logprobs_internal, latent_idx, "batch layer [batch]").cpu()




translated = list(zip(df[cfg.src_lang][:10], model.tokenizer.convert_ids_to_tokens(logits.argmax(dim=-1))))

print(tabulate(translated, headers=[f'{cfg.src_lang=}', f'{cfg.dest_lang=}']))

print(list(zip(model.tokenizer.convert_ids_to_tokens(logits.argmax(dim=-1)), 
                [x.item() for x in torch.softmax(logits, dim=-1).max(dim=-1).values])))


def plot_ci_series(tensors, dim=0):
    fig, ax = plt.subplots()
    for tensor in tensors:
        plot_ci_simple(tensor, ax, dim=dim)
    plt.legend()
    plt.show()
    

plot_ci_series([logprobs_dest, logprobs_latent], dim=0)
# %%

    
plot_ci_series([logprobs_dest, logprobs_latent], dim=0)
# %%

# %%
def good_unbatched_uni_token():
    """ 
    known good latent lang plot unbatched
    """
    dest, latent = [], []
    for i in tqdm(range(len(df))):
        uni_logits, uni_cache = model.run_with_cache(prompt + suffixes[i], names_filter = all_post_resid)
        dest_id = safe_tokenize(df[cfg.dest_lang][i], model).input_ids[:, 0]
        latent_id = safe_tokenize(" " + df[cfg.latent_lang][i], model).input_ids[:, 0]
        all_resid = torch.stack(list(uni_cache.values()),dim=1)[0, :, -1]

        uni_logit_per_layer = model.unembed(model.ln_final(all_resid))
        uni_logprobs = torch.softmax(uni_logit_per_layer, dim=-1)
        uni_logprobs_dest = uni_logprobs[:, dest_id]
        uni_logprobs_latent = uni_logprobs[:, latent_id]
        dest.append(uni_logprobs_dest)
        latent.append(uni_logprobs_latent)
    dest = torch.stack(dest, dim=0).cpu().squeeze()
    latent = torch.stack(latent, dim=0).cpu().squeeze()

    plot_ci_series([dest, latent], dim=0)
# %%
def good_unbatched_multi_token():
    """ 
    improve response on unbatched by measuring other tokens
    """
    dest, latent = [], []
    for i in tqdm(range(len(df))):
        uni_logits, uni_cache = model.run_with_cache(prompt + suffixes[i], names_filter = all_post_resid)
        #dest_id = safe_tokenize(df[cfg.dest_lang][i], model).input_ids[0]
        #latent_id = safe_tokenize(" " + df[cfg.latent_lang][i], model).input_ids[0]
        
        dest_words = [df[cfg.dest_lang][i]] + parse_word_list(df[f'claude_{cfg.dest_lang}'][i]) 
        dest_ids = torch.unique(safe_tokenize(dest_words, model).input_ids[:, 0])
        
        latent_words = [df[cfg.latent_lang][i]] + parse_word_list(df[f'claude_{cfg.latent_lang}'][i])
        latent_ids = torch.unique(safe_tokenize([" " + x for x in latent_words], model).input_ids[:, 0])
        
        all_resid = torch.stack(list(uni_cache.values()),dim=1)[0, :, -1]

        uni_logit_per_layer = model.unembed(model.ln_final(all_resid))
        uni_probs = torch.softmax(uni_logit_per_layer, dim=-1)
        uni_probs_dest = uni_probs[:, dest_ids].sum(-1)
        uni_probs_latent = uni_probs[:, latent_ids].sum(-1)
        dest.append(uni_probs_dest)
        latent.append(uni_probs_latent)
    dest = torch.stack(dest, dim=0).cpu().squeeze()
    latent = torch.stack(latent, dim=0).cpu().squeeze()

    print(f"{dest[:, -1].mean()} ± {dest[:, -1].std()}")

    plot_ci_series([dest, latent], dim=0)


def good_unbatched_multi_token_reject():
    """ 
    improve response on unbatched by measuring other tokens
    then reject with latent ids
    """
    dest, latent = [], []
    
    intervention = Intervention("hook_reject_subspace", range(model.cfg.n_layers))
    
    for i in tqdm(range(len(df))):
        
        #dest_id = safe_tokenize(df[cfg.dest_lang][i], model).input_ids[0]
        #latent_id = safe_tokenize(" " + df[cfg.latent_lang][i], model).input_ids[0]
        
        dest_words = [df[cfg.dest_lang][i]] + parse_word_list(df[f'claude_{cfg.dest_lang}'][i]) 
        dest_ids = torch.unique(safe_tokenize(dest_words, model).input_ids[:, 0])
        j = (i+1) % len(df)
        alt_latent_words = [df[cfg.latent_lang][j]] + parse_word_list(df[f'claude_{cfg.latent_lang}'][j])
        alt_latent_ids = torch.unique(safe_tokenize([" " + x for x in alt_latent_words], model).input_ids[:, 0])
        
        latent_words = [df[cfg.latent_lang][i]] + parse_word_list(df[f'claude_{cfg.latent_lang}'][i])
        latent_ids = torch.unique(safe_tokenize([" " + x for x in latent_words], model).input_ids[:, 0])
        
        fwd_hooks = intervention.fwd_hooks(model, latent_ids = latent_ids)
        with model.hooks(fwd_hooks=fwd_hooks):
            uni_logits, uni_cache = model.run_with_cache(prompt + suffixes[i], names_filter = all_post_resid)
        
        
        all_resid = torch.stack(list(uni_cache.values()),dim=1)[0, :, -1]

        uni_logit_per_layer = model.unembed(model.ln_final(all_resid))
        uni_probs = torch.softmax(uni_logit_per_layer, dim=-1)
        uni_probs_dest = uni_probs[:, dest_ids].sum(-1)
        uni_probs_latent = uni_probs[:, latent_ids].sum(-1)
        dest.append(uni_probs_dest)
        latent.append(uni_probs_latent)
    dest = torch.stack(dest, dim=0).cpu().squeeze()
    latent = torch.stack(latent, dim=0).cpu().squeeze()

    print(f"{dest[:, -1].mean()} ± {dest[:, -1].std()}")

    plot_ci_series([dest, latent], dim=0)



# %%

def gen_lang_ids(df, langs):
    id_bank = {}
    for lang in langs:
        id_bank[lang] = gen_ids(df, lang)
    return id_bank

def gen_ids(df, lang):
    all_ids = []
    space_tok = safe_tokenize(" ", model).input_ids.item()
    for primary, word_list in df[[lang, f'claude_{lang}']].values:
        dest_words = [primary] + parse_word_list(word_list)
        padded_words = [" " + x for x in dest_words] + dest_words
            
        dest_ids = safe_tokenize(padded_words, model).input_ids[:, 0]
        dest_ids = dest_ids[dest_ids != space_tok]
        dest_ids = torch.unique(dest_ids)
        all_ids.append(dest_ids)
    all_ids = torch.nn.utils.rnn.pad_sequence(all_ids, batch_first=True, 
                                              padding_value=model.tokenizer.unk_token_id)
    return all_ids
    

def ci(data, dim=0):
    mean = data.mean(dim=dim)
    std = data.std(dim=dim)
    sem95 = 1.96 * std / (len(data)**0.5) 
    print(f"{mean} ± {sem95}")
    return mean, sem95

def good_batched_multi_token_reject(all_lang_ids,
                                    fast = True,
                                    intervention = None):
    """ 
    known good latent lang plot batched with measuring other tokens
    """
    from src.kv_cache import broadcast_kv_cache
    # prompt_cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, 1) # flush cache
    # model(prompt, past_kv_cache = prompt_cache) #fill kv_cache
    # kv_cache.freeze()
    
    latent_idx = all_lang_ids[cfg.latent_lang]
    dest_idx = all_lang_ids[cfg.dest_lang]
    src_idx = all_lang_ids[cfg.src_lang]
    
    alt_latent_idx = torch.empty_like(latent_idx)
    alt_latent_idx[:-1] = latent_idx[1:]
    alt_latent_idx[-1] = latent_idx[0] #alt_latent_idx is rotated by 1

    prompt_cache = gen_kv_cache(prompt, model)
    broadcast_kv_cache(prompt_cache, len(suffixes))

    suffix_toks = safe_tokenize(suffixes, model)

    rejection_subspace = model.unembed.W_U.T[latent_idx]
    rejection_subspace[latent_idx==0] = 0

    #intervention = Intervention("hook_batch_reject", range(model.cfg.n_layers))
    fwd_hooks = [] if intervention is None else intervention.fwd_hooks(model, 
                                       rejection_subspaces = rejection_subspace, 
                                       latent_ids = latent_idx,
                                       suffix_idx = suffix_toks.indices,
                                       fast = fast)
    
    with model.hooks(fwd_hooks=fwd_hooks):
        logits, cache = model.run_with_cache(suffix_toks.input_ids, past_kv_cache = prompt_cache, names_filter=all_post_resid)
        
    cache_stacked = torch.stack(tuple(cache.values()),dim=1) #(batch, num_layer, seq, d_model)

        
    cache_last = eindex(cache_stacked, suffix_toks.indices, "batch layer [batch] dmodel")

    logits_per_layer = model.unembed(model.ln_final(cache_last)) # (batch, num_layer, vocab)
    probs_per_layer = torch.softmax(logits_per_layer, dim=-1)
    
    probs_dest = eindex(probs_per_layer, dest_idx, "batch layer [batch num_correct]").cpu().sum(-1)
    probs_latent = eindex(probs_per_layer, latent_idx, "batch layer [batch num_correct]").cpu().sum(-1)

    ci(probs_dest[:, -1])
    fig, ax = plt.subplots()
    plot_ci_simple(probs_dest, ax, label="dest", dim=0)
    plot_ci_simple(probs_latent, ax, label="latent", dim=0)
    plt.legend()
    plt.show()
    
def good_batched_multi_token_only_end(fast = True,
                                    intervention = None):
    """ 
    known good latent lang plot batched with measuring other tokens
    """
    from src.kv_cache import broadcast_kv_cache
    # prompt_cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, 1) # flush cache
    # model(prompt, past_kv_cache = prompt_cache) #fill kv_cache
    # kv_cache.freeze()
    
    latent_idx = gen_ids(df, cfg.latent_lang)
    dest_idx = gen_ids(df, cfg.dest_lang)
    src_idx = gen_ids(df, cfg.src_lang)
    
    alt_latent_idx = torch.empty_like(latent_idx)
    alt_latent_idx[:-1] = latent_idx[1:]
    alt_latent_idx[-1] = latent_idx[0] #alt_latent_idx is rotated by 1

    prompt_cache = gen_kv_cache(prompt, model)
    broadcast_kv_cache(prompt_cache, len(suffixes))

    suffix_toks = safe_tokenize(suffixes, model)

    rejection_subspace = model.unembed.W_U.T[latent_idx]
    rejection_subspace[latent_idx==0] = 0

    #intervention = Intervention("hook_batch_reject", range(model.cfg.n_layers))
    fwd_hooks = [] if intervention is None else intervention.fwd_hooks(model, 
                                       rejection_subspaces = rejection_subspace, 
                                       latent_ids = latent_idx,
                                       suffix_idx = suffix_toks.indices,
                                       fast = fast)
    
    with model.hooks(fwd_hooks=fwd_hooks):
        logits = model(suffix_toks.input_ids, past_kv_cache = prompt_cache) # (batch, seq vocab)
        
    logits_last = eindex(logits, suffix_toks.indices, "batch [batch] dmodel")
    probs_last = torch.softmax(logits_last, dim=-1)
    
    probs_dest = eindex(probs_last, dest_idx, "batch [batch num_correct]").cpu().sum(-1) 
    probs_latent = eindex(probs_last, latent_idx, "batch [batch num_correct]").cpu().sum(-1)
    ci(probs_dest)
# %%

def good_batched_multi_token():
    """ 
    known good latent lang plot batched with measuring other tokens
    """
    from src.kv_cache import broadcast_kv_cache
    # prompt_cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, 1) # flush cache
    # model(prompt, past_kv_cache = prompt_cache) #fill kv_cache
    # kv_cache.freeze()

    prompt_cache = gen_kv_cache(prompt, model)
    broadcast_kv_cache(prompt_cache, len(suffixes))

    suffix_toks = safe_tokenize(suffixes, model)

    logits, cache = model.run_with_cache(suffix_toks.input_ids, past_kv_cache = prompt_cache, names_filter=all_post_resid)
    cache_stacked = torch.stack(tuple(cache.values()),dim=1) #(batch, num_layer, seq, d_model)

        
    cache_last = eindex(cache_stacked, suffix_toks.indices, "batch layer [batch] dmodel")

    logits_per_layer = model.unembed(model.ln_final(cache_last)) # (batch, num_layer, vocab)
    probs_per_layer = torch.softmax(logits_per_layer, dim=-1)



    # dest_idx = safe_tokenize(df[cfg.dest_lang], model).input_ids[:,0]
    # latent_idx = safe_tokenize([" " + x for x in df[cfg.latent_lang]], model).input_ids[:,0]

    #latent_idx = safe_tokenize([" " + x for x in df[cfg.latent_lang]], model).input_ids[:,0]
    latent_idx = gen_ids(df, cfg.latent_lang)
    dest_idx = gen_ids(df, cfg.dest_lang)
    probs_dest = eindex(probs_per_layer, dest_idx, "batch layer [batch num_correct]").cpu().sum(-1)
    probs_latent = eindex(probs_per_layer, latent_idx, "batch layer [batch num_correct]").cpu().sum(-1)

    ci(probs_dest[:, -1])
    fig, ax = plt.subplots()
    plot_ci_simple(probs_dest, ax, label="dest", dim=0)
    plot_ci_simple(probs_latent, ax, label="latent", dim=0)
    plt.legend()
    plt.show()
# %%


# %%
def good_batched_uni_token():
    """ 
    known good latent lang plot batched
    """
    from src.kv_cache import broadcast_kv_cache
    # prompt_cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, 1) # flush cache
    # model(prompt, past_kv_cache = prompt_cache) #fill kv_cache
    # kv_cache.freeze()

    prompt_cache = gen_kv_cache(prompt, model)
    N = len(suffixes)
    broadcast_kv_cache(prompt_cache, N)

    suffix_toks = safe_tokenize(suffixes[:N], model)

    logits, cache = model.run_with_cache(suffix_toks.input_ids, past_kv_cache = prompt_cache, names_filter=all_post_resid)
    cache_stacked = torch.stack(tuple(cache.values()),dim=1) #(batch, num_layer, seq, d_model)

        
    cache_last = eindex(cache_stacked, suffix_toks.indices, "batch layer [batch] dmodel")

    logits_per_layer = model.unembed(model.ln_final(cache_last)) # (batch, num_layer, vocab)
    probs_per_layer = torch.softmax(logits_per_layer, dim=-1)

    dest_idx = safe_tokenize(list(df[cfg.dest_lang][:N]), model).input_ids[:,0]
    latent_idx = safe_tokenize([" " + x for x in list(df[cfg.latent_lang][:N])], model).input_ids[:,0]

    idx = torch.arange(N)
    probs_dest = eindex(probs_per_layer, dest_idx, "batch layer [batch]").cpu()
    probs_latent = eindex(probs_per_layer, latent_idx, "batch layer [batch]").cpu()


    ci(probs_dest[:, -1])
    fig, ax = plt.subplots()
    plot_ci_simple(probs_dest, ax, label="dest", dim=0)
    plot_ci_simple(probs_latent, ax, label="latent", dim=0)
    plt.legend()
    plt.show()

# plt.plot(logprobs_dest, label="dest")
# plt.plot(logprobs_latent, label="latent")
# plt.legend()
# plt.show()
# %%
def run():
    id_bank = gen_lang_ids(df, ['en', 'zh', 'fr'])
    good_batched_uni_token()
    good_batched_multi_token()
    #good_unbatched_uni_token()
    #good_unbatched_multi_token()
    #good_unbatched_multi_token_reject()
    good_batched_multi_token_reject(id_bank, intervention = None)
    good_batched_multi_token_reject(id_bank, intervention = Intervention("hook_batch_reject", range(model.cfg.n_layers)))
    
    good_batched_multi_token_only_end(None)
    good_batched_multi_token_only_end(intervention = Intervention("hook_batch_reject", range(model.cfg.n_layers)))
run()
# %%
