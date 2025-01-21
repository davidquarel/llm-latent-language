# %%
# %load_ext autoreload
# %autoreload 2
# %%
from imports import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
from src.prompt import gen_prompt, gen_common_suffixes, find_all_tokens
from src.kv_cache import gen_kv_cache, run_with_kv_cache
from src.intervention import Intervention
from src.constants import LANG_TO_NAME, LANG_BANK
from src.llm import safe_tokenize
from utils.data import gen_lang_ids, results_dict_to_csv

from utils.plot import plot_ci_simple
from utils.config_argparse import try_parse_args
from utils.data import parse_word_list, gen_lang_ids, gen_ids
from utils.misc import ci

from eindex import eindex
from collections import namedtuple
import warnings
import re
from transformers import AutoTokenizer
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache
from src.kv_cache import broadcast_kv_cache
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
    model_name: str = "meta-llama/Llama-2-7b-hf" # 'meta-llama/Meta-Llama-3-8B'
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

cfg = Config()
cfg = try_parse_args(cfg)
cfg_dict = asdict(cfg)
print(cfg_dict)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
os.makedirs(cfg.out_dir, exist_ok=True)

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

short_model_name = cfg.model_name.split("/")[-1]

os.makedirs(os.path.join(cfg.out_dir, short_model_name), exist_ok=True)

# %%

    
def good_batched_multi_token_only_end(prompt = None,
                                      model = None,
                                      suffixes = None, 
                                    intervention = None,
                                    id_bank = None,
                                    cfg=None,
                                    fast = True,
                                    use_alt_latent = False):
    """ 
    known good latent lang plot batched with measuring other tokens
    """
    from src.kv_cache import broadcast_kv_cache
    # prompt_cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, 1) # flush cache
    # model(prompt, past_kv_cache = prompt_cache) #fill kv_cache
    # kv_cache.freeze()
    
    latent_idx = id_bank[cfg.latent_lang]
    dest_idx = id_bank[cfg.dest_lang]
    src_idx = id_bank[cfg.src_lang]
    
    if use_alt_latent:
        alt_latent_idx = torch.empty_like(latent_idx)
        alt_latent_idx[:-1] = latent_idx[1:]
        alt_latent_idx[-1] = latent_idx[0] #alt_latent_idx is rotated by 1
        interv_idx = alt_latent_idx
    else:
        interv_idx = latent_idx

    prompt_cache = gen_kv_cache(prompt, model)
    broadcast_kv_cache(prompt_cache, len(suffixes))

    suffix_toks = safe_tokenize(suffixes, model)

    if fast:
        rejection_subspace = model.unembed.W_U.T[interv_idx]
        rejection_subspace[interv_idx==0] = 0
    else:
        rejection_subspace = None

    #intervention = Intervention("hook_batch_reject", range(model.cfg.n_layers))
    fwd_hooks = [] if intervention is None else intervention.fwd_hooks(model, 
                                       rejection_subspaces = rejection_subspace,
                                       latent_idx = interv_idx, 
                                       suffix_idx = suffix_toks.indices)
    
    with model.hooks(fwd_hooks=fwd_hooks):
        logits = model(suffix_toks.input_ids, past_kv_cache = prompt_cache) # (batch, seq, vocab)
        
    logits_last = eindex(logits, suffix_toks.indices, "batch [batch] dmodel") # (batch, vocab)
    probs_last = torch.softmax(logits_last, dim=-1) # (batch, vocab)
    probs_last[:, model.tokenizer.unk_token_id] = 0 # zero out unk token
    probs_dest = eindex(probs_last, dest_idx, "batch [batch num_correct]").cpu().sum(-1) 
    probs_latent = eindex(probs_last, latent_idx, "batch [batch num_correct]").cpu().sum(-1)
    return ci(probs_dest)

def raw_good_batched_multi_token_only_end(prompt = None,
                                      model = None,
                                      suffixes = None, 
                                    intervention = None,
                                    id_bank = None,
                                    cfg=None,
                                    fast = True,
                                    use_alt_latent = False):
    """ 
    known good latent lang plot batched with measuring other tokens
    """
    from src.kv_cache import broadcast_kv_cache
    # prompt_cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, 1) # flush cache
    # model(prompt, past_kv_cache = prompt_cache) #fill kv_cache
    # kv_cache.freeze()
    
    latent_idx = id_bank[cfg.latent_lang]
    dest_idx = id_bank[cfg.dest_lang]
    src_idx = id_bank[cfg.src_lang]
    
    if use_alt_latent:
        alt_latent_idx = torch.empty_like(latent_idx)
        alt_latent_idx[:-1] = latent_idx[1:]
        alt_latent_idx[-1] = latent_idx[0] #alt_latent_idx is rotated by 1
        interv_idx = alt_latent_idx
    else:
        interv_idx = latent_idx

    prompt_cache = gen_kv_cache(prompt, model)
    broadcast_kv_cache(prompt_cache, len(suffixes))

    suffix_toks = safe_tokenize(suffixes, model)

    if fast:
        rejection_subspace = model.unembed.W_U.T[interv_idx]
        rejection_subspace[interv_idx==0] = 0
    else:
        rejection_subspace = None

    #intervention = Intervention("hook_batch_reject", range(model.cfg.n_layers))
    fwd_hooks = [] if intervention is None else intervention.fwd_hooks(model, 
                                       rejection_subspaces = rejection_subspace,
                                       latent_idx = interv_idx, 
                                       suffix_idx = suffix_toks.indices)
    
    with model.hooks(fwd_hooks=fwd_hooks):
        logits, cache = model.run_with_cache(suffix_toks.input_ids, past_kv_cache = prompt_cache, names_filter = all_post_resid) # (batch, seq, vocab)
        
    cache_stacked = torch.stack(tuple(cache.values()),dim=1) #(batch, num_layer, seq, d_model)
    
    batch, num_layer, seq, d_model = cache_stacked.shape
    vocab = model.cfg.d_vocab
    
    logits_per_layer = torch.empty((batch, num_layer, vocab), device=device)
    for num_layer in range(model.cfg.n_layers):
        pre_seq_logits = model.unembed(model.ln_final(cache_stacked[:, num_layer])) # (batch, seq, vocab)
        logits_per_layer[:, num_layer] = eindex(pre_seq_logits, suffix_toks.indices, "batch [batch] vocab")
    
    # uses too much memory
    # logits_per_layer = model.unembed(model.ln_final(cache_stacked)) # (batch, num_layer, seq, vocab)
    # logits_per_layer = eindex(logits_per_layer, suffix_toks.indices, "batch num_layer [batch] vocab") # (batch, num_layer, vocab)
    
    probs_per_layer = torch.softmax(logits_per_layer, dim=-1) # (batch, num_layer, seq, vocab)
    
    #probs_per_layer = eindex(probs_per_layer, suffix_toks.indices, "batch num_layer [batch] vocab") # (batch, num_layer, vocab)
    #logits_last = eindex(logits, suffix_toks.indices, "batch [batch] dmodel") # (batch, vocab)
    #probs_last = torch.softmax(logits_last, dim=-1) # (batch, vocab)
    #probs_last[:, model.tokenizer.unk_token_id] = 0 # zero out unk token
    return probs_per_layer # (batch, vocab)
    # probs_dest = eindex(probs_last, dest_idx, "batch [batch num_correct]").cpu().sum(-1) 
    # probs_latent = eindex(probs_last, latent_idx, "batch [batch num_correct]").cpu().sum(-1)
    # return ci(probs_dest)


# %%
from itertools import permutations

all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
id_bank = gen_lang_ids(df, model, ['en', 'zh', 'fr', 'es', 'de', 'ru'])
combos_pairs = list(permutations(id_bank.keys(), 2))    

results = {}
runner = tqdm(combos_pairs)
for (src_lang, dest_lang) in runner:
    
    cfg_ex = Config(src_lang = src_lang, dest_lang = dest_lang)
    prompt = gen_prompt(src_lang=src_lang, dest_lang=dest_lang)
    src_words = df[cfg_ex.src_lang]
    suffixes = gen_common_suffixes(src_words,
                            src_lang = cfg_ex.src_lang,
                            dest_lang = cfg_ex.dest_lang)
    
    probs_per_layer = raw_good_batched_multi_token_only_end(prompt = prompt, 
                                        model = model, 
                                        suffixes = suffixes,
                                        intervention = None,
                                        id_bank = id_bank,
                                        cfg=cfg_ex) # (batch, num_layer, vocab)
    probs_last = probs_per_layer[:, -1] #(batch, vocab)
    
    
    probs_dest = eindex(probs_last, id_bank[dest_lang], "batch [batch num_correct]").cpu().sum(-1)
    avg_prob_dest, sem95_prob_dest = ci(probs_dest)
    
    for latent_lang in id_bank.keys():
        if latent_lang == src_lang or latent_lang == dest_lang:
            continue
        latent_idx = id_bank[latent_lang]
        #take latent over all layers!
        probs_latent_per_layer = eindex(probs_per_layer, latent_idx, "batch num_layer [batch num_correct]").cpu().sum(-1) # (batch, num_layer)
        
        best_layer = torch.argmax(probs_latent_per_layer.mean(dim=0)) # (num_layer)
        best_probs_latent = probs_latent_per_layer[:, best_layer]
        #probs_latent = eindex(probs_last, latent_idx, "batch [batch num_correct]").cpu().sum(-1)
        print(f" Running {src_lang} -> {dest_lang} with {latent_lang} and best layer {best_layer.item()}")
        avg_prob_latent, sem95_prob_latent = ci(best_probs_latent)
        
        results[(src_lang, dest_lang, latent_lang)] = (avg_prob_dest.item(), sem95_prob_dest.item(), avg_prob_latent.item(), sem95_prob_latent.item()) 
        print(f"Running {src_lang} -> {dest_lang} with {latent_lang}: {avg_prob_dest.item():.2f} ± {sem95_prob_dest.item():.2f} and {avg_prob_latent.item():.2f} ± {sem95_prob_latent.item():.2f}")
    runner.set_description(f"Running {src_lang} -> {dest_lang}: {avg_prob_dest.item():.2f} ± {sem95_prob_dest.item():.2f}")
    
    
results_dict_to_csv(results, os.path.join(cfg.out_dir, short_model_name, "translation_no_interv_latent.csv"), latent=True)
#quit() #TODO: remove this line
# %%
combos_latent = list(permutations(id_bank.keys(), 3))
print("Running experiments latent and alt...")

results_interv = {}
intervention = Intervention("hook_batch_reject", range(model.cfg.n_layers))

for use_alt_latent in [True, False]:
    runner = tqdm(combos_latent)
    for (latent_lang, src_lang, dest_lang) in runner:
        cfg_ex = Config(src_lang = src_lang, dest_lang = dest_lang, latent_lang = latent_lang)
        prompt = gen_prompt(src_lang=src_lang, dest_lang=dest_lang)
        src_words = df[cfg_ex.src_lang]
        suffixes = gen_common_suffixes(src_words,
                                src_lang = cfg_ex.src_lang,
                                dest_lang = cfg_ex.dest_lang)
        
        
        avg_prob, sem95_prob = good_batched_multi_token_only_end(prompt = prompt, 
                                            model = model, 
                                            suffixes = suffixes,
                                            intervention = intervention,
                                            id_bank = id_bank,
                                            use_alt_latent=use_alt_latent,
                                            cfg=cfg_ex)
        
        results_interv[(use_alt_latent, src_lang, dest_lang, latent_lang)] = (avg_prob.item(), sem95_prob.item())
        if use_alt_latent:
            runner.set_description(f"Running {src_lang} -> {dest_lang} with ALT {latent_lang}: {avg_prob.item():.2f} ± {sem95_prob.item():.2f}")
        else:
            runner.set_description(f"Running {src_lang} -> {dest_lang} {latent_lang}: {avg_prob.item():.2f} ± {sem95_prob.item():.2f}")
# %%
# split results_interv into two dicts, one for use_alt_latent = True and one for False
results_interv_no_alt = {k[1:]: v for k, v in results_interv.items() if not k[0]}
results_interv_alt = {k[1:]: v for k, v in results_interv.items() if k[0]}
# %%



results_dict_to_csv(results_interv_no_alt, os.path.join(cfg.out_dir, short_model_name, "translation_interv.csv"))
results_dict_to_csv(results_interv_alt, os.path.join(cfg.out_dir, short_model_name, "translation_interv_alt.csv"))
# %%
