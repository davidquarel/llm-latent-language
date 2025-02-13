# This script does the exact same experiments as the orignal paper https://arxiv.org/pdf/2402.10588
# but faster, in batched mode.
# %%
from pandas import DataFrame
from torch._tensor import Tensor
from imports import *
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# %%
from utils.misc import autoreload
autoreload()
# %%
# ==== Custom Libraries ====
from src.prompt import gen_prompt, gen_common_suffixes, get_answer_tensor2, get_valid_answer
from src.kv_cache import gen_kv_cache, run_with_kv_cache
from src.intervention import Intervention
from src.constants import LANGS, WORD_LIST
from src.llm import safe_tokenize, proj_batched
from utils.data import gen_lang_ids, results_dict_to_csv

from utils.plot import plot_ci_simple
from utils.config_argparse import try_parse_args
from utils.data import parse_word_list, gen_lang_ids, gen_ids
from utils.misc import ci, wilson_ci, dearrange
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
from einops import einsum
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
    #model_name: str = "google/gemma-2-2b"
    #model_name: str = "mistralai/Mistral-7B-v0.1"
    # single_token_only: bool = False
    # multi_token_only: bool = False
    # out_dir: str = './out_iclr'
    dataset_path: str = "data_wendler/word_list.csv"
    task : Literal["translate", "copy", "cloze"] = "translate"
    debug: bool = False
    
    devices : Optional[str] = "1,2"
    out : Optional[str] = "out_icml_2025"
    token_add_capitalization : bool = True
    token_add_prefixes : bool = True
    token_add_spaces : bool = True
    token_utf8_byte : bool = True
    num_multi_shot : int = 5

    mini_batch_size : int = None
    steer_mini_batch_size : int = None
    dummy_run : bool = False

cfg = Config()
cfg = try_parse_args(cfg)
cfg_dict = asdict(cfg)
print(cfg_dict)

model_basename = cfg.model_name.split('/')[-1]

# %%

if cfg.devices is not None:
    os.environ["CUDA_VISIBLE_DEVICES"] = cfg.devices  # Makes only GPUs 0 and 1 visible
    n_devices = torch.cuda.device_count()
    assert n_devices == len(cfg.devices.split(',')), f"Expected {n_devices} devices, got {cfg.devices}"
else:
    n_devices = 1
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
#os.makedirs(cfg.out_dir, exist_ok=True)

if cfg.debug:
    # Store original repr method
    original_tensor_repr = torch.Tensor.__repr__

    # Define new repr function that shows both shape and values
    def shape_and_value_repr(self):
        shape_info = f"Tensor with shape {list(self.shape)}\n"
        value_info = original_tensor_repr(self)
        return shape_info + value_info

    # Override repr globally
    torch.Tensor.__repr__ = shape_and_value_repr

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


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


def save_df(df, latent_type, cfg) -> None:
    model_basename = cfg.model_name.split('/')[-1]
    out_path = os.path.join(cfg.out, model_basename, f"translation_{latent_type}.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"{latent_type} saved to", out_path)


def stats(src : str, 
        dest : str, 
        logits : Float[Tensor, "batch seq d_vocab"],
        answers : Int[Tensor, "batch answers"], 
        padding_id : int = tokenizer.pad_token_id, 
        verbose : bool = False,
        **kwargs
)-> dict[str, Any]:
    preds = logits.argmax(dim=-1, keepdim=True).to(answers.device)
    correct_mask = (preds == answers) & (preds != tokenizer.pad_token_id)
    correct = torch.any(correct_mask, dim=-1).sum().item()
    probs = probs_on_answer(logits, answers, padding_id = tokenizer.pad_token_id, log_probs=False)
    mean_loss = -torch.log(probs).mean().item()
    ci95_loss = 1.96 * torch.log(probs).std().item() / np.sqrt(probs.shape[0])
    mean_probs = probs.mean().item()
    ci95_probs = 1.96 * probs.std().item() / np.sqrt(probs.shape[0])

    acc = correct / answers.shape[0]
    ci95_acc = 1.96 * np.sqrt(acc * (1-acc) / answers.shape[0])
    lang_latent = kwargs.get('lang_latent', None)
    if verbose:
        print(f"{src} -> del {lang_latent} -> {dest} Translated {correct}/{answers.shape[0]} correctly. Accuracy: {acc:.2%} Loss: {mean_loss:.2f} Probs {mean_probs:.2f} ± {ci95_probs:.2f}")
    result = {  'src_lang': src, 
                'dest_lang': dest, 
                'latent_lang': lang_latent, 
                'prob': mean_probs, 
                'ci95_prob': ci95_probs,
                'loss' : mean_loss,
                'ci95_loss': ci95_loss,
                'correct': correct,
                'total': answers.shape[0],
                'acc': acc,
                'ci95_acc': ci95_acc}
    
    result.update(kwargs)
    return result


def run(cfg):
    LANGS = ['fr', 'de', 'ru', 'zh', 'es','en']
    output_results = []

    print("Computing translation probabilities for each language pair")

    df = pd.read_csv(cfg.dataset_path)

    mask = df['en'].isin(WORD_LIST)
    prompt_df, suffix_df = df[mask], df[~mask]

    runner = tqdm(permutations(LANGS, 2))

    for (src, dest) in runner:
        #df = wendler.load_data("data_wendler/langs", SimpleNamespace(src_lang=src, dest_lang=dest))
        prompt = gen_prompt(prompt_df, src, dest)
        common_suffixes = gen_common_suffixes(suffix_df[src], src, dest)
        kv_cache = gen_kv_cache(prompt, model)
        suffix_toks = safe_tokenize(common_suffixes, model)
    

        logits, _ = run_with_kv_cache(suffix_toks, 
                                    kv_cache, 
                                    model, 
                                    mini_batch_size = cfg.mini_batch_size,
                                    last_seq = True,
                                    keep_resid_cache= False)

        answers = get_answer_tensor2(suffix_df[dest], tokenizer, vocab, cfg, 
                                               padding_value=tokenizer.pad_token_id)
        
        result = stats(src=src, dest=dest, lang_latent=None, logits=logits, 
                       answers=answers, padding_id=tokenizer.pad_token_id, latent_type='clean')
        output_results.append(result)
    
    output_results = pd.DataFrame(output_results)
    save_df(output_results, "clean", cfg)
    return output_results
model_basename = cfg.model_name.split('/')[-1]
out_path = os.path.join(cfg.out, model_basename, f"translation_clean.csv")

# %%
# David 04 Feb 2025
# High level: write a function that takes all answer words as input
# PCA's them down to one dimension
# and then also takes the minibatch slice, and uses proj_batch


def prepare_hooks(latent_words):
    fwd_hooks = []
    for word in latent_words:
        resid_filter = lambda x : x.endswith("resid_post")
        fwd_hooks.append((resid_filter, lambda resid, hook : hook_word_delete(resid, hook,word)))

def run_latent(cfg,
               latent_type="unembed",
) -> DataFrame:
    LANGS = ['fr', 'de', 'ru', 'zh', 'es','en']
    LATENT_LANGS = LANGS

    output_results = []
    
    resid_filter = lambda x : x.endswith("resid_post")
    print(f"Computing translation probabilities, interv {latent_type}")

    df = pd.read_csv(cfg.dataset_path)

    mask = df['en'].isin(WORD_LIST)
    prompt_df, suffix_df = df[mask], df[~mask]

    all_answers = {}
    all_subspaces = {}
    for lang in LATENT_LANGS:
        answers = get_answer_tensor2(suffix_df[lang], tokenizer, vocab, cfg, 
                                               padding_value=tokenizer.pad_token_id)
        if lang == 'zh':
            lang_space = ""
        else:
            lang_space = " "
        primary_answers = safe_tokenize(lang_space + suffix_df[lang], tokenizer).input_ids[:, 0] # (batch,)
        subspaces = model.W_U.T[primary_answers] # (batch, d_model)
        if latent_type == "unembed":
            pass
        elif latent_type == "shuffle":
            subspaces = dearrange(subspaces, dim=0) # shuffle the batch
        elif latent_type == "random":
            # generate random subspaces with same mean/std as the original
            mean = torch.mean(subspaces, dim=0)
            std = torch.std(subspaces, dim=0)
            subspaces = torch.randn_like(subspaces) * std + mean
        else:
            raise ValueError(f"run_latent: Unknown latent type {latent_type}")
            

        all_answers[lang] = answers
        all_subspaces[lang] = subspaces
    
    runner = tqdm(permutations(LANGS, 2))


    for (src, dest) in runner:
        #df = wendler.load_data("data_wendler/langs", SimpleNamespace(src_lang=src, dest_lang=dest))
        prompt = gen_prompt(prompt_df, src, dest)
        common_suffixes = gen_common_suffixes(suffix_df[src], src, dest)
        kv_cache = gen_kv_cache(prompt, model)
        suffix_toks = safe_tokenize(common_suffixes, model)
    
        # loop though all distinct triples (src, latent, dest)
        for lang_latent in LATENT_LANGS:

            if lang_latent == src or lang_latent == dest:
                continue

            def hook_word_delete(resid : Float[Tensor, "minibatch seq d_model"], 
                    hook : HookPoint,
                    slice_idx : slice,
            ) -> Float[Tensor, "minibatch seq d_model"]:
                subspace = all_subspaces[lang_latent][slice_idx].unsqueeze(1).to(device = resid.device) # (minibatch, 1, d_model)
                assert resid.shape[0] == subspace.shape[0], f"Resid shape {resid.shape} subspace shape {subspace.shape}"
                assert resid.shape[2] == subspace.shape[2], F"Resid shape {resid.shape} subspace shape {subspace.shape}"

                last_seq = suffix_toks.indices[slice_idx] # (batch,)
                idx = torch.arange(resid.shape[0], device=resid.device)

                resid_last_seq = resid[idx, last_seq] # (minibatch, d_model)

                assert resid_last_seq.shape == subspace.squeeze().shape, "shape mismatch"
                proj_resid_last = proj_batched(resid_last_seq, subspace) # (minibatch, d_model)
                
                resid[idx, last_seq] = resid[idx, last_seq] - proj_resid_last
                return resid
            
            hook_gen = (resid_filter, hook_word_delete)

            # latent_word = suffix_df[lang_latent]
            # fwd_hooks = [(resid_filter, lambda resid, hook : hook_word_delete(resid, hook,latent_word))]

            logits, _ = run_with_kv_cache(suffix_toks, 
                                kv_cache, 
                                model, 
                                mini_batch_size = cfg.mini_batch_size,
                                keep_resid_cache= False,
                                last_seq = True,
                                hook_generator = hook_gen) 
            
            if latent_type == "random":
                lang_latent = "?"
            result = stats(src=src, dest=dest, lang_latent=lang_latent, logits=logits,
                            answers=all_answers[dest], padding_id=tokenizer.pad_token_id, latent_type=latent_type)
            output_results.append(result)
            if latent_type == 'random': # only do one latent language
                break
    
    output_results = pd.DataFrame(output_results)
    save_df(output_results, latent_type=latent_type, cfg=cfg)

    return output_results
# %%
# ==============================================
# %%
def run_steer(src, dest, dummy_run = False, count= None):
    LANGS = ['fr', 'de', 'ru', 'zh', 'es','en']
    #if not dummy_run:
        
    FOREIGN_LANGS = [lang for lang in LANGS if lang not in ['en']]

    resid_filter = lambda x : x.endswith("resid_post")

    df = pd.read_csv(cfg.dataset_path)

    mask = df['en'].isin(WORD_LIST)
    prompt_df, suffix_df = df[mask], df[~mask]

    shuffle_idx = dearrange(torch.arange(len(suffix_df)))
    suffix_df_dearrange = suffix_df.iloc[shuffle_idx].reset_index(drop=True)

    all_answers = {}

    for lang in LANGS:
        answers = get_answer_tensor2(suffix_df[lang], tokenizer, vocab, cfg, 
                                                padding_value=tokenizer.pad_token_id)
        all_answers[lang] = answers

    primary_en_answer = safe_tokenize(" " + suffix_df['en'], tokenizer).input_ids[:, 0] # (batch,)
    en_subspace = model.W_U.T[primary_en_answer] #(batch, d_model)
    en_subspace = en_subspace / torch.norm(en_subspace, dim=-1, keepdim=True) # normalized subspace

    #src, dest = 'zh', 'fr'
    # #pairs = list(permutations(LANGS, 2))
    # pairs= [('zh', 'fr')]
    # runner = pairs
    # for (src, dest) in runner:
    
    #src, dest = 'zh', 'fr'
        
    prompt = gen_prompt(prompt_df, src, dest)
    clean_common_suffixes = gen_common_suffixes(suffix_df[src], src, dest)
    kv_cache = gen_kv_cache(prompt, model)
    clean_suffix_toks = safe_tokenize(clean_common_suffixes, model)

    clean_logits, clean_cache = run_with_kv_cache(tokens = clean_suffix_toks, 
                                past_kv_cache = kv_cache, 
                                model = model,
                                mini_batch_size = cfg.mini_batch_size,
                                keep_resid_cache= True,
                                last_seq = True,
                                verbose=False)
                                        
    c_new_all = torch.stack([einsum(en_subspace, resid.to(en_subspace.device), "batch d_model, batch d_model -> batch")
                    for resid in clean_cache.values()], dim=0) # (n_layers, batch)

    steer_common_suffixes = gen_common_suffixes(suffix_df_dearrange[src], src, dest)
    steer_suffix_toks = safe_tokenize(steer_common_suffixes, model)


    def hook_steer(resid : Float[Tensor, "minibatch seq d_model"], #resid in dearranged order
                hook : HookPoint,
                slice_idx : slice,
        ) -> Float[Tensor, "minibatch seq d_model"]:
            dev = resid.device
            last_seq = steer_suffix_toks.indices[slice_idx].to(dev) #(minibatch)
            idx = torch.arange(resid.shape[0], device=resid.device) # (minibatch)
            layer = hook.layer()
            #W_U_old = en_subspace[slice_idx] # (minibatch, d_model)
            #c_old = c_old[layer, slice_idx, None] # (minibatch, 1)
            resid_last_seq = resid[idx, last_seq] # (minibatch, d_model)

            W_U_new = en_subspace[slice_idx].to(dev) # (minibatch, d_model)
            c_new = c_new_all[layer][slice_idx].to(dev) # (minibatch) 

            W_U_old = en_subspace[shuffle_idx][slice_idx].to(dev) # (minibatch, d_model)
            c_old = einsum(resid_last_seq, W_U_old, "minibatch d_model, minibatch d_model -> minibatch") # (minibatch)
            new_resid = resid_last_seq - c_old[:, None] * W_U_old + c_new[:, None] * W_U_new

            resid[idx, last_seq] = new_resid
            return resid

    n_layers = model.cfg.n_layers
    pairs = [(i, j) for i in range(n_layers) for j in range(i+1, n_layers)]
    output_results = []
    
    for (start,end) in tqdm(pairs):


        def hook_steer_filter(x) -> bool:
            pattern = r"blocks\.(\d+)\.hook_resid_post"
            match = re.match(pattern, x)
            if match:
                layer = int(match.group(1))
                return start <= layer < end

            return False

        steer_logits, _ = run_with_kv_cache(tokens = steer_suffix_toks,
                                            past_kv_cache=kv_cache,
                                            model=model,
                                            mini_batch_size=cfg.mini_batch_size,
                                            keep_resid_cache=False,
                                            last_seq=True,
                                            hook_generator=(hook_steer_filter, hook_steer))

        clean_stats = stats(src=src, dest=dest, logits=clean_logits,
                            answers=all_answers[dest], latent_type='clean', verbose=False, l_start = start, l_end = end)
        
        steer_from = stats(src=src, dest=dest, logits=steer_logits,
                            answers=all_answers[dest][shuffle_idx], latent_type = 'steer_from', verbose=False, l_start = start, l_end = end)
        
        steer_from_en = stats(src=src, dest='en', logits=steer_logits,
                            answers=all_answers['en'][shuffle_idx], latent_type = 'steer_from', verbose=False, l_start = start, l_end = end)
        
        steer_to = stats(src=src, dest=dest, lang_latent='new', logits=steer_logits,
                            answers=all_answers[dest], latent_type = 'steer_to', verbose=False, l_start = start, l_end = end)
        
        steer_to_en = stats(src=src, dest='en', lang_latent='new', logits=steer_logits,
                            answers=all_answers['en'], latent_type = 'steer_to', verbose=False, l_start = start, l_end = end)
        
        output_results.append(clean_stats)
        output_results.append(steer_from)
        output_results.append(steer_from_en)
        output_results.append(steer_to)
        output_results.append(steer_to_en)

        clean_acc, steer_from_acc, steer_from_en_acc, steer_to_acc, steer_to_en_acc = [x['acc'] for x in [clean_stats, steer_from, steer_from_en, steer_to, steer_to_en]]
        clean_probs, steer_from_probs, steer_from_en_probs, steer_to_probs, steer_to_en_probs = [x['prob'] for x in [clean_stats, steer_from, steer_from_en, steer_to, steer_to_en]]

        table_data = [
            ["Metric", "Clean", "Steer From", "Steer From EN", "Steer To", "Steer To EN"],
            ["Accuracy", f"{clean_acc:.2%}", f"{steer_from_acc:.2%}", f"{steer_from_en_acc:.2%}", f"{steer_to_acc:.2%}", f"{steer_to_en_acc:.2%}"],
            ["Probability", f"{clean_probs:.2f}", f"{steer_from_probs:.2f}", f"{steer_from_en_probs:.2f}", f"{steer_to_probs:.2f}", f"{steer_to_en_probs:.2f}"]
        ]
        print(f'Steer layers {start} to {end}, {src} -> {dest} : {count} / 20')
        print(tabulate(table_data, headers="firstrow", tablefmt="pretty"))

    return output_results
#output_results = pd.DataFrame(output_results)
#save_df(output_results, latent_type=f"steer_{start}_{end}", cfg=cfg)


# %%
#out = run_steer()
if __name__ ==  "__main__":
    
    print("running clean")
    output_results = run(cfg)
    for latent_type in ['random', 'shuffle', 'unembed']:
        print("running latent")
        output_results = run_latent(cfg, latent_type=latent_type)
    print("running steer")
    output_results = run_steer()
# %%
F_LANGS = ['fr', 'de', 'ru', 'zh', 'es']
count = 1
for (src, dest) in list(permutations(F_LANGS, 2)):
    print(f"{src} -> {dest}")
    full_sweep = run_steer(src, dest, count=count)
    full_sweep_df = pd.DataFrame(full_sweep)
    save_df(full_sweep_df, f"steer_{src}_{dest}", cfg)
    count +=1

# %%
