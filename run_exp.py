# %%
from tracemalloc import start
import pandas as pd
from dataclasses import dataclass, field, asdict
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from tqdm.auto import tqdm
import pickle 
import argparse
import pprint
# === Typing Libraries ===
from typing import Tuple, List, Optional, Dict, Callable, Iterable, Any
from jaxtyping import Int, Float
from beartype import beartype

# ==== Torch/Transformer Libraries ====
import torch
from torch import Tensor
import torch.nn.functional as F
import torch.nn as nn
from transformer_lens.hook_points import HookPoint
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformer_lens import HookedTransformer


sys.path.append(os.path.dirname(os.path.abspath(__file__)))
# ==== Custom Libraries ====
import gen_data
from utils import plot_ci_plus_heatmap
from tuned_lens_wrap import load_tuned_lens
from dq_utils import proj, entropy, plot_ci, is_chinese_char, measure_performance
from logit_lens import get_logits, plot_logit_lens_latents, latent_heatmap
import intervention
from intervention import Intervention
from config_argparse import parse_args
# %%
@dataclass
class Config:
    seed: int = 42
    src_lang: str = 'fr'
    dest_lang: str = 'zh'
    latent_lang: str = 'en'
    model_name: str = 'meta-llama/Llama-2-7b-hf'
    single_token_only: bool = False
    multi_token_only: bool = False
    out_dir: str = './visuals'
    hf_token: str = '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'
    dataset_path: str = "./data/synth_llama2"
    debug: bool = True
    num_multi_shot : int = 5
    token_add_spaces: bool = True
    token_add_leading_byte: bool = False
    token_add_prefixes : bool = False
    dataset_filter_correct : bool = True
    use_tuned_lens : bool = False
    intervention_correct_latent_space : bool = True
    steer_scale_coeff : float = 1.0
    start_layer_low : int = 0
    start_layer_high : int = 32
    end_layer_low : int = 0
    end_layer_high : int = 32
    intervention_func : str = 'hook_reject_subspace'
    log_file : str = 'DUMMY_NAME'
    metric : str = 'p_alt'
    metric_goal : str = 'max'
    only_compute_stats : bool = True
    translation_threshold : float = 0.5

cfg = Config()

try:
    # The get_ipython function is available in IPython environments
    ipython = get_ipython()
    if 'IPKernelApp' not in ipython.config:  # Check if not within an IPython kernel
        raise ImportError("Not in IPython")
    print("Enabling autoreload in IPython.")
    ipython.run_line_magic('load_ext', 'autoreload')
    ipython.run_line_magic('autoreload', '2')
except Exception as e:
    print(f"Not in an IPython environment: {e}")
    # Parse command line arguments
    # parser = argparse.ArgumentParser()
    # parser.add_argument("--log_file", type=str, default="experiment.log", help="File to write experiment log to")
    # cli_args = parser.parse_args()
    # print(f"Writing experiment log to {cli_args.log_file}")
    cfg = parse_args(cfg)
    #pprint.pprint(asdict(cfg))
    assert cfg.log_file != 'DUMMY_NAME', "ERROR: log_file not set"
cfg_dict = asdict(cfg)
# %%
# fix random seed
seed = 42
np.random.seed(seed)
torch.manual_seed(seed)
torch.set_grad_enabled(False)
# %%
pd.set_option('display.max_rows', 100)  # Show all rows
pd.set_option('display.max_columns', None)  # Show all columns
pd.set_option('display.width', None)  # Auto-detect the display width for wrapping
pd.set_option('display.max_colwidth', None)  # Show full length of data in columns

# %%

    
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# tokenizer = AutoTokenizer.from_pretrained(cfg.model_name, use_fast=False, add_prefix_space=False)
# tokenizer_vocab = tokenizer.get_vocab()
# %%
if 'LOAD_MODEL' not in globals():
    LOAD_MODEL = False
    model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device, 
                                                            dtype = torch.float16)
    tokenizer_vocab = model.tokenizer.get_vocab() # type: ignore
    if cfg.use_tuned_lens:
        tuned_lens = load_tuned_lens(model).float()
        model.tuned_lens = tuned_lens
    # depricated, doesn't work
    # if cfg.use_reverse_lens:
    #     reverse_lens = ReverseLens.from_tuned_lens(tuned_lens)
    #     model.reverse_lens = reverse_lens
# %%


# TODO THINK OF A BETTER WAYTO DO THIS
assert cfg.src_lang != cfg.dest_lang, "No point converting a language to itself"
if cfg.src_lang != 'en' and cfg.dest_lang != 'en':
    df_src = pd.read_csv(os.path.join(cfg.dataset_path, f"en_to_{cfg.src_lang}")).reindex()
    df_dest = pd.read_csv(os.path.join(cfg.dataset_path, f"en_to_{cfg.dest_lang}")).reindex()
    df_raw_data = df_dest.merge(df_src, on=['en, en_tok'])
    df_raw_data = gen_data.filter_correct(df_raw_data, model) # TODO
else:
    if cfg_src_lang != 'en':
        df_raw_data = pd.read_csv(os.path.join(cfg.dataset_path, f"en_to_{cfg.src_lang}")).reindex()
    else:
        df_raw_data = pd.read_csv(os.path.join(cfg.dataset_path, f"en_to_{cfg.src_lang}")).reindex()
        
        
# %%

measure_performance(correct_dataset, model) # TODO
# %%

layer_log2 = {}

start_lower, start_upper = cfg.start_layer_low, cfg.start_layer_high
end_lower, end_upper = cfg.end_layer_low, cfg.end_layer_high

total_iterations = dq_utils.calculate_iterations(start_lower, start_upper, end_lower, end_upper)
outer_pbar = tqdm(total=total_iterations, desc='Overall Progress', leave=True)

import intervention
from logit_lens import get_logits, plot_logit_lens_latents


def is_better(stats, best_stats, cfg):
    if cfg.metric_goal == 'max':
        return stats[cfg.metric] > best_stats[cfg.metric]
    else:
        return stats[cfg.metric] < best_stats[cfg.metric]

if cfg.metric_goal == 'max':
    best_stats = {cfg.metric: -np.inf}
else:
    best_stats = {cfg.metric: np.inf}
    
    
    
kv_cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, 1) # flush cache
prefix = gen_data.generate_translation_prompt(None, src_lang, dest_lang, translations = translation_bank)
prefix_tok = model.tokenizer.encode(prefix, return_tensors="pt").to(device)
model(prefix_tok, past_kv_cache = kv_cache) #fill kv_cache
kv_cache.freeze()
    
for start_layer in range(start_lower,start_upper):
    for end_layer in range(end_lower, end_upper):
        if start_layer >= end_layer:
            continue
        
        intervene_diff = Intervention(cfg.intervention_func, range(start_layer, end_layer))
        latent_diff, logits_diff = get_logits(correct_dataset, model, intervention=intervene_diff,  **cfg_dict)
        latent_diff = latent_diff.float()
        logits_diff = logits_diff.float()
        stats = plot_logit_lens_latents(logits_diff, correct_dataset, **cfg_dict, title="diff", cfg=cfg)
        
        if is_better(stats, best_stats, cfg):
            print("==========!!!!!==========")
            new_best_msg = f"NEW BEST STATS: start_layer={start_layer}, end_layer={end_layer}, {dq_utils.format_dict_single_line_custom(stats)}"
            print("==========!!!!!==========")
            tqdm.write(new_best_msg)  # Using tqdm.write to avoid interference with the progress bar
            outer_pbar.set_description(f"")
            best_stats = stats
        else:
            outer_pbar.set_description(f"Trying: {dq_utils.format_dict_single_line_custom(stats)}")
        outer_pbar.update(1)  # Increment the progress bar after each inner iteration
        layer_log2[(start_layer, end_layer)] = stats


outer_pbar.close()  # Ensure to close the progress bar after the loop completes
# Save layer_log2 to a pickle file

base_log_file_path = cfg.log_file.rsplit('.', 1)[0]  # Strip off the extension if provided

# Ensure directory exists
os.makedirs(os.path.dirname(base_log_file_path), exist_ok=True)

with open(base_log_file_path + ".pkl", "wb") as pickle_file:
    pickle.dump(layer_log2, pickle_file)

# pickle.dump(layer_log2, open(cfg.log_file + ".pkl", "wb"))

log_legend = """
Measuring 
lp_out/p_out : logprobs/probs of correct answer
lp_alt/p_alt logprobs/probs of alternate answer
lp_diff/p_ratio: logprob_diff/probs ration of alt-correct or alt/correct
"""

pp = pprint.PrettyPrinter(sort_dicts=False)
# Save log_legend to the log file
with open(base_log_file_path + ".log", "a") as f:
    f.write("Command: " + ' '.join(sys.argv) + "\n")
    f.write(pp.pformat(asdict(cfg)))
    f.write("\n==============\n")
    f.write(intervene_diff.description)
    f.write("\n==============\n")
    f.write(log_legend)
    f.write("\n==============\n")
    f.write(f"size of dataset: {len(dataset)}")
    f.write(f"size of correct dataset: {len(correct_dataset)}")

print("Done!")

# %%
