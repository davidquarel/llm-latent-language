# %%
%load_ext autoreload
%autoreload 2
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
from reverse_tuned_lens import ReverseLens
from dq_utils import proj, entropy, plot_ci, is_chinese_char, measure_performance
from logit_lens import get_logits, plot_logit_lens_latents, latent_heatmap, get_logits_batched
import intervention
from intervention import Intervention
from config_argparse import parse_args
from llama_merge_csv import construct_dataset
# %%
@dataclass
class Config:
    seed: int = 42
    src_lang: str = 'fr'
    dest_lang: str = 'zh'
    latent_lang: str = 'en'
    model_size: str = '7b'
    model_name: str = 'gemma-2b'  #'meta-llama/Llama-2-%s-hf' % model_size
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
    use_reverse_lens : bool = False
    rev_lens_scale : bool = 1
    only_compute_stats : bool = False
    cache_prefix : bool = True

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
try: 
    ipython = get_ipython()
    # if in jupyter notebook, force variables
    #cfg.use_reverse_lens = True
    
except:
    pass

model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                        device=device, 
                                                        dtype = torch.float16)
tokenizer_vocab = model.tokenizer.get_vocab() # type: ignore    


# %%
def translate(src_word, model):
    df_fr = pd.read_csv(os.path.join(cfg.dataset_path, 'llama2_filtered.csv'))
    prompt = gen_data.generate_translation_prompt(src_word, **cfg_dict)
    #with_cache_logits = model(rest_of_tokens, past_kv_cache=kv_cache)

    prompt_tok = model.tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
    all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
    logits, cache = model.run_with_cache(prompt_tok, names_filter = all_post_resid)
    latents = torch.stack([c[0, -1] for c in cache.values()],  dim=0)
    approx_logits = model.unembed(model.ln_final(latents.unsqueeze(0))).squeeze() #(layers, vocab)
    approx_logprobs= F.log_softmax(approx_logits, dim=-1)
    top_logprobs, top_idx = torch.topk(approx_logprobs, 10, dim=-1)
    top_logprobs = top_logprobs.cpu().numpy()
    top_idx = top_idx.cpu().numpy()
    top_tokens = [model.tokenizer.convert_ids_to_tokens(x) for x in top_idx]

    print(f"Prompt: {prompt}")
    print(f"Top Tokens: {top_tokens[-1]}")
    print(f"Top Probs: {np.exp(top_logprobs[-1])}")
    
    n_layers = model.cfg.n_layers
    top_k = 10
    fig, ax = plt.subplots(figsize=(10, 12))
    # Create the heatmap
    cax = ax.imshow(top_logprobs, cmap='cividis', aspect='auto')
    from matplotlib.font_manager import FontProperties

    font_properties = FontProperties(fname='NotoSansSC-Bold.ttf')

    # Add colorbar
    cbar = fig.colorbar(cax)
    cbar.set_label('Probability')

    def get_text_color(background_color):
        r, g, b, _ = background_color
        brightness = 0.299 * r + 0.587 * g + 0.114 * b
        return 'white' if brightness < 0.3 else 'black'

    from matplotlib.colors import Normalize
    from matplotlib.cm import ScalarMappable

    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap='cividis')
    # Set axis labels
    ax.set_ylabel('Layers')
    # Annotate each cell with the token
    for i in range(n_layers):
        for j in range(top_k):
            token = top_tokens[i][j]
            logprob = top_logprobs[i, j]
            
            color = sm.to_rgba(logprob)
            text_color = get_text_color(color)
            
            ax.text(j, i, token, ha='center', va='center', color='black', 
                    fontsize=10, font_properties = font_properties)

    plt.title(f'{model.cfg.model_name} {cfg.src_lang}: {src_word} -> {cfg.dest_lang}:?')
    plt.savefig(f'./logit_lenses/gemma_{cfg.src_lang}_{src_word}_to_{cfg.dest_lang}.svg')
    plt.tight_layout()
    plt.show()
    
for src_word in ['roi', 'élect', 'livre', 'chanson', 'sept']:
    translate(src_word, model)
# %%
