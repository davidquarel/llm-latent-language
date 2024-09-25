# %%

import torch
# from transformers import GPT2LMHeadModel, GPT2Tokenizer
# from transformer_lens import HookedTransformer, utils
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from utils.plot import plot_ci
from src.kv_cache import run_with_kv_cache
from src.datatypes import TokenizedSuffixesResult, LangIdx
from torch import Tensor
from typing import Tuple, List, Optional, Dict, Callable, Iterable, Any, Union
from jaxtyping import Int, Float
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from matplotlib.colors import LogNorm
import numpy as np
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCache
# import src.prefix as prefix
from eindex import eindex
from torch.utils.data import TensorDataset, DataLoader, Dataset
# %%




@torch.no_grad
def run_with_shared_prefix(prefix_toks : Int[Tensor, "seq"], 
                           suffix_toks : Int[Tensor, "batch seq2"], 
                           model, 
                           fwd_hooks = []):
    
    n = len(suffix_toks)
    kv_cache = HookedTransformerKeyValueCache.init_cache(
        model.cfg, model.cfg.device, 1
    )
    model(prefix_toks, past_kv_cache=kv_cache)
    kv_cache.freeze()

    # repeat the cache for every suffix
    # using expand to avoid copying the data
    for entry in kv_cache:
        entry.past_keys = entry.past_keys.expand(n, -1, -1, -1)
        entry.past_values = entry.past_values.expand(n, -1, -1, -1)
    kv_cache.previous_attention_mask = kv_cache.previous_attention_mask.expand(n, -1)
    
    hookpoint_names_filter = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
    
    with model.hooks(fwd_hooks=fwd_hooks):
        output, cache = model.run_with_cache(suffix_toks, names_filter=hookpoint_names_filter, past_kv_cache=kv_cache)
    return output, cache
    
    

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
    
@torch.no_grad
def logit_lens(prompt : Union[str, Int[Tensor, "seq"]],
                model,
                tuned_lens = None, 
                intervention=None, **kwargs) -> Int[Tensor, "num_layers vocab"]:
    
    device = next(model.parameters()).device
    
    use_tuned_lens = kwargs.get('use_tuned_lens', False)
    
    #return_float = kwargs.get('return_float', False)
    
    fwd_hooks = [] if intervention is None else intervention.fwd_hooks(model, **kwargs)
    all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
  
    if isinstance(prompt, str):
        prompt = model.tokenizer.encode(prompt, return_tensors="pt").to(device)
        
    with model.hooks(fwd_hooks = fwd_hooks):
        logits, cache = model.run_with_cache(prompt, names_filter=all_post_resid)
        
    latents = torch.stack([resid[0, -1] for resid in cache.values()], dim=0) # (num_layers, d_model)
    
    if use_tuned_lens:
        approx_logits = model.tuned_lens(latents)
    else:
        approx_logits = model.unembed(model.ln_final(latents))
        
    return approx_logits # (num_layers, vocab)
    
@torch.no_grad
def logit_lens_batched(kv_cache : HookedTransformerKeyValueCache, 
                       suffix_toks : TokenizedSuffixesResult,
                       model,
                       lang_idx : LangIdx,
                       tuned_lens = None, 
                       intervention=None,
                       **kwargs):
    
    batch_size = kwargs.get('batch_size', 1)
    use_tuned_lens = kwargs.get('use_tuned_lens', False)
    
    #return_float = kwargs.get('return_float', False)
    
    fwd_hooks = [] if intervention is None else intervention.fwd_hooks(model, **kwargs)
    all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
    
    dataset = MixedDataset(suffix_toks, lang_idx)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=MixedDataset.collate_fn)

    runner = tqdm(dataloader, total=len(suffix_toks), desc="Computing logits", position=0, leave=True)
    
    all_src_logprobs = []
    all_latent_logprobs = []
    all_dest_logprobs = []
    
    for i, (toks, toks_mask, toks_idx, src_idx, latent_idx, dest_idx) in enumerate(runner):
        _, cache = run_with_kv_cache(toks, kv_cache, model, fwd_hooks = fwd_hooks, names_filter = all_post_resid)
        
        latents = torch.stack(tuple(cache.values()), dim=1) # (batch, num_layers, seq, d_model)
        latents = eindex(latents, toks_idx, "batch num_layers [batch] d_model") # (batch, num_layers, d_model)
        
        if use_tuned_lens:
            approx_logits = model.tuned_lens(latents)
        else:
            approx_logits = model.unembed(model.ln_final(latents))
            
        logprobs = torch.log_softmax(approx_logits, dim=-1) # (batch, num_layers, vocab)
        
        
        for batch, (src, latent, dest) in enumerate(zip(src_idx, latent_idx, dest_idx)):
            src_logprobs = torch.logsumexp(logprobs[batch, :, src], dim=-1)  # (n_layers)
            latent_logprobs = torch.logsumexp(logprobs[batch, :, latent], dim=-1)  # (n_layers)
            dest_logprobs = torch.logsumexp(logprobs[batch, :, dest], dim=-1)  # (n_layers)
            all_src_logprobs.append(src_logprobs)
            all_latent_logprobs.append(latent_logprobs)
            all_dest_logprobs.append(dest_logprobs)
        
        runner.update(len(toks))
        
    all_src_logprobs = torch.stack(all_src_logprobs, dim=0) #(bs, n)
    all_latent_logprobs = torch.stack(all_latent_logprobs, dim=0) #(bs, n)
    all_dest_logprobs = torch.stack(all_dest_logprobs, dim=0) #(bs, n)
    return torch.stack([all_src_logprobs, all_latent_logprobs, all_dest_logprobs], dim=0) #(lang, bs, n)


# @torch.no_grad
# def logit_lens_batched(kv_cache : HookedTransformerKeyValueCache, 
#                        suffix_toks : Int[Tensor, "batch seq2"],
#                        model,
#                        lang_idx : Int[Tensor, "lang batch"],
#                        tuned_lens = None, 
#                        intervention=None,
#                        **kwargs):
    
#     batch_size = kwargs.get('batch_size', 1)
#     src_lang = kwargs.get('src_lang', 'fr')
#     latent_lang = kwargs.get('latent_lang', 'en')
#     dest_lang = kwargs.get('dest_lang', 'zh')
#     use_tuned_lens = kwargs.get('use_tuned_lens', False)
#     use_tuned_lens = kwargs.get('use_tuned_lens', False)
    
#     device = next(model.parameters()).device
#     lang_idx = lang_idx.to(device)
#     suffix_toks = suffix_toks.to(device)
    
#     #return_float = kwargs.get('return_float', False)
    
#     fwd_hooks = [] if intervention is None else intervention.fwd_hooks(model, **kwargs)
#     all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
    
#     dataset = TensorDataset(suffix_toks, lang_idx)
#     dataloader = DataLoader(dataset, batch_size=batch_size)
            
    
#     # suffix_toks_batched = torch.split(suffix_toks, batch_size, dim=0)
    
#     # latents = [act[:, -1, :] for act in cache.values()]
#     # latents = torch.stack(latents, dim=1) #(batch, num_layers, d_model)

#     runner = tqdm(dataloader, total=len(suffix_toks), desc="Computing logits", position=0, leave=True)
    
#     all_probs = []
    
#     for i, (toks, idx) in enumerate(runner):
#         cache = prefix.run_with_kv_cache(toks, kv_cache, model, fwd_hooks = fwd_hooks, names_filter = all_post_resid).cache    
        
#         latents = [act[:, -1, :] for act in cache.values()] # List of (batch, d_model)
#         latents = torch.stack(latents, dim=1) # (batch, num_layers, d_model)
        
#         if use_tuned_lens:
#             approx_logits = model.tuned_lens(latents)
#         else:
#             approx_logits = model.unembed(latents)
            
#         probs = torch.softmax(approx_logits, dim=-1)
        
#         probs = eindex(probs, idx, "bs n_layer [bs lang] -> lang n_layer bs")
#         all_probs.append(probs)
#         runner.update(len(toks))
#     all_probs = torch.cat(all_probs, dim=-1)
#     return all_probs
    
# %%
@torch.no_grad
def get_logits(dataset, model, intervention=None, **kwargs):
    """
    Compute the logits for a given dataset using a language model.

    Args:
        dataset (list): The dataset to compute logits for.
        intervention (object, optional): An intervention object for forward hooks. Defaults to None.
        tuned_lens (object, optional): An object for tuning the latents. Defaults to None.
        model (object, optional): The language model. Defaults to model.
        tokenizer (object, optional): The tokenizer. Defaults to tokenizer.

    Returns:
        torch.Tensor: The computed logits for the dataset.
    """
    
    device = next(model.parameters()).device
    tokenizer = model.tokenizer
    use_tuned_lens = kwargs.get('use_tuned_lens', False)
    return_float = kwargs.get('return_float', False)
    cache_prefix = kwargs.get('cache_prefix', False)
    
    print(f"Kwargs: {kwargs}") 
    def get_latents(tokens, datapoint):
        """
        Get the latents for a given input sequence.

        Args:
            tokens (torch.Tensor): The input sequence tokens.
            datapoint (object, optional): The datapoint object. Defaults to None.

        Returns:
            torch.Tensor: The latents for the input sequence.
        """
        all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
        hook_scale = ['unembed.final_norm.hook_scale']
        hookpoint_names_filter = hook_scale + all_post_resid
        
        
        if intervention is None:
            fwd_hooks = []
        else:
            fwd_hooks = intervention.fwd_hooks(model, **datapoint, **kwargs)
                
        with model.hooks(fwd_hooks=fwd_hooks):
            output, cache = model.run_with_cache(tokens, names_filter=hookpoint_names_filter)
            
        latents = [act[:, -1, :] for act in cache.values()]
        latents = torch.stack(latents, dim=1)
        return latents
    
    # def unemb(latents):
    #     """
    #     Compute the logits for a given set of latents.
    #     """
    #     latents_ln = model.ln_final(latents)
    #     logits = latents_ln @ model.unembed.W_U.as_type(latents_ln) + model.unembed.b_U.as_type(latents_ln)
    #     return logits 
    
    all_logits = []
        
    with torch.no_grad():
        for idx, d in tqdm(enumerate(dataset), total=len(dataset)):
            
            tokens = tokenizer.encode(d['prompt'], return_tensors="pt").to(device)
            latents = get_latents(tokens, d)
            if use_tuned_lens:
                logits = torch.stack([model.tuned_lens(latents[:, i], i) for i in range(model.cfg.n_layers)], dim=1)
            else:
                logits = model.unembed(latents)
            
            all_logits.append(logits)
        
    all_logits = torch.stack(all_logits)
    
    if return_float:
        all_logits = all_logits.float()
        latents = latents.float()
    
    return latents.squeeze(), all_logits.squeeze()

# @torch.no_grad
# def logit_lens(dataset, model, tuned_lens=None):
#     """
#     Measure language probabilities for a given dataset.

#     Args:
#         dataset (iterable): The dataset to measure language probabilities on.
#         steer (str, optional): The steering method. Defaults to None.
#             unembed: Use the unembeeded vectors of the latent ids
#             embed: Use the embedded vectors of the latent ids
#             both: Use both the unembeeded and embedded vectors of the latent ids
#         model (Model, optional): The language model. Defaults to model.
#         tokenizer (Tokenizer, optional): The tokenizer. Defaults to tokenizer.
#         device (str, optional): The device to run the model on. Defaults to device.

#     Returns:
#         tuple: Logits from each layer. You work out what to do with it.
#     """
#     tokenizer = model.tokenizer
#     device = next(model.parameters()).device
    
#     def get_latents(tokens, model):
#         all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
#         # if latent_ids is None:
#         #     output, cache = model.run_with_cache(tokens, names_filter=all_post_resid)
#         # else:    
#         #     subspace = model.unembed.W_U.T[latent_ids]
        
#         with model.hooks(fwd_hooks=[]):
#             output, cache = model.run_with_cache(tokens, names_filter=all_post_resid)
            
#         latents = [act[:, -1, :] for act in cache.values()]
#         #latents = [cache[f'blocks.{i}.hook_resid_post'][:, -1, :] for i in range(model.cfg.n_layers)] 
#         latents = torch.stack(latents, dim=1)
#         return latents #(batch=1, num_layers, d_model)
    
#     def unemb(latents, model):
#         latents_ln = model.ln_final(latents)
#         logits = latents_ln @ model.unembed.W_U + model.unembed.b_U
#         return logits 

#     all_logits = []
        
#     with torch.no_grad():
#         for idx, d in tqdm(enumerate(dataset), total=len(dataset)):
            
#             latent_ids = d['latent_ids']
#             out_ids = d['out_ids']
            
#             tokens = tokenizer.encode(d['prompt'], return_tensors="pt").to(device)
            
#             latents = get_latents(tokens, model)
#             if tuned_lens is not None:
#                 logits = torch.stack([tuned_lens(latents[:,i],i) for i in range(model.cfg.n_layers)], dim=1)
#             else:
#                 logits = unemb(latents, model) #(batch=1, num_layers, vocab_size)
#             #last = logits.softmax(dim=-1).detach().cpu().squeeze()
#             all_logits.append(logits)
#     all_logits = torch.stack(all_logits)
#     return all_logits.float().squeeze()

def compute_layer_probs(logprobs: Float[Tensor, "num_data num_layer vocab"],
                        token_ids: List[Int[Tensor, "num_idx"]],
    ) -> Float[Tensor, "datapoints num_layers"]:
        """
        Compute the layer probabilities for each token ID.

        Args:
            probs (List[Float[Tensor, "num_vocab"]]): The probabilities for each token ID.
            token_ids (List[List[int]]): The token IDs for each datapoint.

        Returns:
            Float[Tensor, "datapoints num_layers"]: The layer probabilities for each datapoint.
        """
        layer_logprobs = []
        for i, tok_id in enumerate(token_ids):
            layer_logprob = torch.logsumexp(logprobs[i, :, tok_id], dim=-1) #(num_layers)
            layer_logprobs.append(layer_logprob.detach().cpu())
        return torch.stack(layer_logprobs)

from matplotlib.offsetbox import AnchoredText

def plotter(logprobs_list, label_list, **kwargs):
    
    out_path = kwargs.get('out_path', None)
    title = kwargs.get('title', None)
    figsize = kwargs.get('figsize', (6, 12))
    stats = kwargs.get('stats', None)
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=figsize)
    
    if stats is not None:
        stats_text = "\n".join(f"{key}: {value:.03f}" for key, value in stats.items())
        # fig.text(0.75, 0.5, stats_text, 
        #          bbox=dict(facecolor='white', alpha=0.5), 
        #          fontsize=10, 
        #          verticalalignment='upper left')
        
        anchored_text = AnchoredText(stats_text, loc='center left', borderpad=1.0, frameon=True)
        anchored_text.patch.set_facecolor('white')
        anchored_text.patch.set_alpha(0.5)
        anchored_text.prop.set_size(12)  # Set the font size to 8
        ax2.add_artist(anchored_text)  # Add to the first subplot (ax1) or choose another as needed
    
    for logprobs, label in zip(logprobs_list, label_list):
        plot_ci(logprobs, ax1, dim=0, label=label)
        plot_ci(torch.exp(logprobs), ax2, dim=0, label=label)
    plt.legend()
    fig.suptitle(title)
    fig.tight_layout()  # Add this line to reduce the gap between subplots and title
    ax2.set_xlabel('Layer')
    ax1.set_ylabel('Log Probability')
    ax2.set_ylabel('Raw Probability')
    ax1.grid(True, which='both', linestyle=':', linewidth=0.5)  # Add minor gridlines to ax1
    ax2.grid(True, which='both', linestyle=':', linewidth=0.5)  # Add minor gridlines to ax2
    plt.grid(True, which='both', linestyle=':', linewidth=0.5)  # Add minor gridlines to the whole figure
    if out_path is not None:
        plt.savefig(out_path, format='svg')
    plt.show()


def plot_logit_lens_latents(logits : Float[Tensor, "num_data num_layer vocab"], 
                            dataset,
                            only_compute_stats = True,
                            **kwargs):
    
    #only_compute_stats = kwargs.get('only_compute_stats', True)
    cfg = kwargs.get('cfg', None)
    dest_lang = cfg.dest_lang
    latent_lang = cfg.latent_lang
    
    logprobs = F.log_softmax(logits, dim=-1)
    logprobs_list = []
    
    id_names = ['latent_ids', 'out_ids', 'alt_latent_ids', 'alt_out_ids']
    for name in id_names:
        ids = ([d[name] for d in dataset])
        logprobs_list.append(compute_layer_probs(logprobs, ids))
    correct_logprobs = logprobs_list[1][:, -1].mean().item()
    alt_logprobs = logprobs_list[3][:, -1].mean().item()
    correct_probs = torch.exp(logprobs_list[1][:, -1]).mean().item()
    alt_probs = torch.exp(logprobs_list[3][:, -1]).mean().item()
    stats = {'lp_out' : correct_logprobs, 
             'lp_alt' : alt_logprobs,
             'p_out' : correct_probs,
                'p_alt' : alt_probs,
             'lp_diff' : alt_logprobs - correct_logprobs,
             'p_ratio' : alt_probs / correct_probs}
    #if not only_compute_stats:
    #plotter(logprobs_list, [latent_lang, dest_lang, latent_lang + "_alt", dest_lang + "_alt"], stats = stats, **kwargs)
    return stats

def latent_heatmap(data, num_bins=250, bin_range=(0, 5), **kwargs):
    """
    Plot a heatmap of histograms for each row of the input data with a fixed range and specified bins.
    Uses logarithmic coloring for both imshow and the colorbar.

    Args:
    - data: A tensor of shape (num_rows, num_values).
    - num_bins: Number of bins to use for the histograms.
    - bin_range: The range for the histogram bins.

    Returns:
    - None
    """
    title = kwargs.get('title', None)
    
    data_np = data.cpu().numpy()
    
    # Calculate histograms for each row
    histograms = []
    for row in data_np:
        hist, bins = np.histogram(row, bins=num_bins, range=bin_range, density=True)
        histograms.append(hist)

    # Convert the list of histograms to a 2D array
    histograms_np = np.maximum(1.0 * np.array(histograms).T, 1e-3)[::-1]

    # Plot the heatmap using Matplotlib with logarithmic coloring
    plt.figure(figsize=(12, 8))
    plt.imshow(histograms_np, aspect='auto', cmap='viridis', norm=LogNorm(), extent=[0, data.shape[0], bin_range[0],bin_range[1]])
    cbar = plt.colorbar(label='Density', norm=LogNorm())
    cbar.set_label('Log Density')
    plt.title(title)
    plt.ylabel('Activation Magnitude')
    plt.xlabel('Layer')
    plt.show()



# %%
# @torch.no_grad
# def batch_logit_lens(nn_model, prompts, only_last_token=True):
#     """
#     Compute the logits for each layer of a neural network model given a set of prompts.

#     Args:
#         nn_model (torch.nn.Module): The neural network model.
#         prompts (list[str]): The list of prompts.
#         only_last_token (bool, optional): Whether to consider only the last token of each prompt. 
#             Defaults to True.

#     Returns:
#         torch.Tensor: The logits per layer of the model.

#     """
#     model.eval()
#     tok_prompts = tokenizer(prompts, return_tensors="pt", padding=True)
#     # Todo?: This is a hacky way to get the last token index for each prompt
#     last_token_index = tok_prompts.attention_mask.cumsum(1).argmax(-1)
    
#     output, cache = model.run_with_cache(prompts) #Expensive!
    
#     hidden_l = []
    
#     for i in range(model.cfg.n_layers):
#         layer_cache = cache[f'blocks.{i}.hook_resid_post']  # (batch, seq, d_model)
#         if only_last_token:
#             layer_cache = eindex(layer_cache, last_token_index, "i [i] j") # (batch, d_model)
#         hidden_l.append(layer_cache) # (batch, seq?, d_model)
            
#     hidden = torch.stack(hidden_l, dim=1)  # (batch, num_layers, seq?, d_model)
#     rms_out_ln = model.ln_final(hidden) # (batch, num_layers, seq?, d_model)
#     logits_per_layer = model.unembed(rms_out_ln) # (batch, num_layers, seq?, vocab_size)
    
#     return logits_per_layer
# %%




# %%
# prompt_zh_to_fr = "中文:花 français: fleur 中文:山 français: montagne 中文:月 français: lune 中文:水 français: eau"

# df = pd.read_csv('data/test/single_tok_lang6.csv')

# # %%
# def find_common_prefix(strings):
#     if not strings:
#         return ""
    
#     prefix = strings[0]
#     for string in strings[1:]:
#         while not string.startswith(prefix):
#             prefix = prefix[:-1]
#             if not prefix:
#                 return ""
    
#     return prefix


# @torch.no_grad
# def logit_lens_fast(nn_model, prompts):
#     """
#     Get the probabilities of the next token for the last token of each prompt at each layer using the logit lens.

#     Args:
#         nn_model: NNSight LanguageModel object
#         prompts: List of prompts or a single prompt
#         qk_caching: Perform a forward pass on the common prefix of the prompts to cache the query and key tensors.

#     Returns:
#         A tensor of shape (num_layers, num_prompts, vocab_size) containing the probabilities
#         of the next token for each prompt at each layer. Tensor is on the CPU.
#     """
    
#     prefix = find_common_prefix(prompts)
    
#     model.eval()
#     tok_prompts = tokenizer(prompts, return_tensors="pt", padding=True)
#     # Todo?: This is a hacky way to get the last token index for each prompt
#     last_token_index = tok_prompts.attention_mask.cumsum(1).argmax(-1)
    
#     output, cache = model.run_with_cache(prompts) #Expensive!
    
#     hidden_l = []
#     for i in range(model.cfg.n_layers):
#         hidden_l.append(eindex(cache[f'blocks.{i}.hook_resid_post'], last_token_index, "i [i] j"))
    
#     hiddens = torch.stack(hidden_l, dim=0)  # (num_layers, num_prompts, d_model) 
#     rms_out_ln = model.ln_final(hiddens) # (num_layers, num_prompts, d_model)
#     logits_per_layer = model.unembed(rms_out_ln) # (num_layers, num_prompts, vocab_size)
#     probs_per_layer = logits.softmax(-1) # (num_layers, num_prompts, vocab_size)
    
    
#     assert torch.allclose(
#         logits_per_layer[-1],
#         eindex(output, last_token_index, "i [i] j")
#     )
#     return probs_per_layer

# # %%
# # Common prefix
# common_prefix = "The cat sat on the"

# # Unique suffixes
# suffixes = ["mat.", "rug.", "floor."]

# # Tokenize the common prefix
# tokens_prefix = tokenizer(common_prefix, return_tensors="pt")
# with torch.no_grad():
#     # Forward pass for the common prefix to obtain the last hidden state
#     outputs_prefix = model(**tokens_prefix, output_hidden_states=True)
#     hidden_states_prefix = outputs_prefix.hidden_states[-1][:, -1, :]

# # Process each suffix
# for suffix in suffixes:
#     # Tokenize suffix
#     tokens_suffix = tokenizer(suffix, return_tensors="pt")
#     # Concatenate the last hidden state of the prefix with the input ids of the suffix
#     # Adjust depending on whether you need to include the prefix's last token for context
#     inputs_with_state = {"input_ids": tokens_suffix["input_ids"], "past_key_values": outputs_prefix.past_key_values}
    
#     with torch.no_grad():
#         # Forward pass for the suffix, reusing the hidden state
#         outputs_suffix = model(**inputs_with_state)
    
#     # Convert logits to probabilities
#     probabilities = torch.softmax(outputs_suffix.logits[:, -1, :], dim=-1)
    
#     # Here you can extract and process the predictions as needed
#     print(f"Processed suffix '{suffix}'")