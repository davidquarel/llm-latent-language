from collections import namedtuple
from typing import List, Tuple, Callable, Optional
from transformer_lens import HookedTransformerKeyValueCache, HookedTransformer
from jaxtyping import Int
from torch import Tensor
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader, TensorDataset
from src.schema import TokenizedSuffixesResult
from eindex import eindex
from transformer_lens.past_key_value_caching import HookedTransformerKeyValueCacheEntry


def broadcast_kv_cache(kv_cache : HookedTransformerKeyValueCache, batch : int):
    """
    Broadcasts the key-value kv_cache for parallel processing, reshaping its elements
    from (a, b, c, d) to (n, b, c, d), assuming all elements in dimension 'a' are identical
    and can be replicated to dimension 'batch'.

    Args:
        kv_cache (object): The key-value cache object.
        n (int): The number of parallel processes.

    Returns:
        None
    """
    for e in kv_cache:
        if e.past_keys.dim() == 4 and e.past_keys.size(0) > 1:
            # Assuming the first dimension has redundant copies, we take one and expand it
            e.past_keys = e.past_keys[0].unsqueeze(0).expand(batch, -1, -1, -1)
            e.past_values = e.past_values[0].unsqueeze(0).expand(batch, -1, -1, -1)
        else:
            # If already in correct form or not expanded, simply adjust the dimensions
            e.past_keys = e.past_keys.expand(batch, -1, -1, -1)
            e.past_values = e.past_values.expand(batch, -1, -1, -1)
    if kv_cache.previous_attention_mask.dim() == 2 and kv_cache.previous_attention_mask.size(0) > 1:
        # Similarly adjust the attention mask
        kv_cache.previous_attention_mask = kv_cache.previous_attention_mask[0].unsqueeze(0).expand(batch, -1)
    else:
        kv_cache.previous_attention_mask = kv_cache.previous_attention_mask.expand(batch, -1)

    
def gen_kv_cache(prompt : str | Int[Tensor, "batch seq"] | Int[Tensor, "seq"],
                 model : HookedTransformer
) -> HookedTransformerKeyValueCache:
    device = next(model.parameters()).device
    if isinstance(prompt, str):
        prompt = model.tokenizer.encode(prompt, return_tensors="pt").to(device)
    else:
        prompt = prompt.to(device)
    if prompt.dim() == 1:
        bs = 1
    else:
        bs = prompt.size(0)
        
    kv_cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, bs) # flush cache
    model(prompt, past_kv_cache = kv_cache) #fill kv_cache
    kv_cache.freeze()
    return kv_cache
    

# %%
from torch.utils.data import TensorDataset, DataLoader
from src.schema import TokenizedSuffixesResult
import copy

RunWithKVCacheResult = namedtuple('RunWithKVCacheResult', ['logits', 'cache'], defaults=[None])


@torch.no_grad()
def run_with_kv_cache(tokens : Int[Tensor, "batch seq"] | TokenizedSuffixesResult,
                    past_kv_cache : HookedTransformerKeyValueCache,
                    model : HookedTransformer,
                    fwd_hooks : List[Callable] = [],
                    mini_batch_size : Optional[int] = None,
                    last_seq : bool = False,
                    keep_resid_cache : bool = False,
                    verbose : bool = False,
                    **kwargs
) -> Tuple[Tensor, Tensor]:
    batch, seq = tokens.input_ids.shape
    d_model, d_vocab = model.cfg.d_model, model.cfg.d_vocab
    mini_batch_size = mini_batch_size if mini_batch_size is not None else batch
    
    if keep_resid_cache:
        resid_name_filter = ["hook_embed"] + [f"blocks.{x}.hook_resid_post" for x in range(model.cfg.n_layers)]
    else:
        resid_name_filter = []

    if isinstance(tokens, TokenizedSuffixesResult):
        if tokens.attention_mask is None:
            tokens.attention_mask = torch.ones_like(tokens.input_ids)

    tensor_dataset = TensorDataset(*tokens.to_tuple())
    data_loader = DataLoader(tensor_dataset, batch_size=mini_batch_size, shuffle=False)
   

    past_kv_cache.freeze()
    broadcast_kv_cache(past_kv_cache, mini_batch_size)

    #dummy run to get shapes
    sample_logits, sample_cache = model.run_with_cache(tokens.input_ids[:1])

    seq_shape = () if last_seq else (seq,)
    
    all_logits = torch.empty((batch, *seq_shape, d_vocab), 
                            device=sample_logits.device,
                            dtype=sample_logits.dtype)
    all_cache = {}
    for key in sample_cache:
        if key in resid_name_filter:
            # only works for resid
            all_cache[key] = torch.empty((batch, *seq_shape, d_model),
                                        device=sample_cache[key].device,
                                        dtype=sample_cache[key].dtype)
        
    if verbose:
        runner = tqdm(total=batch)
    tok_c = 0
    for i, chunk in enumerate(data_loader):
        chunk_tokens, chunk_mask, chunk_ids = chunk
        chunk_size = chunk_tokens.shape[0]
        if chunk_size != mini_batch_size:
            broadcast_kv_cache(past_kv_cache, chunk_size)
        
        with model.hooks(fwd_hooks = fwd_hooks):
            logits, cache = model.run_with_cache(chunk_tokens,
                                                past_kv_cache=past_kv_cache,
                                                names_filter=resid_name_filter, 
                                                attention_mask = chunk_mask,
                                                **kwargs)
            if last_seq:
                # Index while on GPU
                logits = eindex(logits, chunk_ids, "b [b] v -> b v")
            all_logits[tok_c:tok_c+chunk_size] = logits

            for key, value in cache.items():
                if last_seq:
                    value = eindex(value, chunk_ids, "b [b] m -> b m")
                    assert value.shape == (chunk_size, d_model)
                all_cache[key][tok_c:tok_c+chunk_size] = value
        if verbose:
            runner.update(chunk_size)
        tok_c += chunk_size
        
    return RunWithKVCacheResult(logits=all_logits, cache=all_cache)
    
# %%
    # with model.hooks(fwd_hooks = fwd_hooks):
    #     if names_filter == []:
    #         logits = model(tokens, past_kv_cache=kv_cache)
    #         return RunWithKVCacheResult(logits=logits, cache=None)
    #     else:
    #         logits, cache = model.run_with_cache(tokens, past_kv_cache=kv_cache, names_filter=names_filter)
    #         return RunWithKVCacheResult(logits=logits, cache=cache)    
    
def batched_predict_next(kv_cache : HookedTransformerKeyValueCache,
                           suffixes_toks : Int[Tensor, "batch seq"],
                           model,
                           fwd_hooks = [],
                           hooks_filter = [], 
                           batch_size = 1,
                           return_logits = False,
                           **kwargs):
    
    desc = kwargs.get("desc", "")
    position = kwargs.get("position", 0)
    leave = kwargs.get("leave", True)
    # assume all suffixes tokenize to the same number of tokens
    all_outs = []
    all_toks = []

    suffix_toks_batched = torch.split(suffixes_toks, batch_size, dim=0)
    
    runner = tqdm(suffix_toks_batched, total=len(suffixes_toks), desc=desc, position=position, leave=leave)
    
    for batch in runner:
        logits = run_with_kv_cache(batch, kv_cache, model, fwd_hooks, hooks_filter).logits[:, -1].detach()
        if return_logits:
            outs = logits
        else:
            outs = torch.softmax(logits, dim=-1)
        max_outs, max_tokens = torch.max(outs, dim=-1)
        
        all_outs.append(max_outs)
        all_toks.append(max_tokens)
        runner.update(len(batch))
        
    all_outs = torch.cat(all_outs, dim=0)
    all_toks = torch.cat(all_toks, dim=0)
    return all_outs, all_toks