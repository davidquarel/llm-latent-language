from collections import namedtuple
from typing import List, Tuple, Callable
from transformer_lens import HookedTransformerKeyValueCache, HookedTransformer
from jaxtyping import Int
from torch import Tensor
from tqdm import tqdm
import torch


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
    
RunWithKVCacheResult = namedtuple('RunWithKVCacheResult', ['logits', 'cache'], defaults=[None])
    
def run_with_kv_cache(tokens : Int[Tensor, "batch seq"],
                    past_kv_cache : HookedTransformerKeyValueCache,
                    model : HookedTransformer,
                    fwd_hooks : List[Callable] = [],
                    **kwargs
) -> Tuple[Tensor, Tensor]:

    past_kv_cache.freeze()
    broadcast_kv_cache(past_kv_cache, len(tokens))
    
    with model.hooks(fwd_hooks = fwd_hooks):
        logits, cache = model.run_with_cache(tokens, 
                                             past_kv_cache=past_kv_cache,
                                                **kwargs)
        
    return RunWithKVCacheResult(logits=logits, cache=cache)
    
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