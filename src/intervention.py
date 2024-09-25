from typing import Any, List, Callable, Dict, Iterable
from torch import Tensor
import torch
from jaxtyping import Int, Float
from transformer_lens.hook_points import HookPoint
from src.llm import proj
from beartype import beartype
import inspect
import re
# class Intervention:
#     def __init__(self, func: Callable[..., Tensor], layers: List[int]):
#         self.func = func
#         self.layers = layers
        
#     def apply(self, resid: Tensor, hook: Any, model, **kwargs) -> Tensor:
#         return self.func(resid, hook, model, **kwargs)

#     def fwd_hooks(self, model: Any, **kwargs) -> List:
#         temp_hook_fn = lambda resid, hook: self.apply(resid, hook, model, **kwargs)
#         return [(f'blocks.{j}.hook_resid_post', temp_hook_fn) for j in self.layers]

class Intervention:
    def __init__(self, func_name: str, layers: Iterable[int]):
        # Check if the function name exists in the global scope and is callable
        if func_name in globals() and callable(globals()[func_name]):
            self.func = globals()[func_name]
            # Attempt to retrieve the source code of the function
            try:
                self.description = inspect.getsource(self.func)
            except OSError:
                self.description = f"Source code for '{func_name}' not available."
        else:
            raise ValueError(f"No such function '{func_name}' found in global scope.")

        self.layers = list(layers)
        self.func_name = func_name
        
    def apply(self, resid: Tensor, hook: Any, model, **kwargs) -> Tensor:
        return self.func(resid, hook, model, **kwargs)

    def fwd_hooks(self, model: Any, **kwargs) -> List:
        # Using a lambda to capture the current model and additional args
        temp_hook_fn = lambda resid, hook: self.apply(resid, hook, model, **kwargs)
        return [(f'blocks.{j}.hook_resid_post', temp_hook_fn) for j in self.layers]

# Example usage assuming hook functions are defined and available globally
# hook_reject_subspace is supposed to be a previously defined function

def hook_move_subspace(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids: Int[Tensor, "num_latent_tokens"] = None,
    alt_latent_ids: Int[Tensor, "num_alt_latent_tokens"] = None,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    
    subspace = model.unembed.W_U.T[latent_ids]
    subspace_alt = model.unembed.W_U.T[alt_latent_ids]
    
    v = resid[:, -1]
    # subspace = W_U.T[latent_tok_ids]
    proj_correct = proj(v.float(), subspace.float()).half()
    #resid_alt = proj(last_tblock.float(), subspace_alt.float())
    proj_counter = proj(v.float(), subspace_alt.float()).half()
    resid[:, -1] = v - proj_correct + proj_counter
    return resid

def hook_move_subspace2(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids: Int[Tensor, "num_latent_tokens"] = None,
    alt_latent_ids: Int[Tensor, "num_alt_latent_tokens"] = None,
    scale_coeff : Float = 1,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    subspace = model.unembed.W_U.T[latent_ids]
    subspace_alt = model.unembed.W_U.T[alt_latent_ids]
    
    v = resid[:, -1]
    # subspace = W_U.T[latent_tok_ids]
    proj_correct = proj(v.float(), subspace.float()).half()
    #resid_alt = proj(last_tblock.float(), subspace_alt.float())
    proj_counter = proj(proj_correct.float(), subspace_alt.float()).half()
    resid[:, -1] = v - proj_correct + scale_coeff * proj_counter
    return resid

def hook_reject_tok(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids : Int[Tensor, "num_latent_tokens"] = None,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    # modify attn_pattern (can be inplace)
    subspace = model.unembed.W_U.T[latent_ids]
    last_tblock = resid[:, -1]
    # subspace = W_U.T[latent_tok_ids]
    last_tblock = last_tblock - proj(last_tblock.float(), subspace.float())
    resid[:, -1] = last_tblock
    return resid

def hook_diff_subspace(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids: Int[Tensor, "num_latent_tokens"] = None,
    alt_latent_ids: Int[Tensor, "num_alt_latent_tokens"] = None,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    interv_steer_coeff = kwargs.get('interv_steer_coeff', None)
    assert interv_steer_coeff is not None, "interv_steer_coeff must be provided"
    subspace_latent = model.unembed.W_U.T[latent_ids]
    latent_vec = subspace_latent.mean(dim=0)
    alt_latent_vec = model.unembed.W_U.T[alt_latent_ids].mean(dim=0)
    v = resid[:, -1]
    proj_latent = proj(v.float(), subspace_latent.float()).half()
    #print(v.shape, latent_vec.shape, alt_latent_vec.shape)
    resid[:, -1] =  v - proj_latent + interv_steer_coeff * torch.linalg.norm(proj_latent) * (alt_latent_vec - latent_vec)
    return resid

def hook_only_new_subspace(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids: Int[Tensor, "num_latent_tokens"] = None,
    alt_latent_ids: Int[Tensor, "num_alt_latent_tokens"] = None,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    interv_steer_coeff = kwargs.get('interv_steer_coeff', None)
    assert interv_steer_coeff is not None, "interv_steer_coeff must be provided"
    subspace_latent = model.unembed.W_U.T[latent_ids]
    latent_vec = subspace_latent.mean(dim=0)
    alt_latent_vec = model.unembed.W_U.T[alt_latent_ids].mean(dim=0)
    v = resid[:, -1]
    proj_latent = proj(v.float(), subspace_latent.float()).half()
    #print(v.shape, latent_vec.shape, alt_latent_vec.shape)
    resid[:, -1] =  v - proj_latent + interv_steer_coeff * alt_latent_vec
    return resid


def hook_reject_subspace(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids : Int[Tensor, "num_latent_tokens"] = None,
    alt_latent_ids : Int[Tensor, "num_alt_latent_tokens"] = None,
    interv_match_latent : bool = True,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    # modify attn_pattern (can be inplace)
    if interv_match_latent:
        subspace = model.unembed.W_U.T[latent_ids]
    else:
        subspace = model.unembed.W_U.T[alt_latent_ids]
        
    last_tblock = resid[:, -1]
    # subspace = W_U.T[latent_tok_ids]
    last_tblock = last_tblock - proj(last_tblock.float(), subspace.float())
    resid[:, -1] = last_tblock
    return resid

def hook_reject_subspace_v2(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids : Int[Tensor, "num_latent_tokens"] = None,
    alt_latent_ids : Int[Tensor, "num_alt_latent_tokens"] = None,
    interv_match_latent : bool = True,
    cache : Dict = None,
    use_reverse_lens : bool = False,
    rev_lens_scale : float = None,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    # modify attn_pattern (can be inplace)
    
     # Define the regular expression pattern

    idx = latent_ids if interv_match_latent else alt_latent_ids
    
    if use_reverse_lens:
        #assert cache is not None, "cache required for reverse_lens"
        assert rev_lens_scale is not None, "rev_lens_scale required for reverse_lens"
        pattern = r'^blocks\.(\d+)\.hook_resid_post$'
        match = re.match(pattern, hook.name)
        if match:
            layer = int(match.group(1))
        else:
            raise ValueError(f"String '{hook.name}' no match 'blocks.<number>.hook_resid_post'")
        #print(f"hello! I'm here! {rev_lens_scale}")
        subspace = model.reverse_lens(idx, layer, rev_lens_scale, use_logits=False)
        
    else:
        subspace = model.unembed.W_U.T[idx]
        
    last_tblock = resid[:, -1]
    # subspace = W_U.T[latent_tok_ids]
    last_tblock = last_tblock - proj(last_tblock.float(), subspace.float())
    resid[:, -1] = last_tblock
    return resid


def hook_diff_subspace_v2(
    resid: Float[Tensor, "batch seq dmodel"],
    hook: HookPoint,
    model,
    latent_ids: Int[Tensor, "num_latent_tokens"] = None,
    alt_latent_ids: Int[Tensor, "num_alt_latent_tokens"] = None,
    cache : Dict = None,
    use_reverse_lens : bool = False,
    rev_lens_scale : float = None,
    **kwargs
) -> Float[Tensor, "batch seq dmodel"]:
    interv_steer_coeff = kwargs.get('interv_steer_coeff', None)
    assert interv_steer_coeff is not None, "interv_steer_coeff must be provided"
    
    
    if use_reverse_lens:
        #assert cache is not None, "cache required for reverse_lens"
        assert rev_lens_scale is not None, "rev_lens_scale required for reverse_lens"
        pattern = r'^blocks\.(\d+)\.hook_resid_post$'
        match = re.match(pattern, hook.name)
        if match:
            layer = int(match.group(1))
        else:
            raise ValueError(f"String '{hook.name}' no match 'blocks.<number>.hook_resid_post'")
        #print(f"hello! I'm here! {rev_lens_scale}")
        subspace_alt_latent = model.reverse_lens(latent_ids, layer, rev_lens_scale, use_logits=False).squeeze()
        subspace_latent = model.reverse_lens(alt_latent_ids, layer, rev_lens_scale, use_logits=False).squeeze()
        
        simple_subspace_latent = model.unembed.W_U.T[latent_ids].mean(dim=0)
        simple_subspace_alt_latent = model.unembed.W_U.T[alt_latent_ids].mean(dim=0)
        simple_diff = (simple_subspace_alt_latent - simple_subspace_latent)
        
    else:
        assert False
        subspace_latent = model.unembed.W_U.T[latent_ids]
        subspace_alt_latent = model.unembed.W_U.T[alt_latent_ids]
    
    latent_vec = subspace_latent.mean(dim=0)
    alt_latent_vec = subspace_alt_latent.mean(dim=0)
    v = resid[:, -1]
    proj_latent = proj(v.float(), simple_subspace_latent.float()).half()
    #print(v.shape, latent_vec.shape, alt_latent_vec.shape)
    diff = (alt_latent_vec - latent_vec)
    diff = (torch.linalg.norm(simple_diff) / torch.linalg.norm(diff)) * diff
    resid[:, -1] =  v - proj_latent + 2 * torch.linalg.norm(proj_latent) * diff
    return resid