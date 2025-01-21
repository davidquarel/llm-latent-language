import matplotlib.pyplot as plt
import torch
from typing import List
from jaxtyping import Float, Int
from torch import Tensor
import torch
from utils.misc import printd
from typing import List, Tuple, Callable
from jaxtyping import Int
from dataclasses import dataclass
from typing import Optional
from tqdm.auto import tqdm
from transformer_lens import HookedTransformerKeyValueCache, HookedTransformer
from transformers import AutoTokenizer
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from collections import namedtuple
from einops import rearrange
import warnings
from .constants import LANG_TO_NAME
from .prompt import gen_prompt, gen_common_suffixes
from .kv_cache import gen_kv_cache, run_with_kv_cache, batched_predict_next

# %%

def proj(x : Float[Tensor, "batch dmodel"], 
         Y : Float[Tensor, "numvec dmodel"],
         tol : float = 1e-8 
) -> Float[Tensor, "... dmodel"]:
    # Computes the projection of x onto the subspace spanned by the columns of Y
    Y = ensure_3d(Y)
        
    zero_rows = torch.all(Y==0, dim=-1)
    Y = Y[~zero_rows]
        
    Y = Y.mT #(dmodel, numvec) #require column vectors
    
    # Solve the linear system (Y^T @ Y) @ c = Y^T @ x
    # c is the coefficients of the projection of x onto the subspace spanned by the columns of Y
    # so the projection of x onto the subspace spanned by the columns of Y is Y @ c
    if x.ndim == 1:
        x = x.unsqueeze(0)
    
    c = torch.linalg.solve(Y.mT  @ Y, (x @ Y).mT)    
    proj_x = (Y @ c).mT 
    return proj_x.squeeze()

def proj_batched_slow(x : Float[Tensor, "batch dmodel"], 
             Y : Float[Tensor, "batch numvec dmodel"], 
             epsilon=1e-8) -> Float[Tensor, "batch dmodel"]:
    if x.ndim == 1:
        x = x.unsqueeze(0)
    if Y.ndim == 2:
        Y = Y.unsqueeze(0)
    
    batch_size = x.shape[0]
    results = []
    
    for i in range(batch_size):
        
        zero_rows = torch.norm(Y[i], dim=-1) < 1e-5
        Y_pruned = Y[i][~zero_rows]
        
        result = proj(x[i], Y_pruned)
        results.append(result)
    
    return torch.stack(results).squeeze()


def proj_batched(x : Float[Tensor, "batch dmodel"], 
             Y : Float[Tensor, "batch numvec dmodel"], 
             epsilon=1e-8) -> Float[Tensor, "batch dmodel"]:
    """
    Computes the projection of x onto the subspace spanned by the rows of Y for each batch.

    Args:
        x: Tensor of shape (batch, dmodel)
        Y: Tensor of shape (batch, n, dmodel), may contain zero rows (zero vectors)
        epsilon: Small constant for regularization to handle rank-deficient matrices

    Returns:
        proj_x: Tensor of shape (batch, dmodel), the projection of x onto the subspace spanned by Y
    """
    # x: (B, dmodel)
    # Y: (B, n, dmodel)
    B, n, dmodel = Y.shape

    # Transpose Y to shape (B, dmodel, n)
    Y_t = Y.permute(0, 2, 1)  # (B, dmodel, n)

    # Reshape x to (B, dmodel, 1)
    x_ = x.unsqueeze(-1)  # (B, dmodel, 1)

    # Compute Y^T Y: (B, n, n)
    # Note: Y^T is (B, n, dmodel), Y is (B, dmodel, n)
    YTY = torch.bmm(Y_t.permute(0, 2, 1), Y_t)  # (B, n, n)

    # Regularize YTY by adding epsilon * I to handle rank deficiency
    I = torch.eye(n, device=Y.device).unsqueeze(0).expand(B, n, n)
    YTY_reg = YTY + epsilon * I  # (B, n, n)

    # Compute the pseudoinverse of YTY_reg: (B, n, n)
    YTY_inv = torch.linalg.pinv(YTY_reg)  # (B, n, n)

    # Compute Y^T x: (B, n, 1)
    YT_x = torch.bmm(Y_t.permute(0, 2, 1), x_)  # (B, n, 1)

    # Compute the coefficients c: (B, n, 1)
    c = torch.bmm(YTY_inv, YT_x)  # (B, n, 1)

    # Compute the projection proj_x = Y_t @ c: (B, dmodel, 1)
    proj_x = torch.bmm(Y_t, c).squeeze(-1)  # (B, dmodel)

    return proj_x

def ensure_3d(x):
    dims_to_add = max(0, 3 - x.dim())
    new_shape = (1,) * dims_to_add + x.shape
    return x.view(*new_shape)

def proj(x : Float[Tensor, "batch dmodel"], 
         Y : Float[Tensor, "batch numvec dmodel"],
         remove_zero_rows : bool = False,
         tol : float = 1e-8 
) -> Float[Tensor, "... dmodel"]:
    # Computes the projection of x onto the subspace spanned by the columns of Y
    Y = ensure_3d(Y)
        
    if remove_zero_rows:
        mask = torch.all(Y==0, dim=-1)
        Y = Y[mask]
        
    Y = Y.mT #(batch, dmodel, numvec) #require column vectors
    
    # Solve the linear system (Y^T @ Y) @ c = Y^T @ x
    # c is the coefficients of the projection of x onto the subspace spanned by the columns of Y
    # so the projection of x onto the subspace spanned by the columns of Y is Y @ c
    if x.ndim == 1:
        x = x.unsqueeze(0)
    
    c = torch.linalg.solve(Y.mT  @ Y, (x @ Y).mT)    
    proj_x = (Y @ c).mT 
    return proj_x.squeeze()


# %%

def entropy(probas):
    probas = probas[probas>0]
    return (-probas*torch.log2(probas)).sum(dim=-1)


# def rejection(x : Float[Tensor, "batch dmodel"], Y : Float[Tensor, "numvec dmodel"]) -> Float[Tensor, "batch dmodel"]:
#     return x - proj(x, Y)

@dataclass
class TokenizedSuffixesResult:
    input_ids: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    indices: Optional[torch.Tensor] = None

    def to(self, *args, **kwargs):
        """Move all tensor attributes to the specified device/dtype."""
        return TokenizedSuffixesResult(
            input_ids=self.input_ids.to(*args, **kwargs) if self.input_ids is not None else None,
            attention_mask=self.attention_mask.to(*args, **kwargs) if self.attention_mask is not None else None,
            indices=self.indices.to(*args, **kwargs) if self.indices is not None else None
        )
        

#TODO: test
def safe_tokenize(suffixes : List[str] | str, 
                  tokenizer : AutoTokenizer,
                  device = None
) -> TokenizedSuffixesResult:
    
    try:
        tokenizer = tokenizer.tokenizer
        warnings.warn("Passing models to safe_tokenize is deprecated, pass tokenizer instead", DeprecationWarning)
    except AttributeError:
        pass

    if isinstance(suffixes, str):    
        suffixes = [suffixes]
    
    if "Llama-2" in tokenizer.name_or_path:
        suffixes = ["🌍" + x for x in suffixes]
        space_token_id = tokenizer.convert_tokens_to_ids("▁")
        earth_token_id = tokenizer.convert_tokens_to_ids("🌍")
        
        suffix_tokens, attn_mask = tokenizer(suffixes,
                                                add_special_tokens=False,
                                                return_tensors="pt",
                                                padding=True).values()
        
        assert torch.all(suffix_tokens[:, 0] == space_token_id), "llama2 has leading space token"
        assert torch.all(suffix_tokens[:, 1] == earth_token_id), "llama2 single token for 🌍"
        
        suffix_tokens = suffix_tokens[:, 2:]
        attn_mask = attn_mask[:, 2:]
        idx = attn_mask.sum(dim=-1) - 1 #-1, and another two more: one for the space token, one for the 🌍 token
    
    else: # models that do not add leading spaces
        suffix_tokens, attn_mask = tokenizer(suffixes,
                                            add_special_tokens=False,
                                            return_tensors="pt",
                                            padding=True).values()
        idx = attn_mask.sum(dim=-1) - 1
        
    assert torch.all(idx >= 0), "Attention mask has zeros, empty suffixes"
    
    out = TokenizedSuffixesResult(
        input_ids=suffix_tokens,
        attention_mask=attn_mask,
        indices=idx
    )
    if device is not None:
        out = out.to(device)

    return out

# def safe_tokenize(suffixes : List[str] | str, 
#                   model : HookedTransformer
# ) -> TokenizedSuffixesResult:
#     device = next(model.parameters()).device
#     model.tokenizer.pad_token = model.tokenizer.eos_token
    
#     if isinstance(suffixes, str):    
#         suffixes = [suffixes]
    
#     if "Llama-2" in model.tokenizer.name_or_path:
#         suffixes = ["🌍" + x for x in suffixes]
#         space_token_id = model.tokenizer.convert_tokens_to_ids("▁")
#         earth_token_id = model.tokenizer.convert_tokens_to_ids("🌍")
        
#         suffix_tokens, attn_mask = model.tokenizer(suffixes,
#                                                 add_special_tokens=False,
#                                                 return_tensors="pt",
#                                                 padding=True).values()
        
#         assert torch.all(suffix_tokens[:, 0] == space_token_id), "llama2 has leading space token"
#         assert torch.all(suffix_tokens[:, 1] == earth_token_id), "llama2 single token for 🌍"
        
#         suffix_tokens = suffix_tokens[:, 2:]
#         attn_mask = attn_mask[:, 2:]
#         idx = attn_mask.sum(dim=-1) - 1 #-1, and another two more: one for the space token, one for the 🌍 token
    
#     else: # models that do not add leading spaces
#         suffix_tokens, attn_mask = model.tokenizer(suffixes,
#                                                 add_special_tokens=False,
#                                                 return_tensors="pt",
#                                                 padding=True).values()
#         idx = attn_mask.sum(dim=-1) - 1
        
#     assert torch.all(idx >= 0), "Attention mask has zeros, empty suffixes"
#     suffix_tokens = suffix_tokens.to(device)
#     attn_mask = attn_mask.to(device)
#     idx = idx.to(device)
    
#     return TokenizedSuffixesResult(
#         input_ids=suffix_tokens,
#         attention_mask=attn_mask,
#         indices=idx
#     )



#from gen_data import LANG_TO_NAME
# %%

# __DEBUG__ = True


# SuffixPreambleReturn = namedtuple('SuffixPreambleReturn', ['kv_cache', 'tokens', "indices"])
# from IPython.core.debugger import set_trace

# def process_suffix_toks(suffix_toks, model):
#     vocab = model.tokenizer.get_vocab()
#     try:
#         if "Llama-2" in model.cfg.model_name:
#             #assert torch.all( ((suffix_toks[:, 0] == vocab["▁"]) | (suffix_toks[:, 0] == vocab["▁<"]))), "LLama tokenizer should prepend space token"
#             suffix_toks = suffix_toks[:, 1:] # remove space token
            
#         elif "gemma" in model.cfg.model_name: # gemma tokenizer does not prepend space token automatically
#             pass  
        
#         else:
#             raise ValueError(f"Check {model.cfg.model_name} tokenization first, add case to prefix.py")
#         printd(suffix_toks)
#         return suffix_toks
#     except AssertionError as e:
#         print(e)
#         set_trace()  # Start the debugger

# def suffix_preamble(src_words, model, keep_idx, src_lang = None, dest_lang = None, **kwargs):
#     assert src_lang in LANG_TO_NAME and dest_lang in LANG_TO_NAME, "src_lang and dest_lang must be provided"
#     prompt = gen_prompt(None, src_lang, dest_lang)
#     kv_cache = gen_kv_cache(prompt, model)
#     suffixes = gen_common_suffixes(src_words, src_lang, dest_lang)
#     suffix_toks, keep_idx = safe_tokenize(suffixes, model)
    
#     return SuffixPreambleReturn(kv_cache, suffix_toks, keep_idx)


# def run(src_words, model, src_lang, dest_lang, batch_size = None, **kwargs):    
#     assert batch_size is not None, "prefix.run: batch_size must be provided"
#     kv_cache, suffix_toks, keep_idx = suffix_preamble(src_words, model, src_lang, dest_lang)
#     all_probs, all_toks = batched_predict_next(kv_cache, suffix_toks, model, 
#                                                       batch_size=batch_size, desc=f"{src_lang} -> {dest_lang}")
    
#     unk_id = model.tokenizer.convert_tokens_to_ids('<unk>')
#     mask_all_probs = -torch.ones_like(all_probs)
#     mask_all_toks = torch.zeros_like(all_toks) + unk_id
#     mask_suffix_toks = torch.zeros_like(suffix_toks[:, 0]) + unk_id
    
#     mask_all_probs[keep_idx] = all_probs[keep_idx]
#     mask_all_toks[keep_idx] = all_toks[keep_idx]
#     mask_suffix_toks[keep_idx] = suffix_toks[keep_idx][:, 0]
    
#     # Keep outputs the same size, but mask out the ones that are not in the vocab
#     # keep_idx = boolean_mask of which ones are in the vocab
#     return mask_all_probs, mask_all_toks, mask_suffix_toks, keep_idx

# @torch.no_grad()
# def measure_performance(df, model, **kwargs): 
#     src_lang = kwargs.get("src_lang", None)
#     dest_lang = kwargs.get("dest_lang", None)
#     batch_size = kwargs.get("batch_size", 1)
#     device = next(model.parameters()).device
#     assert src_lang is not None and dest_lang is not None, "src_lang and dest_lang must be provided"
#     correct = 0
#     processed = 0
#     total_loss = 0
#     tokenizer = model.tokenizer
    
#     kv_cache, suffix_toks, _ = suffix_preamble(df[src_lang], model, src_lang, dest_lang)
    
#     target_toks = torch.LongTensor(list(df[f'{dest_lang}_tok'])).to(device)
#     tensor_dataset = TensorDataset(suffix_toks, target_toks)
#     dataloader = DataLoader(tensor_dataset, batch_size=batch_size)
#     runner = tqdm(dataloader, total=len(dataloader), desc="Measuring performance", position=0, leave=True)
    
#     loss = torch.nn.CrossEntropyLoss()
    
#     for i, (suffix, target) in enumerate(runner):
#         logits = prefix.run_with_kv_cache(suffix, kv_cache, model).logits[:, -1].detach()
#         total_loss += loss(logits, target)
#         correct += (logits.argmax(dim=-1) == target).sum()
#         processed += len(target)
#         runner.set_description(f"Accuracy: {correct.item() / processed:.3f}, Loss: {total_loss.item() / (i+1):.3f}")
#     return correct / len(df), total_loss / len(dataloader)

# %%
