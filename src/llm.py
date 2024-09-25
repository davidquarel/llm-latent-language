import matplotlib.pyplot as plt
import torch
from typing import List
from jaxtyping import Float, Int
from torch import Tensor
import torch
from utils.misc import printd
from typing import List, Tuple, Callable
from jaxtyping import Int
from tqdm.auto import tqdm
from transformer_lens import HookedTransformerKeyValueCache, HookedTransformer
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from collections import namedtuple

from .constants import LANG2NAME
from .prompt import gen_prompt, gen_common_suffixes
from .kv_cache import gen_kv_cache, run_with_kv_cache, batched_predict_next

def proj(x : Float[Tensor, "... dmodel"], Y : Float[Tensor, "numvec dmodel"]) -> Float[Tensor, "... dmodel"]:
    # Computes the projection of x onto the subspace spanned by the columns of Y
    if Y.dim() == 1:
        Y = Y.unsqueeze(0)
    Y = Y.mT #(dmodel, numvec) #require column vectors
    # Solve the linear system (Y^T @ Y) @ c = Y^T @ x
    # c is the coefficients of the projection of x onto the subspace spanned by the columns of Y
    # so the projection of x onto the subspace spanned by the columns of Y is Y @ c
    if x.ndim == 1:
        x = x.unsqueeze(0)
    
    c = torch.linalg.solve(Y.mT  @ Y, (x @ Y).mT)    
    proj_x = (Y @ c).mT 
    return proj_x.squeeze()

def entropy(probas):
    probas = probas[probas>0]
    return (-probas*torch.log2(probas)).sum(dim=-1)


def rejection(x : Float[Tensor, "batch dmodel"], Y : Float[Tensor, "numvec dmodel"]) -> Float[Tensor, "batch dmodel"]:
    return x - proj(x, Y)
    
# %%

#from gen_data import LANG2NAME
# %%

__DEBUG__ = True


SuffixPreambleReturn = namedtuple('SuffixPreambleReturn', ['kv_cache', 'tokens', "indices"])
from IPython.core.debugger import set_trace

def process_suffix_toks(suffix_toks, model):
    vocab = model.tokenizer.get_vocab()
    try:
        if "Llama-2" in model.cfg.model_name:
            #assert torch.all( ((suffix_toks[:, 0] == vocab["▁"]) | (suffix_toks[:, 0] == vocab["▁<"]))), "LLama tokenizer should prepend space token"
            suffix_toks = suffix_toks[:, 1:] # remove space token
            
        elif "gemma" in model.cfg.model_name: # gemma tokenizer does not prepend space token automatically
            pass  
        
        else:
            raise ValueError(f"Check {model.cfg.model_name} tokenization first, add case to prefix.py")
        printd(suffix_toks)
        return suffix_toks
    except AssertionError as e:
        print(e)
        set_trace()  # Start the debugger
        
TokenizeSuffixesResult = namedtuple('TokenizeSuffixes', ['tokens', 'indices'])

def raw_tokenize(suffixes : List[str], model):
    device = next(model.parameters()).device
    raw_suffix_toks, attention_mask = model.tokenizer(suffixes, 
                                                    add_special_tokens=False, 
                                                    return_tensors="pt", 
                                                    padding = True).values() #remove start of sequence character
    attention_mask = attention_mask.to(device)
    good_idx = torch.ones(len(raw_suffix_toks), dtype=torch.bool, device=device)
    if torch.any(attention_mask == 0):
        print("Warning: raw_tokenize - Attention mask has zeros")
        # assume that most common number of tokens is correct
        # we only get more tokens if something screwed up and a chinese character was sampled
        # when it shouldn't have, so take the ids with shortest attention mask
        counts = attention_mask.sum(dim=1)
        correct_count = torch.mode(counts, dim=0).values
        good_idx = (counts == correct_count).to(device)
        # global bad_idx
        # bad_idx = torch.where(counts != correct_count)
        
        suffix_toks = raw_suffix_toks[:, :correct_count]
        assert torch.all(attention_mask[good_idx][:, :correct_count] == 1), "Incorrect attention mask trimming"
        # assert not torch.all(attention_mask[good_idx][:, :correct_count+1] == 1), "Incorrect attention mask trimming"
    else:
        suffix_toks = raw_suffix_toks
    
    suffix_toks = suffix_toks.to(device)
    suffix_toks = process_suffix_toks(suffix_toks, model)
    assert suffix_toks.shape[0] == len(suffixes), "Suffixes and tokens should have the same length"
    return TokenizeSuffixesResult(suffix_toks, good_idx)

def suffix_preamble(src_words, model, keep_idx, src_lang = None, dest_lang = None, **kwargs):
    assert src_lang in LANG2NAME and dest_lang in LANG2NAME, "src_lang and dest_lang must be provided"
    prompt = gen_prompt(None, src_lang, dest_lang)
    kv_cache = gen_kv_cache(prompt, model)
    suffixes = gen_common_suffixes(src_words, src_lang, dest_lang)
    suffix_toks, keep_idx = raw_tokenize(suffixes, model)
    
    return SuffixPreambleReturn(kv_cache, suffix_toks, keep_idx)


def run(src_words, model, src_lang, dest_lang, batch_size = None, **kwargs):    
    assert batch_size is not None, "prefix.run: batch_size must be provided"
    kv_cache, suffix_toks, keep_idx = suffix_preamble(src_words, model, src_lang, dest_lang)
    all_probs, all_toks = batched_predict_next(kv_cache, suffix_toks, model, 
                                                      batch_size=batch_size, desc=f"{src_lang} -> {dest_lang}")
    
    unk_id = model.tokenizer.convert_tokens_to_ids('<unk>')
    mask_all_probs = -torch.ones_like(all_probs)
    mask_all_toks = torch.zeros_like(all_toks) + unk_id
    mask_suffix_toks = torch.zeros_like(suffix_toks[:, 0]) + unk_id
    
    mask_all_probs[keep_idx] = all_probs[keep_idx]
    mask_all_toks[keep_idx] = all_toks[keep_idx]
    mask_suffix_toks[keep_idx] = suffix_toks[keep_idx][:, 0]
    
    # Keep outputs the same size, but mask out the ones that are not in the vocab
    # keep_idx = boolean_mask of which ones are in the vocab
    return mask_all_probs, mask_all_toks, mask_suffix_toks, keep_idx


@torch.no_grad()
def measure_performance(df, model, **kwargs): 
    src_lang = kwargs.get("src_lang", None)
    dest_lang = kwargs.get("dest_lang", None)
    batch_size = kwargs.get("batch_size", 1)
    device = next(model.parameters()).device
    assert src_lang is not None and dest_lang is not None, "src_lang and dest_lang must be provided"
    correct = 0
    processed = 0
    total_loss = 0
    tokenizer = model.tokenizer
    
    kv_cache, suffix_toks, _ = suffix_preamble(df[src_lang], model, src_lang, dest_lang)
    
    target_toks = torch.LongTensor(list(df[f'{dest_lang}_tok'])).to(device)
    tensor_dataset = TensorDataset(suffix_toks, target_toks)
    dataloader = DataLoader(tensor_dataset, batch_size=batch_size)
    runner = tqdm(dataloader, total=len(dataloader), desc="Measuring performance", position=0, leave=True)
    
    loss = torch.nn.CrossEntropyLoss()
    
    for i, (suffix, target) in enumerate(runner):
        logits = prefix.run_with_kv_cache(suffix, kv_cache, model).logits[:, -1].detach()
        total_loss += loss(logits, target)
        correct += (logits.argmax(dim=-1) == target).sum()
        processed += len(target)
        runner.set_description(f"Accuracy: {correct.item() / processed:.3f}, Loss: {total_loss.item() / (i+1):.3f}")
    return correct / len(df), total_loss / len(dataloader)

# %%
