# %%
# %load_ext autoreload
# %autoreload 2

# %%
import pandas as pd
from dataclasses import dataclass, field, asdict
import numpy as np
from matplotlib import pyplot as plt
import os, sys
from tqdm.auto import tqdm

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
from transformer_lens import HookedTransformer, HookedTransformerKeyValueCache

# %%
# ==== Custom Libraries ====
import gen_data
from utils import plot_ci_plus_heatmap
from tuned_lens_wrap import load_tuned_lens
from dq_utils import proj, entropy, plot_ci, is_chinese_char, broadcast_kv_cache, printd
from logit_lens import get_logits, plot_logit_lens_latents, latent_heatmap
from config_argparse import try_parse_args
from const import language_labels, translation_bank, lang2name
__DEBUG__ = True
# %%
torch.set_grad_enabled(False)
@dataclass
class Config:
    model_name : str = 'gemma-2b'
    #model_name : str = 'meta-llama/Llama-2-7b-hf'
    #save_dir : str = './data/synth_gemma_2b'
    #save_dir : str = './data/synth_llama_2_7b_new'
    save_dir : str = "DUMMY_NAME"
    #translation_bank_name : str = "llama" # just always use llama
    model_dtype : torch.dtype = torch.float16
    translation_threshold : float = 0.0
    batch_size : int = 128

cfg = Config()
cfg = try_parse_args(cfg)
# %%


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = HookedTransformer.from_pretrained_no_processing(cfg.model_name,
                                                            device=device,
                                                            dtype=cfg.model_dtype)
vocab = model.tokenizer.get_vocab()
# %%

# all_translation_bank = [
#     {'day': {'zh': '日', 'en': 'day', 'fr': 'jour', 'de': 'Tag', 'ru': 'день'},
#     'man': {'zh': '男', 'en': 'man', 'fr': 'homme', 'de': 'Mann', 'ru': 'муж'},
#     'five': {'zh': '五', 'en': 'five', 'fr': 'cinq', 'de': 'fünf', 'ru': 'три'},
#     'new': {'zh': '新', 'en': 'village', 'fr': 'nouveau', 'de': 'neu', 'ru': 'пя'}},
    
#     {'water': {'zh': '水', 'en': 'water', 'fr': 'eau', 'de': 'Wasser', 'ru': 'вода'},
#     'middle': {'zh': '中', 'en': 'middle', 'fr': 'milieu', 'de': 'Mitte', 'ru': 'середина'},
#     'three': {'zh': '三', 'en': 'three', 'fr': 'trois', 'de': 'drei', 'ru': 'три'},
#     'woman': {'zh': '女', 'en': 'woman', 'fr': 'femme', 'de': 'Frau', 'ru': 'женщина'}}
# ]
    


def check_bank_valid(bank):
    """
    Check if all tokens in the translation bank are in the vocabulary
    """    
    toks = sum([list(x.values()) for x in bank.values()],[])
    new_toks = []
    for x in toks:
        if is_chinese_char(x):
            new_toks.append(x)
        else:
            new_toks.append("▁" + x)
                
    ids = [vocab.get(x,None) for x in new_toks]
    if None in ids:
        print(f"Invalid tokens: {ids}, {new_toks}")
        return False
    return True

assert check_bank_valid(translation_bank), f"Translation bank contains invalid tokens for model {model.cfg.model_name}"

# %%

def remove_dup_translation(df):
    # Filter columns list based on columns present in the DataFrame
    valid_columns = [col for col in language_labels if col in df.columns]
    
    # Apply a function across the DataFrame rows
    #filtered_df = df[df.apply(lambda row: len(set(row[valid_columns])) == len(valid_columns), axis=1)]
    filtered_df = df[df.apply(lambda row: len(set([str(x).lower() for x in row[valid_columns]])) == len(valid_columns), axis=1)]

    
    removed_count = len(df) - len(filtered_df)
    print(f"Removed {removed_count} duplicates for {valid_columns}")
    
    return filtered_df

# %%
    
def printd(*args, **kwargs):
    # Check if '__DEBUG__' is in the global namespace and if it is set to True
    if globals().get('__DEBUG__', False):
        print("DEBUG:", end=" ")
        print(*args, **kwargs)
# %%
@torch.no_grad()
def translate(src_words, 
              src_lang, 
              dest_lang, 
              model = model, 
              translation_bank = translation_bank, 
              batch_size = None, 
              debug = False, 
              **kwargs):
    vocab = model.tokenizer.get_vocab()

    global __DEBUG__
    __DEBUG__ = debug

    if batch_size is None:
        batch_size = len(src_words)
    bs = batch_size
    
    threshold = 0

    def process_suffix_toks(suffix_toks):
        if "Llama-2" in model.cfg.model_name:
            assert torch.all(suffix_toks[:, 0] == vocab["▁"]), "LLama tokenizer should prepend space token"
            suffix_toks = suffix_toks[:, 1:]
            
        elif "gemma" in model.cfg.model_name:
            pass 
        
        else:
            raise ValueError(f"Check {model.cfg.model_name} tokenization first, add case to make_synth_dataset")
        if debug:
            printd(suffix_toks)
        return suffix_toks
    
    def run(src_words, src_lang, dest_lang):
        
        all_probs = []
        all_toks = []
        
        kv_cache = HookedTransformerKeyValueCache.init_cache(model.cfg, device, 1) # flush cache
        prefix = gen_data.generate_translation_prompt(None, src_lang, dest_lang, translations = translation_bank)
        prefix_tok = model.tokenizer.encode(prefix, return_tensors="pt").to(device)
        model(prefix_tok, past_kv_cache = kv_cache) #fill kv_cache
        kv_cache.freeze()
        
        suffixes = gen_data.generate_common_suffixes(src_words, src_lang, dest_lang) #suffixes will have leading space for gemma
        global raw_suffix_toks
        raw_suffix_toks, attention_mask = model.tokenizer(suffixes, add_special_tokens=False, return_tensors="pt", padding = True).values() #remove start of sequence character
        good_idx = torch.ones(len(raw_suffix_toks), dtype=torch.bool)
        if torch.any(attention_mask == 0):
            printd("Attention mask has zeros")
            # assume that most common number of tokens is correct
            # we only get more tokens if somethign screwed up and a chinese character was sampled
            # when it shouldn't have, so take the ids with shortest attention mask
            counts = attention_mask.sum(dim=1)
            correct_count = torch.mode(counts, dim=0).values
            good_idx = counts == correct_count
            # global bad_idx
            # bad_idx = torch.where(counts != correct_count)
            
            suffix_toks = raw_suffix_toks[good_idx][:, :correct_count]
            assert torch.all(attention_mask[good_idx][:, :correct_count] == 1), "Some tokens have zero attention mask"
        else:
            suffix_toks = raw_suffix_toks
        
        suffix_toks = suffix_toks.to(device)
        suffix_toks = process_suffix_toks(suffix_toks)
        suffix_toks_batched = torch.split(suffix_toks, bs, dim=0)
        
        runner = tqdm(suffix_toks_batched, total=len(suffix_toks), desc=f"{src_lang} -> {dest_lang}", position=0, leave=True)
        
        for batch in runner:
            broadcast_kv_cache(kv_cache, len(batch))
            batch = batch.to(device)
            logits = model(batch, past_kv_cache = kv_cache)[:, -1].detach()  # model returns (batch, seq, dvocab)
            probs = torch.softmax(logits, dim=-1)
            max_probs, max_tokens = torch.max(probs, dim=-1)
            
            all_probs.append(max_probs)
            all_toks.append(max_tokens)
            runner.update(len(batch))
            
        all_probs = torch.cat(all_probs, dim=0)
        all_toks = torch.cat(all_toks, dim=0)
        return all_probs, all_toks, suffix_toks, good_idx

    # if debug:
    #     for src_word in src_words:
    #         if is_chinese_char(src_word):
    #             assert src_word in vocab, f"Input zh string {src_word} not in vocabulary"
    #         else:
    #             assert "▁" + src_word in vocab, f"Input non-zh string {src_word} not in vocabulary"
    
    to_dest_probs, to_dest_tokens, src_suffix_toks, good_idx = run(src_words, src_lang, dest_lang)
    
    idx = (to_dest_probs > threshold) & good_idx.to(device)
    to_dest_probs = to_dest_probs[idx]
    to_dest_tokens = to_dest_tokens[idx]
    src_tokens = src_suffix_toks[idx,0]
    #print(f"{src_tokens.shape=}, {src_suffix_toks.shape=}")
    
    print(f"Kept {len(to_dest_probs)} / {len(idx)} translations")
    to_dest_words = model.tokenizer.convert_ids_to_tokens(to_dest_tokens)
    rev_src_probs, rev_src_tokens, dest_suffix_toks, good_idx2 = run(to_dest_words, dest_lang, src_lang)

    dest_tokens = dest_suffix_toks[:, 0]
    
    printd(model.tokenizer.convert_ids_to_tokens(src_tokens[:50]))
    printd(model.tokenizer.convert_ids_to_tokens(dest_tokens[:50]))
    printd(model.tokenizer.convert_ids_to_tokens(rev_src_tokens[:50]))

    to_dest_probs = to_dest_probs[good_idx2]
    to_dest_tokens = to_dest_tokens[good_idx2]
    src_tokens = src_tokens[good_idx2]

    cidx = (src_tokens == rev_src_tokens) & (rev_src_probs > threshold)
    
    print(f"{src_lang} = {dest_lang} Correct translations: {cidx.sum()} / {len(src_words)}")
    
    data = {
        src_lang: model.tokenizer.convert_ids_to_tokens(src_tokens[cidx]),
        dest_lang: model.tokenizer.convert_ids_to_tokens(dest_tokens[cidx]),
        src_lang + "_tok" : src_tokens[cidx].cpu(),
        dest_lang + "_tok" : dest_tokens[cidx].cpu(),
        f'{src_lang}_to_{dest_lang}_prob': to_dest_probs[cidx].cpu(),
        f'{dest_lang}_to_{src_lang}_prob': rev_src_probs[cidx].cpu()
    }

    df = pd.DataFrame(data)
    df = remove_dup_translation(df)
    return df

# df = translate(zh_tokens, "zh", "en", bs=128, debug=False, threshold =0)
# df
# %%
# %%
def en_tokens():
    en_words = []
    with open("./data/dict/en_dict.txt") as en_dict:
        for word in tqdm(en_dict):
            word = word.strip()
            if "▁" + word in vocab:
                en_words.append("▁" + word)
    print(f"Found {len(en_words)} in vocabulary from dictionary")
    return en_words
# %%


# %%
def auto_translate(translation_threshold = 0.5, **kwargs):

    en_words = en_tokens()
    

    bank = {}

    print(f"Translation Bank: {translation_bank}")


    en_to_zh = translate(en_words, 'en', 'zh', **kwargs)
    en_to_fr = translate(en_words, 'en', 'fr', **kwargs)
    en_to_de = translate(en_words, 'en', 'de', **kwargs)
    en_to_ru = translate(en_words, 'en', 'ru', **kwargs)

  
    #filtered_df.to_csv(os.path.join(save_dir, 'llama2_filtered_30_tol.csv'), index=False)
    
    bank["en_to_zh"] = en_to_zh
    bank["en_to_fr"] = en_to_fr
    bank["en_to_de"] = en_to_de
    bank["en_to_ru"] = en_to_ru
    
    
    
    all = pd.merge(en_to_zh, en_to_fr, on=['en','en_tok'], how='inner')
    all = pd.merge(all, en_to_de, on=['en','en_tok'], how='inner')
    all = pd.merge(all, en_to_ru, on=['en','en_tok'], how='inner')

    print(f"Merged size {len(all)}")
    print(all[:10])
    # Save filtered_df dataframe
    all = remove_dup_translation(all)
    mask = all.filter(like='_prob').gt(translation_threshold).all(axis=1)
    # Filter the DataFrame using the mask
    filtered_df = all[mask]
    filtered_df.reset_index(drop=True, inplace=True)
    
    
    
    bank["all"] = all
    bank["filtered"] = filtered_df

    return bank
# %%
# en_words = en_tokens()
# en_to_zh = translate(en_words, 'en', 'zh', batch_size=1)
#en_to_zh = translate(en_words[:200], 'en', 'zh', batch_size=64, debug=True)
# %%

def main(save_dir = None, **kwargs):
    assert save_dir is not None, "Please provide a save directory"
    os.makedirs(save_dir, exist_ok=True)
    
    bank = auto_translate(**kwargs)
    
    for key, df in tqdm(bank.items()):
        df.to_csv(os.path.join(save_dir, f'{key}.csv'), mode = "w", index=False)
# %%
cfg_dict = asdict(cfg)
main(**cfg_dict)
# %%