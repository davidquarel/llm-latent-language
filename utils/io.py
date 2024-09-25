from ast import literal_eval
from tabulate import tabulate
import warnings
from src.prompt import find_all_tokens
import re
import torch

from src.datatypes import LangIdx

def parse_word_list(s):
    # Remove the outer brackets and split by commas
    try:
        result = literal_eval(s)
        return result
    except:
        warnings.warn(f"Could not parse row: {s}")
        s = s.strip()[1:-1]
        items = re.split(r',\s*', s)
        
        result = []
        for item in items:
            # Remove surrounding quotes if present
            if (item.startswith("'") and item.endswith("'")) or (item.startswith('"') and item.endswith('"')):
                item = item[1:-1]
            # Handle apostrophes within words
            item = item.replace("'", "'")
            result.append(item)
    
        return result
    

def build_lang_idx(df, lang, vocab, **kwargs):
    array = []
    word_list_key = kwargs.get('word_list_key', 'claude')
    for primary, word_list in df[[lang, f'{word_list_key}_{lang}']].values:
        row_list = parse_word_list(word_list)
        row_list.append(primary)
        tokens = [find_all_tokens(x, vocab, **kwargs) for x in row_list]
        try:
            idx = torch.unique(torch.cat(tokens))
        except:
            print(f"Could not concatenate tokens for row: {row_list}")
            print(f'{i=}')
            print(f'{row=}')
            print(f'{row_list=}')
            print(f'{tokens=}')
            
        array.append(idx)
    return array

def create_lang_idx(df, vocab, **kwargs):
    
    src_idx = build_lang_idx(df, kwargs['src_lang'], vocab, **kwargs)
    dest_idx = build_lang_idx(df, kwargs['dest_lang'], vocab,**kwargs)
    latent_idx = build_lang_idx(df, kwargs['latent_lang'], vocab,**kwargs)
    
    assert len(src_idx) == len(dest_idx), "Mismatch between src_idx and dest_idx lengths"
    assert len(src_idx) == len(latent_idx), "Mismatch between src_idx and latent_idx lengths"
    
    return LangIdx(src_idx, latent_idx, dest_idx)

