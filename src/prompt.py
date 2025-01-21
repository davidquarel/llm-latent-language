# %%
from .constants import LANG_TO_NAME, LANG_BANK, LANGS_NO_SPACE
import torch
from typing import List
from collections import namedtuple

def gen_prompt(src_words = None, 
               dest_words = None, 
               src_lang = None, 
               dest_lang = None, 
               num_examples= None):
    """
    Generate a prompt for translation tasks.

    Args:
        src_words (list): List of source language words/phrases.
        dest_words (list): List of corresponding destination language words/phrases.
        src_lang (str): Source language code (e.g., 'fr' for French).
        dest_lang (str): Destination language code (e.g., 'zh' for Chinese).
        num_examples (int): Number of examples to include in the prompt (default: 1).

    Returns:
        str: The generated prompt string.
    """
    assert src_lang is not None, "Source language must be provided"
    assert dest_lang is not None, "Destination language must be provided"
    
    if src_words is None:
        src_words = LANG_BANK[src_lang]
    if dest_words is None:
        dest_words = LANG_BANK[dest_lang]

    src_space = "" if src_lang == "zh" else " "
    dest_space = "" if dest_lang == "zh" else " "

    if num_examples is None:
        num_examples = len(dest_words)

    assert len(src_words) in [len(dest_words), len(dest_words)+1] , "Need N or N+1 source words for N dest words"

    prompt = ""
    for i in range(min(num_examples, len(src_words))):
        prompt += f'{LANG_TO_NAME[src_lang]}: "{src_space}{src_words[i]}" - {LANG_TO_NAME[dest_lang]}: "{dest_space}{dest_words[i]}"\n'

    # Add the last example without the destination translation
    prompt += f'{LANG_TO_NAME[src_lang]}: "'

    return prompt
# %%

def gen_common_suffixes(src_words, 
                        src_lang = None, 
                        dest_lang = None):
    assert src_lang is not None, "Source language must be provided"
    assert dest_lang is not None, "Destination language must be provided"
    common_suffixes = []
    src_space = "" if src_lang in LANGS_NO_SPACE else " "
    
    for src_word in src_words:
        src_word = src_word.split('▁')[-1] # Remove leading space token if present
        suffix = f'{src_space}{src_word}" {LANG_TO_NAME[dest_lang]}: "'
        common_suffixes.append(suffix)
    return common_suffixes

    
def token_prefixes(token_str: str):
        return [token_str[:i] for i in range(1, len(token_str)+1)]

def add_spaces(tokens):
    return ['▁' + t for t in tokens]        


def unicode_leading_byte(token_str : str):
        """
        Returns the leading byte of a given token string if it is outside the ASCII range.

        Args:
            token_str (str): The token string to check.

        Returns:
            str or None: The leading byte of the token string if it is outside the ASCII range, None otherwise.
        """
        leading_byte = token_str.encode("utf-8")[0]
        if leading_byte >= 128: #outside ASCII range
            leading_byte = f'<0x{(token_str.encode("utf-8")[0]):X}>' # "好" -> "<0xE5>" 
            return leading_byte
        else:
            return None
    
def find_all_tokens(token_str: str, vocab, **kwargs):
    """
    Finds all valid tokens in a given token string based on the provided vocabulary.

    Args:
        token_str (str): The token string to search for tokens in.
        vocab (list): The vocabulary list containing valid tokens.
        **kwargs: Additional keyword arguments for customization.

    Keyword Args:
        token_add_prefixes (bool): Whether to add prefixes of the token string as tokens (default: True).
        token_add_spaces (bool): Whether to add tokens with spaces at the beginning (default: True).
        token_add_leading_byte (bool): Whether to add the leading byte of non-ASCII tokens as tokens (default: True).
        return_tensors (str): The type of tensors to return ('str' or 'pt', default: 'str').

    Returns:
        list or torch.Tensor: The list of valid tokens or a tensor of token indices.

    """
    
    token_str = token_str.strip()
    
    if token_str[0] == '▁' or token_str[0] == 'Ġ':
        token_str = token_str[1:]
    
    token_add_prefixes = kwargs.get('token_add_prefixes', True)
    token_add_spaces = kwargs.get('token_add_spaces', True)
    token_add_capitalization = kwargs.get('token_add_capitalization', True)
    token_add_leading_byte = kwargs.get('token_add_leading_byte', True)
    return_tensors = kwargs.get('return_tensors', 'pt')
    debug = kwargs.get('debug', False)
    
    token_strs = set([token_str])
    
    if token_add_capitalization:
        token_strs = token_strs | set([token_str.lower(), token_str.capitalize(), token_str.upper()])

    if token_add_prefixes:
        new_token_strs = set()
        for tok in token_strs:
            new_token_strs = new_token_strs | set(token_prefixes(tok))
        token_strs = new_token_strs
    
    if token_add_spaces:
        token_strs = token_strs | set(add_spaces(token_strs))
    
    if token_add_leading_byte:
        tokid = unicode_leading_byte(token_str)
        if tokid is not None and tokid in vocab:
            token_strs.add(tokid)
    
    final_tokens = set([tok for tok in set(token_strs) if tok in vocab])
    if debug:
        print(final_tokens)
    # just add leading byte for all languages unless it's in ascii range
    
    if return_tensors == "str":
        return final_tokens
    else:
        return torch.LongTensor([vocab[x] for x in final_tokens])

# if "Llama-2" in cfg.model_name:
#     test_suffixes2 = ["🌍" + x for x in suffixes]
#     raw_suffix_toks = safe_tokenize(test_suffixes2, model)
#     space_token_id = model.tokenizer.convert_tokens_to_ids("▁")
#     earth_id = model.tokenizer.convert_tokens_to_ids("🌍")
#     #print(raw_suffix_toks)     
#     assert torch.all(raw_suffix_toks.input_ids[:, 0] == space_token_id), "llama2 has leading space token"
#     assert torch.all(raw_suffix_toks.input_ids[:, 1] == earth_id), "llama2 single token for 🌍"
#         # they add leading spaces :'(
    
#     # suffix_toks.attention_mask = suffix_toks.attention_mask[:,1:]
#     suffix_toks = TokenizedSuffixesResult(input_ids=new_suffix_toks, 
#                                             attention_mask=new_attention_mask, 
#                                             indices=new_idx)
# else:
#     suffix_toks = safe_tokenize(suffixes, model)





# def generate_translation_prompt(word, src_lang=None, dest_lang=None, translations = translation_bank, **kwargs):
#     word = word.split('▁')[-1] if word is not None else None
    
#     src_space = " " if src_lang != 'zh' else ""
#     dest_space = " " if dest_lang != 'zh' else ""
#     not_dest_space = " " if dest_lang == 'zh' else ""

#     prompt = ""
#     for key, translation in translations.items():
#         prompt += f'{lang2name[src_lang]}: "{src_space}{translation[src_lang]}" {lang2name[dest_lang]}: "{dest_space}{translation[dest_lang]}"\n'
    
#     if word is None: #only generate common prefix
#         prompt += f'{lang2name[src_lang]}: "{src_space}'
#     else:
#         prompt += f'{lang2name[src_lang]}: "{src_space}{word}" {lang2name[dest_lang]}: "{not_dest_space}'
    
#     # Ensure prompt ends with a space for Chinese
#     # actually, no, we don't want this. It messes up the tokenization
#     # non-zh languages include the space. zh doesn't need the space.
#     # if dest_lang == 'zh':
#     #     prompt += ' '
    
#     return prompt
# %%
