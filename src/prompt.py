# %%
from .constants import LANG_TO_NAME, LANG_BANK, LANGS_NO_SPACE
import torch
from typing import List
from collections import namedtuple
from .llm import safe_tokenize
import pandas as pd
from utils.tokenizer import AutoTLTokenizer
from dataclasses import dataclass
# def gen_prompt(src_words = None, 
#                dest_words = None, 
#                src_lang = None, 
#                dest_lang = None, 
#                num_examples= None):
#     """
#     Generate a prompt for translation tasks.

#     Args:
#         src_words (list): List of source language words/phrases.
#         dest_words (list): List of corresponding destination language words/phrases.
#         src_lang (str): Source language code (e.g., 'fr' for French).
#         dest_lang (str): Destination language code (e.g., 'zh' for Chinese).
#         num_examples (int): Number of examples to include in the prompt (default: 1).

#     Returns:
#         str: The generated prompt string.
#     """
#     assert src_lang is not None, "Source language must be provided"
#     assert dest_lang is not None, "Destination language must be provided"
    
#     if src_words is None:
#         src_words = LANG_BANK[src_lang]
#     if dest_words is None:
#         dest_words = LANG_BANK[dest_lang]

#     src_space = "" if src_lang == "zh" else " "
#     dest_space = "" if dest_lang == "zh" else " "

#     if num_examples is None:
#         num_examples = len(dest_words)

#     assert len(src_words) in [len(dest_words), len(dest_words)+1] , "Need N or N+1 source words for N dest words"

#     prompt = ""
#     for i in range(min(num_examples, len(src_words))):
#         prompt += f'{LANG_TO_NAME[src_lang]}: "{src_space}{src_words[i]}" - {LANG_TO_NAME[dest_lang]}: "{dest_space}{dest_words[i]}"\n'

#     # Add the last example without the destination translation
#     prompt += f'{LANG_TO_NAME[src_lang]}: "'

#     return prompt

def gen_prompt(df, src_lang, dest_lang):

    # Chinese does not have spaces between words
    # and chinese tokens do not have a leading space
    src_space = "" if src_lang in LANGS_NO_SPACE else " "
    dest_space = "" if dest_lang in LANGS_NO_SPACE else " "

    src_words = df[src_lang].to_list()
    dest_words = df[dest_lang].to_list()

    prompt = ""
    for (src, dest) in list(zip(src_words, dest_words)):
        prompt += f'{LANG_TO_NAME[src_lang]}: "{src_space}{src}" - {LANG_TO_NAME[dest_lang]}: "{dest_space}{dest}"\n'
    # Add the last example without the destination translation
    # dont need a space, as if present, will be included in the next word
    src = src_words[-1]
    prompt += f'{LANG_TO_NAME[src_lang]}: "'

    return prompt


# %%

def trim_space_token(str, space_token):
    str = str.strip()
    if str[0] == space_token:
        str = str[1:]
    return str

def gen_common_suffixes(src_words, 
                        src_lang = None, 
                        dest_lang = None):
    assert src_lang is not None, "Source language must be provided"
    assert dest_lang is not None, "Destination language must be provided"
    common_suffixes = []
    src_space = "" if src_lang in LANGS_NO_SPACE else " "
    
    for src_word in src_words:
        #src_word = trim_space_token(src_word)
        suffix = f'{src_space}{src_word}" - {LANG_TO_NAME[dest_lang]}: "'
        common_suffixes.append(suffix)
    return common_suffixes

    

    
# def gen_answer_ids(df, dest_lang, tokenizer):
#     #answer_list = []
#     answer_list_ids = []
#     for (word, synonyms) in df[[dest_lang, f"claude_{dest_lang}"]].values:
#         answers = [word] + parse_word_list(synonyms)
#         answers = answers + [f" {x}" for x in answers]
#         #answer_list.append(answers)
#         prefixes = set()
#         for answer in answers:
#             prefixes.update(set(token_prefixes(answer)))


#         answer_id = safe_tokenize(list(prefixes), tokenizer).input_ids[:, 0]
#         answer_id = torch.unique(answer_id)
#         answer_list_ids.append(answer_id)

#     all_answer_ids = torch.nn.utils.rnn.pad_sequence(answer_list_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
#     return all_answer_ids

def get_answer_tensor(dest_words, vocab, space_token, cfg, padding_value=-1):
    answer_list_ids = []
    for word in dest_words:
        answer_id = get_valid_answer(word, vocab, space_token, cfg, return_tensors='pt')
        answer_list_ids.append(answer_id)

    all_answer_ids = torch.nn.utils.rnn.pad_sequence(answer_list_ids, batch_first=True, padding_value=padding_value)
    return all_answer_ids

def get_answer_tensor2(dest_words, tokenizer, vocab, cfg, padding_value=-1):
    answer_list_ids = []
    for word in dest_words:
        answer_id = get_valid_answer2(word, tokenizer, vocab, cfg)
        answer_list_ids.append(answer_id)

    all_answer_ids = torch.nn.utils.rnn.pad_sequence(answer_list_ids, batch_first=True, padding_value=padding_value)
    return all_answer_ids


def get_prefixes(token_str: str):
        return [token_str[:i] for i in range(1, len(token_str)+1)]

def add_spaces(tokens, space_token):
    return [space_token + t for t in tokens]        

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
        
def get_valid_answer(token_str: str, vocab, space_token, cfg, **kwargs):
    """
    Finds all valid tokens in a given token string based on the provided vocabulary.

    Args:
        token_str (str): The token string to search for tokens in.
        vocab (list): The vocabulary list containing valid tokens.
        space_token (str): The space token used in the tokenization.
        cfg: Additional keyword arguments for customization.

    Keyword Args:
        token_add_prefixes (bool): Whether to add prefixes of the token string as tokens (default: True).
        token_add_spaces (bool): Whether to add tokens with spaces at the beginning (default: True).
        token_utf8_byte (bool): Whether to add the leading byte of non-ASCII tokens as tokens (default: True).
        return_tensors (str): The type of tensors to return ('str' or 'pt', default: 'str').

    Returns:
        list or torch.Tensor: The list of valid tokens or a tensor of token indices.

    """
    
    token_str = trim_space_token(token_str, space_token)
    
    
    token_strs = set([token_str])
    
    if cfg.token_add_capitalization:
        token_strs = token_strs | set([token_str.lower(), token_str.capitalize(), token_str.upper()])

    if cfg.token_add_prefixes:
        new_token_strs = set()
        for tok in token_strs:
            new_token_strs = new_token_strs | set(get_prefixes(tok))
        token_strs = new_token_strs
    
    if cfg.token_add_spaces:
        token_strs = token_strs | set(add_spaces(token_strs, space_token))
    
    if cfg.token_utf8_byte:
        tokid = unicode_leading_byte(token_str)
        if tokid is not None and tokid in vocab:
            token_strs.add(tokid)
    
    final_tokens = set([tok for tok in set(token_strs) if tok in vocab])

    if kwargs.get('debug', False):
        print(final_tokens)
    # just add leading byte for all languages unless it's in ascii range
    
    return_tensors = kwargs.get('return_tensors', 'str')
    if return_tensors == "str":
        return final_tokens
    else:
        return torch.LongTensor([vocab[x] for x in final_tokens])
# %%
def get_valid_answer2(token_str: str, tokenizer, vocab, cfg, **kwargs):
    """
    Finds all valid tokens in a given token string based on the provided vocabulary.

    Args:
        token_str (str): The token string to search for tokens in.
        vocab (list): The vocabulary list containing valid tokens. (passed seperately
        cause tokenizer.vocab is SLOW!)
        space_token (str): The space token used in the tokenization.
        cfg: Additional keyword arguments for customization.

    Keyword Args:
        token_add_prefixes (bool): Whether to add prefixes of the token string as tokens (default: True).
        token_add_spaces (bool): Whether to add tokens with spaces at the beginning (default: True).
        token_utf8_byte (bool): Whether to add the leading byte of non-ASCII tokens as tokens (default: True).
        return_tensors (str): The type of tensors to return ('str' or 'pt', default: 'str').

    Returns:
        list or torch.Tensor: The list of valid tokens or a tensor of token indices.

    """

    token_strs = set([token_str])

    if cfg.token_add_capitalization:
        token_strs = token_strs | set([token_str.lower(), token_str.capitalize(), token_str.upper()])

    if cfg.token_add_prefixes:
        new_token_strs = set()
        for tok in token_strs:
            new_token_strs = new_token_strs | set(get_prefixes(tok))
        token_strs = new_token_strs
    
    if cfg.token_add_spaces:
        token_strs = token_strs | set([f" {x}" for x in token_strs])
    
    tokens = safe_tokenize(list(token_strs), tokenizer).input_ids[:, 0].squeeze()

    if cfg.token_utf8_byte:
        utf8_token = unicode_leading_byte(token_str)
        if utf8_token is not None and utf8_token in vocab:
            utf8_id = vocab[utf8_token]
            tokens = torch.cat([tokens, torch.tensor([utf8_id])],dim=0)
    
    tokens = torch.unique(tokens)
    space_token_id = safe_tokenize(" ", tokenizer).input_ids.item()
    tokens = tokens[tokens != space_token_id].view(-1)


    if kwargs.get('debug', False):
        print(tokens, tokenizer.convert_ids_to_tokens(tokens))
    # just add leading byte for all languages unless it's in ascii range

    return tokens
# %%
def test_get_valid_answer():
    @dataclass
    class Config:
        debug: bool = True
        token_add_spaces: bool = True
        token_add_prefixes : bool = False
        token_add_capitalization : bool = True
        token_utf8_byte : bool = True

    cfg = Config()

    l3_tok = AutoTLTokenizer.from_pretrained("meta-llama/Llama-3.1-8B")
    l2_tok = AutoTLTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    l3_space_token = l3_tok.convert_ids_to_tokens(safe_tokenize(" ", l3_tok).input_ids.item())
    l2_space_token = l2_tok.convert_ids_to_tokens(safe_tokenize(" ", l2_tok).input_ids.item())

    for test_str in ["banana", "学习", "hello", "책"]:
        for (tok, space) in [(l2_tok, l2_space_token), (l3_tok, l3_space_token)]:
            print(f"Testing {test_str} with {tok.name_or_path}")
            old = get_valid_answer(test_str, tok.vocab, space, cfg, return_tensors='pt', debug=False).sort().values
            new = get_valid_answer2(test_str, tok, tok.vocab, cfg, debug=False).sort().values
            print(f"Old: {old}")
            print(f"New: {new}")
            print("=================")
# %%