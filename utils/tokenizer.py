# %%
from dataclasses import dataclass
from transformers import AutoTokenizer
import os 
from typing import Literal
from transformer_lens import HookedTransformer
from dataclasses import dataclass
from transformers import AutoTokenizer

@dataclass
class MinimalConfig:
   tokenizer_prepends_bos: bool = None 
   d_vocab: int = -1
   d_vocab_out: int = -1

class MinimalModel:
   def __init__(self):
       self.cfg = MinimalConfig()
       self.tokenizer = None

class AutoTLTokenizer:
   @staticmethod
   def from_pretrained(model_name, **kwargs):
       # Get base HF tokenizer
        default_padding_side: Literal["left", "right"] = "right"
        use_fast = True
        if "phi" in model_name.lower():
            use_fast = False

        huggingface_token = os.environ.get("HF_TOKEN", None)
        hf_tokenizer = AutoTokenizer.from_pretrained(
                        model_name,
                        add_bos_token=True,
                        trust_remote_code=True,
                        use_fast=use_fast,
                        token=huggingface_token,
                    )

        model = MinimalModel()
        HookedTransformer.set_tokenizer(model, hf_tokenizer, default_padding_side=default_padding_side)
        
        if model.tokenizer.unk_token_id is None:
            model.tokenizer.unk_token_id = model.tokenizer.all_special_ids[-1]
        if model.tokenizer.pad_token_id is None:
            model.tokenizer.pad_token_id = model.tokenizer.unk_token_id

        return model.tokenizer
# %%
AutoTLTokenizer.from_pretrained("meta-llama/Llama-3.2-3B")
# %%
