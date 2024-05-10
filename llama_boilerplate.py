# %%
from transformers import LlamaForCausalLM, AutoTokenizer
import torch
from transformers import AutoModelForCausalLM
model_name = 'meta-llama/Llama-2-7b-hf'
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
from transformer_lens import HookedTransformer
# model = LlamaForCausalLM.from_pretrained(model_name, 
#                                          torch_dtype = "auto",
#                                          device_map = "auto")
LLAMA_2_7B_CHAT_PATH = "meta-llama/Llama-2-7b-hf"
inference_dtype = torch.float16
# hf_model = AutoModelForCausalLM.from_pretrained(LLAMA_2_7B_CHAT_PATH,
#                                              torch_dtype=inference_dtype,
#                                              device_map = "cuda:0")

tokenizer = AutoTokenizer.from_pretrained(LLAMA_2_7B_CHAT_PATH)
print('ye')
model = HookedTransformer.from_pretrained(LLAMA_2_7B_CHAT_PATH,
                                             dtype=inference_dtype,
                                             device = device,
                                             fold_ln=False,
                                             fold_value_biases=False,
                                             center_writing_weights=False,
                                             center_unembed=False,
                                             tokenizer=tokenizer)

# %%
