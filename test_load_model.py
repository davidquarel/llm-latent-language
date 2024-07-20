# %%
from transformer_lens import HookedTransformer
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model_name =  "meta-llama/Llama-2-7b-hf"
model = HookedTransformer.from_pretrained_no_processing(model_name, device=device)
# %%
