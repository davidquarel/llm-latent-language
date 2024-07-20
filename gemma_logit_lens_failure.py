# %%
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from transformer_lens import HookedTransformer

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#device = "cpu"

def load_model(model_name='gemma-2b'):
    return HookedTransformer.from_pretrained_no_processing(model_name, device=device, dtype=torch.bfloat16)

def compute_logit_lens(model, prompt):
    with torch.no_grad():
        prompt_tok = model.tokenizer(prompt, return_tensors='pt')['input_ids'].to(device)
        all_post_resid = [f'blocks.{i}.hook_resid_post' for i in range(model.cfg.n_layers)]
        _, cache = model.run_with_cache(prompt_tok, names_filter=all_post_resid)
        latents = torch.stack([c[0, -1] for c in cache.values()], dim=0)
        approx_logits = model.unembed(model.ln_final(latents.unsqueeze(0))).squeeze()
        approx_logprobs = F.log_softmax(approx_logits, dim=-1)
        top_logprobs, top_idx = torch.topk(approx_logprobs, 10, dim=-1)
        return top_logprobs.cpu().numpy(), [model.tokenizer.convert_ids_to_tokens(x) for x in top_idx.cpu().numpy()]

def plot_logit_lens(model, top_logprobs, top_tokens, src_lang, src_word, dest_lang, dest_word):
    n_layers, top_k = model.cfg.n_layers, 10
    fig, ax = plt.subplots(figsize=(10, 12))
    
    cax = ax.imshow(top_logprobs, cmap='cividis', aspect='auto')
    fig.colorbar(cax).set_label('Log Probability')

    ax.set_ylabel('Layers')
    
    norm = Normalize(vmin=0, vmax=1)
    sm = ScalarMappable(norm=norm, cmap='cividis')
    
    for i in range(n_layers):
        for j in range(top_k):
            ax.text(j, i, top_tokens[i][j], ha='center', va='center', color='black', fontsize=10)

    plt.title(f'{model.cfg.model_name} {src_lang}: {src_word} -> {dest_lang}: {dest_word}?')
    plt.tight_layout()
    plt.show()

def test_logit_lens(model, prompt, src_lang, src_word, dest_lang, dest_word):
    top_logprobs, top_tokens = compute_logit_lens(model, prompt)
    
    print(f"Prompt: {prompt}")
    print(f"Top Tokens: {top_tokens[-1]}")
    print(f"Top Probs: {np.exp(top_logprobs[-1])}")
    
    plot_logit_lens(model, top_logprobs, top_tokens, src_lang, src_word, dest_lang, dest_word)

# %%
if __name__ == "__main__":
    model = load_model(model_name = "meta-llama/Llama-2-7b-hf")
    src_word, src_lang = "chanson", "fr"
    dest_lang, dest_word = "de", "Lied"
    
    prompt = f'''Français: " jour" Deutsch: " Tag"
Français: " homme" Deutsch: " Mann"
Français: " cinq" Deutsch: " fünf"
Français: " nouveau" Deutsch: " neu"
Français: " {src_word}" Deutsch: "'''


    test_logit_lens(model, prompt, src_lang, src_word, dest_lang, dest_word)# %%

# %%
