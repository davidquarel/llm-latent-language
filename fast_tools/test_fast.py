# Correctness test: fast engine vs. plain HF forward with hooks (no caching tricks).
import sys, os, torch, time, itertools
sys.path.insert(0, os.getcwd())
import fast_steer as fs
import pandas as pd
from src.prompt import gen_prompt, gen_common_suffixes
from src.constants import WORD_LIST
torch.manual_seed(0)
name = sys.argv[1]; dev = sys.argv[2] if len(sys.argv) > 2 else "0"
cfg = fs.Cfg(model_name=name, devices=dev, dtype=os.environ.get("DT", "bfloat16"))
m, tok = fs.load_model(cfg)
if tok.pad_token is None: tok.pad_token = tok.eos_token
M = fs.LastTokModel(m, tok); nL = M.n_layers
tb = fs.token_bytes_table(tok, M.V)
df = pd.read_csv(cfg.dataset_path); mask = df["en"].isin(WORD_LIST)
pdf, sdf = df[mask], df[~mask].reset_index(drop=True)
src, dest, src2, dest2 = "fr", "zh", "ru", "de"
W = 12
words = sdf.iloc[:W]
perm = torch.roll(torch.arange(W), 1)
wdf = words.iloc[perm.numpy()].reset_index(drop=True)
prompt = gen_prompt(pdf, src, dest)
rctx = fs.prefill(M, prompt, gen_common_suffixes(words[src], src, dest), keep_hf_logits=True)
wctx = fs.prefill(M, prompt, gen_common_suffixes(wdf[src2], src2, dest2), keep_hf_logits=True)
print("bad tok", rctx.n_bad_tok, wctx.n_bad_tok, "P", rctx.P)
u = fs.latent_dirs(M, list(words["en"]), "en", "first")
cand = {"a": [fs.candidates(x, dest) for x in words[dest]]}
sc = fs.Scorer(tb, cand)
rr = fs.run_engine(M, rctx, [(nL, nL + 1)], torch.arange(W), None, sc, max_new=8, record_resid=True)
hf_arg = rctx.hf_logits.argmax(-1)
print("clean first-token agree with HF:", (rr["first_tok"][0] == hf_arg).float().mean().item())
c_new = torch.einsum("lwd,wd->lw", rr["resid"], u)
MODE = os.environ.get("MODE", "swap"); ALPHA = float(os.environ.get("ALPHA", "1"))
steer = fs.SteerSpec(u_old=u[perm], u_new=u, c_new=c_new, mode=MODE, alpha=ALPHA, r_read=rr["resid"])
print("mode", MODE, "alpha", ALPHA)

# reference: full-sequence HF greedy decode with hooks at the query position
p_ids, s_ids, _ = fs.tokenize_with_prompt(tok, prompt, gen_common_suffixes(wdf[src2], src2, dest2))
def ref_decode(w, s, e, steps):
    seq = p_ids + s_ids[w]; qi = len(seq) - 1
    hooks = []
    for l in range(s, min(e, nL)):
        def hk(mod, inp, out, l=l):
            h = out[0] if isinstance(out, tuple) else out
            x = h[0, qi].float()[None]
            uo, un = steer.u_old[w].to(x.device)[None], steer.u_new[w].to(x.device)[None]
            cn = steer.c_new[l, w].to(x.device)[None]
            x = fs.apply_intervention(steer.mode, steer.alpha, x, uo, un, cn, steer.r_read[l, w].to(x.device)[None])
            h[0, qi] = x[0].to(h.dtype)
            return out
        hooks.append(M.layers[l].register_forward_hook(hk))
    toks = []
    try:
        for _ in range(steps):
            ids = torch.tensor([seq + toks], device=M.dev[0])
            lg = m(input_ids=ids).logits[0, -1]
            toks.append(int(lg.argmax()))
    finally:
        for h in hooks: h.remove()
    return toks
configs = [(0, 5), (3, 10), (8, 16), (12, nL - 1), (0, nL), (nL - 3, nL), (5, 6), (nL, nL + 1)]
cand2 = {"to": [fs.candidates(x, dest2) for x in words[dest2]], "from": [fs.candidates(x, dest2) for x in wdf[dest2]]}
sc2 = fs.Scorer(tb, cand2)
t0 = time.time()
res = fs.run_engine(M, wctx, configs, torch.arange(W), steer, sc2, max_new=6)
print("engine time", time.time() - t0, "steps", res["steps"])
agree = tot = 0; bad = []
for ci, (s, e) in enumerate(configs):
    for w in range(W):
        g = [t for t in res["gen_toks"][ci, w].tolist() if t >= 0]
        r = ref_decode(w, s, e, len(g))
        ok = g == r
        agree += ok; tot += 1
        if not ok:
            # logit gap at the first divergent step, under the HF reference
            d = next(i for i in range(min(len(g), len(r))) if g[i] != r[i])
            seq = p_ids + s_ids[w]; qi = len(seq) - 1
            hooks = []
            for l in range(s, min(e, nL)):
                def hk(mod, inp, out, l=l, w=w):
                    h = out[0] if isinstance(out, tuple) else out
                    x = h[0, qi].float()[None]; uo, un = steer.u_old[w].to(x.device)[None], steer.u_new[w].to(x.device)[None]
                    x = fs.apply_intervention(steer.mode, steer.alpha, x, uo, un, steer.c_new[l, w].to(x.device)[None], steer.r_read[l, w].to(x.device)[None])
                    h[0, qi] = x[0].to(h.dtype); return out
                hooks.append(M.layers[l].register_forward_hook(hk))
            lg = m(input_ids=torch.tensor([seq + r[:d]], device=M.dev[0])).logits[0, -1].float()
            for h in hooks: h.remove()
            gap = (lg[r[d]] - lg[g[d]]).item()
            bad.append(((s, e), w, tok.convert_ids_to_tokens(g), tok.convert_ids_to_tokens(r), f"HF logit gap ref-vs-fast at step {d}: {gap:.3f}"))
print(f"greedy token sequences identical to HF reference: {agree}/{tot}")
for b in bad[:10]: print("  MISMATCH", b)
print("example outputs:")
for ci in [0, 3, 7]:
    for w in range(4):
        g = [t for t in res["gen_toks"][ci, w].tolist() if t >= 0]
        print(" ", configs[ci], words["en"][w], "<-", wdf["en"][w], repr(tok.decode(g)), "done", res["done"][ci, w].item())
