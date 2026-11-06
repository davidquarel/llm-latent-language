import sys, os, torch, warnings, traceback, collections
sys.path.insert(0, os.getcwd()); import fast_steer as fs, pandas as pd
from src.prompt import gen_prompt, gen_common_suffixes
from src.constants import WORD_LIST
cfg = fs.Cfg(model_name=sys.argv[1], devices="0,1,2"); m, tok = fs.load_model(cfg)
tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
M = fs.LastTokModel(m, tok); nL = M.n_layers; tb = fs.token_bytes_table(tok, M.V)
df = pd.read_csv(cfg.dataset_path); mask = df["en"].isin(WORD_LIST); pdf, sdf = df[mask], df[~mask].reset_index(drop=True); W = len(sdf)
wctx = fs.prefill(M, gen_prompt(pdf, "fr", "de"), gen_common_suffixes(sdf["ru"], "ru", "zh"))
u = fs.latent_dirs(M, list(sdf["en"]), "en", "first")
steer = fs.SteerSpec(u.roll(1, 0), u, torch.randn(nL, W))
sc = fs.Scorer(tb, {"to": [fs.candidates(x, "zh") for x in sdf["zh"]]})
configs = [(s, e) for s in range(8, 9) for e in range(s + 1, nL)]
fs.run_engine(M, wctx, configs, torch.arange(W), steer, sc)
hits = collections.Counter()
def hook(message, category, filename, lineno, file=None, line=None):
    st = [f for f in traceback.extract_stack() if "fast_steer" in f.filename]
    hits[(st[-1].lineno, st[-1].line) if st else ("?", str(message)[:80])] += 1
warnings.showwarning = hook
torch.cuda.set_sync_debug_mode("warn")
fs.run_engine(M, wctx, configs, torch.arange(W), steer, sc)
torch.cuda.set_sync_debug_mode(0)
for k, v in hits.most_common(20): print(v, k)
