# kernel-level profile of one engine chunk (start group 0..)
import sys, os, torch, time
sys.path.insert(0, os.getcwd()); import fast_steer as fs, pandas as pd
from src.prompt import gen_prompt, gen_common_suffixes
from src.constants import WORD_LIST
cfg = fs.Cfg(model_name=sys.argv[1], devices="0,1,2"); m, tok = fs.load_model(cfg)
tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
M = fs.LastTokModel(m, tok); nL = M.n_layers; tb = fs.token_bytes_table(tok, M.V)
df = pd.read_csv(cfg.dataset_path); mask = df["en"].isin(WORD_LIST); pdf, sdf = df[mask], df[~mask].reset_index(drop=True); W = len(sdf)
sh = torch.load("bench/orig_l2_7b.csv.shuffle.pt"); wdf = sdf.iloc[sh.numpy()].reset_index(drop=True)
t = time.time(); wctx = fs.prefill(M, gen_prompt(pdf, "fr", "de"), gen_common_suffixes(wdf["ru"], "ru", "zh")); fs._sync_all(); print("prefill", time.time() - t)
u = fs.latent_dirs(M, list(sdf["en"]), "en", "first")
steer = fs.SteerSpec(u[sh], u, torch.randn(nL, W))
cand = {"to": [fs.candidates(x, "zh") for x in sdf["zh"]], "from": [fs.candidates(x, "zh") for x in wdf["zh"]]}
sc = fs.Scorer(tb, cand)
configs = [(s, e) for s in range(8, 10) for e in range(s + 1, nL)]
fs.run_engine(M, wctx, configs, torch.arange(W), steer, sc)  # warmup
fs._sync_all(); t = time.time(); fs.run_engine(M, wctx, configs, torch.arange(W), steer, sc); fs._sync_all(); print("chunk wall", time.time() - t)
from torch.profiler import profile, ProfilerActivity
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    fs.run_engine(M, wctx, configs, torch.arange(W), steer, sc); fs._sync_all()
print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=22, max_name_column_width=60))
