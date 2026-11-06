# Decode and show what the model actually outputs under steering, next to the original first-token metric.
import sys, os, torch, itertools
sys.path.insert(0, os.getcwd())
import fast_steer as fs, pandas as pd
from src.prompt import gen_prompt, gen_common_suffixes
from src.constants import WORD_LIST
name = sys.argv[1]; src, dest, src2, dest2 = sys.argv[2].split(","); wins = eval(sys.argv[3])
cfg = fs.Cfg(model_name=name, devices="0,1,2")
m, tok = fs.load_model(cfg); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
M = fs.LastTokModel(m, tok); nL = M.n_layers; tb = fs.token_bytes_table(tok, M.V)
df = pd.read_csv(cfg.dataset_path); mask = df["en"].isin(WORD_LIST)
pdf, sdf = df[mask], df[~mask].reset_index(drop=True); W = len(sdf)
sh = torch.load("bench/orig_l2_7b.csv.shuffle.pt"); wdf = sdf.iloc[sh.numpy()].reset_index(drop=True)
prompt = gen_prompt(pdf, src, dest)
rctx = fs.prefill(M, prompt, gen_common_suffixes(sdf[src], src, dest))
wctx = fs.prefill(M, prompt if os.environ.get("WP", "orig") == "orig" else gen_prompt(pdf, src2, dest2), gen_common_suffixes(wdf[src2], src2, dest2))
u = fs.latent_dirs(M, list(sdf["en"]), "en", "first")
rr = fs.run_engine(M, rctx, [(nL, nL + 1)], torch.arange(W), None, fs.Scorer(tb, {"c": [fs.candidates(x, dest) for x in sdf[dest]]}), record_resid=True)
steer = fs.SteerSpec(u[sh], u, torch.einsum("lwd,wd->lw", rr["resid"], u))
cand = {"to": [fs.candidates(x, dest2) for x in sdf[dest2]], "from": [fs.candidates(x, dest2) for x in wdf[dest2]],
        "to_en": [fs.candidates(x, "en") for x in sdf["en"]], "from_en": [fs.candidates(x, "en") for x in wdf["en"]]}
A = fs.Cfg()
class _A: token_add_capitalization = token_add_prefixes = token_add_spaces = token_utf8_byte = True
oa = {"to": fs.orig_answer_sets(tok, list(sdf[dest2]), _A), "from": fs.orig_answer_sets(tok, list(wdf[dest2]), _A)}
res = fs.run_engine(M, wctx, wins + [(nL, nL + 1)], torch.arange(W), steer, fs.Scorer(tb, cand), orig_ans=oa)
for ci, wnd in enumerate(wins + [(nL, nL + 1)]):
    print(f"=== window {wnd}  exact: " + " ".join(f"{k}={((res['done'][ci] >> i) & 1).float().mean():.3f}" for i, k in enumerate(cand)),
          " orig-acc: " + " ".join(f"{k}={res['orig_acc_'+k][ci].mean():.3f}" for k in oa))
    for w in range(0, W, 9):
        g = [t for t in res["gen_toks"][ci, w].tolist() if t >= 0]
        ft = tok.convert_ids_to_tokens(g[:1])[0]
        print(f"   inject {sdf['en'][w]:>10s} ({sdf[dest2][w]}) into query {wdf[src2][w]:>10s} ({wdf[dest2][w]}): out={tok.decode(g)!r:22s} first_tok={ft!r:10s} orig_to={int(res['orig_acc_to'][ci,w])} orig_from={int(res['orig_acc_from'][ci,w])} exact={res['done'][ci,w].item()}")
