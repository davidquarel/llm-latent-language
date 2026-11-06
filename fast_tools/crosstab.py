# Row-level: what did the model actually output when the ORIGINAL metric credits "steer_to"?
import sys, os, torch, collections
sys.path.insert(0, os.getcwd()); import fast_steer as fs, pandas as pd
from src.prompt import gen_prompt, gen_common_suffixes
from src.constants import WORD_LIST
name = sys.argv[1]; win = eval(sys.argv[2]); perms = [tuple(p.split(",")) for p in sys.argv[3].split(";")]
cfg = fs.Cfg(model_name=name, devices="0,1,2", layer_split="1.4,1,0.75")
m, tok = fs.load_model(cfg); tok.pad_token = tok.pad_token or tok.eos_token; tok.padding_side = "right"
M = fs.LastTokModel(m, tok); nL = M.n_layers; tb = fs.token_bytes_table(tok, M.V)
df = pd.read_csv(cfg.dataset_path); mask = df["en"].isin(WORD_LIST); pdf, sdf = df[mask], df[~mask].reset_index(drop=True); W = len(sdf)
sh = torch.load("bench/orig_l2_7b.csv.shuffle.pt"); wdf = sdf.iloc[sh.numpy()].reset_index(drop=True)
class _A: token_add_capitalization = token_add_prefixes = token_add_spaces = token_utf8_byte = True
u = fs.latent_dirs(M, list(sdf["en"]), "en", "first")
tab = collections.Counter(); ex = collections.defaultdict(list)
for (src, dest, src2, dest2) in perms:
    prompt = gen_prompt(pdf, src, dest)
    rctx = fs.prefill(M, prompt, gen_common_suffixes(sdf[src], src, dest))
    wctx = fs.prefill(M, prompt, gen_common_suffixes(wdf[src2], src2, dest2))
    rr = fs.run_engine(M, rctx, [(nL, nL + 1)], torch.arange(W), None, fs.Scorer(tb, {"c": [fs.candidates(x, dest) for x in sdf[dest]]}), record_resid=True)
    steer = fs.SteerSpec(u[sh], u, torch.einsum("lwd,wd->lw", rr["resid"], u))
    cand = {"to": [fs.candidates(x, dest2) for x in sdf[dest2]], "from": [fs.candidates(x, dest2) for x in wdf[dest2]],
            "to_en": [fs.candidates(x, "en") for x in sdf["en"]], "from_en": [fs.candidates(x, "en") for x in wdf["en"]]}
    oa = {"to": fs.orig_answer_sets(tok, list(sdf[dest2]), _A)}
    res = fs.run_engine(M, wctx, [win], torch.arange(W), steer, fs.Scorer(tb, cand), orig_ans=oa)
    for w in range(W):
        d = res["done"][0, w].item()
        cls = "to" if d & 1 else "from" if d & 2 else "to_en" if d & 4 else "from_en" if d & 8 else "other"
        o = int(res["orig_acc_to"][0, w])
        tab[(dest2, o, cls)] += 1
        if o == 1 and cls != "to" and len(ex[(dest2, cls)]) < 4:
            g = [t for t in res["gen_toks"][0, w].tolist() if t >= 0]
            ex[(dest2, cls)].append(f"target {sdf[dest2][w]!r} got {tok.decode(g)!r} (1st tok {tok.convert_ids_to_tokens(g[:1])[0]!r})")
print("dest2 | orig_metric_says_to | exact outcome : count")
for k in sorted(tab): print(f"  {k[0]} | {k[1]} | {k[2]:8s}: {tab[k]}")
for k, v in ex.items():
    print("examples credited by OLD metric but exact says", k)
    for e in v: print("    ", e)
