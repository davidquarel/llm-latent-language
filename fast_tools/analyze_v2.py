# Cross-run comparison of steering interventions (out_v2).
import sys, os, glob, json
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
root, outdir = sys.argv[1], sys.argv[2]; os.makedirs(outdir, exist_ok=True)
rows, curves = [], {}
for f in sorted(glob.glob(f"{root}/*/steer_*.csv")):
    if f.endswith("_timing.csv"): continue
    name = os.path.basename(f)[:-4]
    d = pd.read_csv(f)
    g = d.groupby(["l_start", "l_end"]).mean(numeric_only=True)
    bk = g["steer_to_known"].idxmax(); be = g["steer_to_exact"].idxmax()
    r = dict(run=name, perms=d.groupby(["src_lang", "dest_lang", "src2", "dest2"]).ngroups, words=None,
             known=g.known_frac.mean(), best=f"[{bk[0]},{bk[1]})",
             to_known=g.loc[bk, "steer_to_known"], to=g.loc[bk, "steer_to_exact"], from_=g.loc[bk, "steer_from_exact"],
             to_en=g.loc[bk, "steer_to_en_exact"], to_rdest=g.loc[bk, "steer_to_rdest_exact"],
             to_lat=g.loc[bk, "steer_to_lat_exact"] if "steer_to_lat_exact" in g else np.nan,
             old_metric_to=g.loc[bk, "steer_to_orig_acc"], sec=pd.read_csv(f.replace(".csv", "_timing.csv")).total.sum())
    by = d[(d.l_start == bk[0]) & (d.l_end == bk[1])].groupby("dest2").steer_to_known.mean().round(3).to_dict()
    r["by_dest2"] = by
    rows.append(r)
    curves[name] = g["steer_to_known"].groupby(level="l_end").max()
T = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print(T.drop(columns=["words"]).round(3).to_string(index=False))
T.to_csv(f"{outdir}/comparison.csv", index=False)
# per-end-layer curves: one figure per group
groups = {"interventions": [n for n in curves if "ctl" not in n and "v2" not in n],
          "controls": [n for n in curves if "ctl" in n], "dataset_v2": [n for n in curves if "v2" in n]}
for gname, names in groups.items():
    if not names: continue
    fig, ax = plt.subplots(figsize=(8, 5))
    for n in names:
        c = curves[n]; ax.plot(c.index, c.values, label=n.replace("steer_", ""), lw=1.6)
    ax.set_xlabel("end layer (exclusive); best start for each end"); ax.set_ylabel("P(new word in target lang | model knows it)")
    ax.set_title(f"Llama-2-7b steering: {gname}"); ax.legend(fontsize=7); ax.grid(alpha=.3)
    fig.tight_layout(); fig.savefig(f"{outdir}/curve_{gname}.png", dpi=120); plt.close(fig)
