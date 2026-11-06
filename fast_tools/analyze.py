# Summarise fast steering sweeps: best windows, exact vs original metric, per-destination breakdown, heatmaps.
import sys, os, glob, json
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
root = sys.argv[1]; outdir = sys.argv[2]; os.makedirs(outdir, exist_ok=True)
summary = {}
files = sorted(f for f in glob.glob(f"{root}/*/steer_*.csv") if not f.endswith("_timing.csv"))
for f in files:
    name = os.path.basename(f)[:-4]; mdl = f.split("/")[-2]
    d = pd.read_csv(f)
    g = d.groupby(["l_start", "l_end"]).mean(numeric_only=True)
    best = g["steer_to_exact"].idxmax()
    best_orig = g["steer_to_orig_acc"].idxmax()
    row = dict(model=mdl, n_perms=d.groupby(["src_lang", "dest_lang", "src2", "dest2"]).ngroups,
               clean_exact=g.clean_exact.mean(), clean_orig=g.clean_orig.mean(),
               best_window_exact=list(best), **{f"{k}@best_exact": g.loc[best, k] for k in g.columns if k.endswith("_exact") or k.endswith("_orig_acc")},
               best_window_orig=list(best_orig), steer_to_orig_acc_at_best_orig=g.loc[best_orig, "steer_to_orig_acc"],
               steer_to_exact_at_best_orig=g.loc[best_orig, "steer_to_exact"])
    # per destination at best exact window
    dd = d[(d.l_start == best[0]) & (d.l_end == best[1])].groupby("dest2")[["steer_to_exact", "steer_to_orig_acc", "steer_from_exact", "steer_from_orig_acc", "steer_to_en_exact"]].mean()
    row["by_dest2@best_exact"] = dd.round(3).to_dict(orient="index")
    t = pd.read_csv(f.replace(".csv", "_timing.csv"))
    row["sec_per_perm"] = t.total.mean(); row["sec_total"] = t.total.sum()
    summary[f"{mdl}/{name}"] = row
    for metric in ["steer_to_exact", "steer_to_en_exact", "steer_from_exact"] + (["steer_to_lat_exact"] if "steer_to_lat_exact" in g else []):
        H = g[metric].unstack()  # rows start, cols end
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(H.values, origin="lower", cmap="viridis", vmin=0, vmax=max(0.05, float(np.nanmax(H.values))),
                       extent=[H.columns.min() - .5, H.columns.max() + .5, H.index.min() - .5, H.index.max() + .5], aspect="auto")
        ax.set_xlabel("end layer (exclusive)"); ax.set_ylabel("start layer"); ax.set_title(f"{mdl} {name}\n{metric} (mean over {row['n_perms']} lang perms)", fontsize=9)
        fig.colorbar(im); fig.tight_layout(); fig.savefig(f"{outdir}/{mdl}__{name}__{metric}.png", dpi=110); plt.close(fig)
json.dump(summary, open(f"{outdir}/summary.json", "w"), indent=1, default=float)
for k, v in summary.items():
    print("=" * 100); print(k)
    for kk, vv in v.items():
        print(f"  {kk}: {round(vv, 3) if isinstance(vv, float) else vv}")
