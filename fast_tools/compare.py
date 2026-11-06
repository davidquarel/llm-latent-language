import pandas as pd, sys
o = pd.read_csv(sys.argv[1]); f = pd.read_csv(sys.argv[2])
o = o[(o.src_lang == f.src_lang[0]) & (o.src2 == f.src2[0]) & (o.dest2 == f.dest2[0])]
po = o.pivot_table(index=["l_start", "l_end"], columns="latent_type", values=["acc", "prob"])
f = f.set_index(["l_start", "l_end"]).loc[po.index]
print("windows compared:", len(po))
for k in ["steer_to", "steer_from", "steer_to_en", "steer_from_en"]:
    da = (po[("acc", k)] - f[f"{k}_orig_acc"]).abs(); dp = (po[("prob", k)] - f[f"{k}_orig_prob"]).abs()
    print(f"{k:14s} acc orig {po[('acc',k)].mean():.4f} fast {f[f'{k}_orig_acc'].mean():.4f} max|d| {da.max():.4f} | prob max|d| {dp.max():.4f} | exact-metric (fast) {f[f'{k}_exact'].mean():.4f}")
print("clean acc orig", po[("acc", "clean")].mean(), " fast(read ctx) orig-metric", f.clean_orig.mean(), "exact", f.clean_exact.mean())
