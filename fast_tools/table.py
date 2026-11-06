import json, sys
s = json.load(open(sys.argv[1]))
for k, v in s.items():
    b = tuple(int(x) for x in v["best_window_exact"])
    lat = f" to_lat={v['steer_to_lat_exact@best_exact']:.3f}" if "steer_to_lat_exact@best_exact" in v else ""
    print(f"{k:42s} n={v['n_perms']:2d} clean exact/orig={v['clean_exact']:.3f}/{v['clean_orig']:.3f} | best{b}: to={v['steer_to_exact@best_exact']:.3f} "
          f"to_en={v['steer_to_en_exact@best_exact']:.3f} from={v['steer_from_exact@best_exact']:.3f}{lat} | orig-metric best {tuple(int(x) for x in v['best_window_orig'])}: "
          f"to={v['steer_to_orig_acc_at_best_orig']:.3f} | {v['sec_per_perm']:.1f}s/perm")
    print("      by dest2 (exact, orig):", {d: (x["steer_to_exact"], x["steer_to_orig_acc"]) for d, x in v["by_dest2@best_exact"].items()})
