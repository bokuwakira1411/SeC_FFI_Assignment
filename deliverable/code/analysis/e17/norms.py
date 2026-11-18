"""Norms of a (memory branch), b (concept branch), (a+b)/2 on concept-call frames (CPU).

Usage (from SeC/): python experiments/e17/norms.py [RUN_DIR]   (default experiments/e17/norms)
"""
import glob, sys
import pandas as pd

d = pd.concat(pd.read_csv(p) for p in glob.glob(f"{sys.argv[1] if len(sys.argv) > 1 else 'experiments/e17/norms'}/frames/*.csv"))
d = d[d.norm_a.notna()].copy()
d["b_over_a"] = d.norm_b / d.norm_a
d["mix_over_a"] = d.norm_mix / d.norm_a
d["lost"] = d.get("e17_lost", 0).fillna(0).astype(int)
cols = ["norm_a", "norm_b", "norm_mix", "b_over_a", "mix_over_a", "cos_ab"]
print("all concept calls, n =", len(d)); print(d[cols].describe(percentiles=[.1, .5, .9]).round(3).to_string())
for k, g in d.groupby("lost"):
    print(f"\nlost={k}, n={len(g)}"); print(g[cols].median().round(3).to_string())
if "e17_accept" in d:
    a = d[d.lost == 1]
    print("\nlost calls:", len(a), "accepted:", int(a.e17_accept.fillna(0).sum()))
    print(a[["video", "obj", "frame", "gt_empty", "J_mix", "J_con", "mix_iou", "con_iou", "mix_logit", "con_logit",
             "e17_accept", "con_maxcos"]].to_string(index=False))
