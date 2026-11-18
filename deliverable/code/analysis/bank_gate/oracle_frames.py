"""Oracle bank frames (E13a/E13b, bank_reason added_gt) vs the normal run's mask at that frame (CPU).

Usage (from SeC/): python experiments/bank_gate/oracle_frames.py
Pairs: E13a -> E0r, E13b -> E10a. Group of each added_gt frame, from the normal run's predicted mask:
empty (0 px), wrong (J < 0.5), right (J >= 0.5). Per object: J&F gain oracle - normal
(experiments/results/per_object_jf.csv) vs number / share of empty-or-wrong frames (Spearman), group means.
Same on the objects of the 10 worst E0 videos (experiments/results/per_video_jf.csv).
Writes experiments/bank_gate/oracle_frames.csv (per frame) and oracle_objects.csv (per object).
"""
import csv, glob
import numpy as np
import pandas as pd
from PIL import Image
from scipy.stats import spearmanr

PAIRS = {"E13a": ("experiments/bank/E13a", "E0r", "experiments/E0r"),
         "E13b": ("experiments/bank/E13b", "E10a", "experiments/E10a")}
GT = "experiments/gt_subset40"
OUT = "experiments/bank_gate"

def mask(p):
    return np.array(Image.open(p)) > 0

rows = []
for orc, (odir, base, bdir) in PAIRS.items():
    for p in sorted(glob.glob(f"{odir}/frames/*.csv")):
        for r in csv.DictReader(open(p)):
            if r["bank_reason"] != "added_gt":
                continue
            v, o, f = r["video"], r["obj"], int(r["frame"])
            pm, g = mask(f"{bdir}/masks/{v}/{o}/{f:05d}.png"), mask(f"{GT}/{v}/{o}/{f:05d}.png")
            J = (pm & g).sum() / (pm | g).sum()
            grp = "empty" if pm.sum() == 0 else "wrong" if J < 0.5 else "right"
            rows.append(dict(oracle=orc, base=base, video=v, obj=o, frame=f, J=round(float(J), 4), group=grp))
fr = pd.DataFrame(rows)
fr.to_csv(f"{OUT}/oracle_frames.csv", index=False)

jf = pd.read_csv(f"{OUT}/../results/per_object_jf.csv", dtype={"obj": str})
pv = pd.read_csv(f"{OUT}/../results/per_video_jf.csv")
worst = set(pv.sort_values("E0").video.head(10))
objs = []
for orc, (_, base, _) in PAIRS.items():
    f = fr[fr.oracle == orc]
    for _, r in jf.iterrows():
        g = f[(f.video == r.video) & (f.obj == r.obj)]
        nb = int(g.group.isin(["empty", "wrong"]).sum())
        objs.append(dict(oracle=orc, video=r.video, obj=r.obj, n_added=len(g), n_bad=nb,
                         share_bad=nb / len(g) if len(g) else np.nan, gain=r[orc] - r[base],
                         worst10=r.video in worst))
ob = pd.DataFrame(objs)
ob.to_csv(f"{OUT}/oracle_objects.csv", index=False)

def bucket(n):
    return "0" if n == 0 else "1-2" if n <= 2 else "3-5" if n <= 5 else ">=6"

for orc in PAIRS:
    print(f"=== {orc} vs {PAIRS[orc][1]} ===")
    f = fr[fr.oracle == orc]
    print("frames:", len(f), f.group.value_counts().to_dict())
    fw = f[f.video.isin(worst)]
    print("  worst10 frames:", len(fw), fw.group.value_counts().to_dict())
    for name, s in [("all", ob[ob.oracle == orc]), ("worst10", ob[(ob.oracle == orc) & ob.worst10])]:
        rho, p = spearmanr(s.n_bad, s.gain)
        rho2, p2 = spearmanr(s.share_bad, s.gain, nan_policy="omit")
        print(f"  [{name}] objects={len(s)} mean gain={s.gain.mean():.2f}  "
              f"Spearman(n_bad,gain)={rho:.2f} (p={p:.3f})  Spearman(share_bad,gain)={rho2:.2f} (p={p2:.3f})")
        t = s.assign(b=s.n_bad.map(bucket)).groupby("b").agg(objects=("gain", "size"), mean_gain=("gain", "mean"),
                                                               median_gain=("gain", "median"), n_added=("n_added", "mean"))
        print(t.reindex([x for x in ["0", "1-2", "3-5", ">=6"] if x in t.index]).round(2).to_string())
