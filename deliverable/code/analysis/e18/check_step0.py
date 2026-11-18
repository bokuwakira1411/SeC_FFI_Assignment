"""E18 step 0 check of --e17_commit output on the 3 smoke videos (CPU).

Usage (from SeC/): python3 experiments/e18/check_step0.py
- tracking state per frame vs E10a (experiments/fps160/E10a/frames): object_score_logit, pred_iou, triggered,
  sam_mem_frames (pixel memory frames), bank_frames, ctx_frames must be equal on every frame;
- masks of frames not accepted must be bit-identical to E10a; accepted frames must differ from E10a only there.
"""
import filecmp
import os

import pandas as pd

SMOKE = [l.strip() for l in open("experiments/e17/subset_smoke.txt") if l.strip()]
COLS = ["object_score_logit", "pred_iou", "triggered", "sam_mem_frames", "bank_frames", "ctx_frames"]
run, ref = "experiments/e18/chk_output", "experiments/fps160/E10a"
n_rows = n_bad = n_acc = n_mask_diff = n_acc_diff = 0
for v in SMOKE:
    a = pd.read_csv(f"{run}/frames/{v}.csv", dtype={"obj": str})
    b = pd.read_csv(f"{ref}/frames/{v}.csv", dtype={"obj": str})
    m = a.merge(b, on=["obj", "frame"], suffixes=("", "_ref"), validate="1:1")
    assert len(m) == len(a) == len(b), v
    for c in COLS:
        x, y = m[c].fillna(""), m[f"{c}_ref"].fillna("")
        bad = (x != y)
        if bad.any():
            print(f"{v} {c}: {int(bad.sum())} frames differ, e.g. frame {m[bad].frame.iloc[0]}")
        n_bad += int(bad.sum())
    n_rows += len(m)
    acc = set(zip(a.obj[a.e17_accept == 1], a.frame[a.e17_accept == 1]))
    n_acc += len(acc)
    for o in sorted(os.listdir(f"{ref}/masks/{v}")):
        for f in sorted(os.listdir(f"{ref}/masks/{v}/{o}")):
            same = filecmp.cmp(f"{run}/masks/{v}/{o}/{f}", f"{ref}/masks/{v}/{o}/{f}", shallow=False)
            if (o, int(f[:-4])) in acc:
                n_acc_diff += not same
            else:
                n_mask_diff += not same
print(f"rows {n_rows}; state mismatches {n_bad}; accepted frames {n_acc} (masks differing from E10a: {n_acc_diff}); "
      f"non-accepted frames with a mask differing from E10a: {n_mask_diff}")
