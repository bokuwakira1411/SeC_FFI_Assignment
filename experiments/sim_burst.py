"""E10 parameter choice (CPU): simulate the concept-call rate of --burst_after_trigger K /
--concept_every_sparse P from E0r's HSV triggers (same rule as propagate_in_video; HSV triggers do not
depend on the settings). Rate = concept calls / tracked (non-cond) object-frames, as "% frame gọi concept".

Usage (from SeC/): python experiments/sim_burst.py [EXP=E0r]
"""
import csv
import glob
import itertools
import sys

exp = sys.argv[1] if len(sys.argv) > 1 else "E0r"
objs = {}
for f in sorted(glob.glob(f"experiments/{exp}/frames/*.csv")):
    for r in csv.DictReader(open(f)):
        objs.setdefault((r["video"], r["obj"]), []).append(r)


def rate(K, P):
    calls = n = 0
    for rows in objs.values():
        burst, last = 0, int(rows[0]["frame"])   # first row = start (cond) frame
        for r in rows:
            if r["is_cond"] == "1":
                continue
            n += 1
            fr, hsv = int(r["frame"]), r["hsv_trigger"] == "1"
            call = hsv
            if hsv:
                burst = K
            elif burst > 0:
                burst -= 1; call = True
            elif P > 0 and fr - last >= P:
                call = True
            if call:
                last = fr; calls += 1
    return 100 * calls / n


print("| K | P | % frame gọi concept |\n|---|---|---|")
res = {}
for K, P in itertools.product((3, 5, 10), (10, 15, 20, 0)):
    res[K, P] = rate(K, P)
    print(f"| {K} | {P} | {res[K, P]:.2f} |")
print(f"check K=0,P=0: {rate(0, 0):.2f} (E0r trigger %)")
for name, tgt in (("E10a", 25.65), ("E10b", 17.14)):
    k = min(res, key=lambda k: abs(res[k] - tgt))
    print(f"{name}: target {tgt} -> K={k[0]}, P={k[1]} ({res[k]:.2f}%)")
