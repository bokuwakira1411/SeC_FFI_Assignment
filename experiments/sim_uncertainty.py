"""E11 stage 1 (CPU): offline simulation of an uncertainty trigger on E0r's trajectories (model outputs only,
no GT in the trigger). The decision for frame f uses outputs of frames < f (causal: the concept for frame f
must be computed before frame f is segmented).
  S1 (logit): logit[f-1] < mean(logit[f-1-w..f-2]) - D
  S2 (area):  area[f-1] < (1 - r) * mean(area[f-1-w..f-2])   (mean > 0)
  S3 (empty): mask empty on frames f-N..f-1
  cooldown C: an uncertainty call is allowed only if the last concept call (HSV or uncertainty) is > C frames back.
Modes: unc (uncertainty only), hsv_or_unc (E0r HSV triggers + uncertainty calls).
Metrics (per object-frame, tracked frames only):
  rate   = % frames with a concept call
  recall = % of E0r incidents (analysis/E0r/incidents.csv) with a call in [onset-2, onset+5];
           recall_miss = same on incidents with no HSV trigger in that window
  spur   = % of calls with no incident of that object in its window [onset-2, onset+5] and no msl5 cut within +-2
Limit: trajectories are E0's (a simulated call does not change later masks): detection only, no J&F.

Usage (from SeC/): python experiments/sim_uncertainty.py  -> experiments/analysis/E11sim/{grid.csv,summary.md}
"""
import collections
import csv
import glob
import itertools
import json
import os

import numpy as np

OUT = "experiments/analysis/E11sim"
W = 5
objs = collections.defaultdict(list)
for f in sorted(glob.glob("experiments/E0r/frames/*.csv")):
    for r in csv.DictReader(open(f)):
        objs[(r["video"], r["obj"])].append(r)
cuts = json.load(open("experiments/scenecuts_msl5.json"))
inc = collections.defaultdict(list)
for r in csv.DictReader(open("experiments/analysis/E0r/incidents.csv")):
    inc[(r["video"], r["obj"])].append(int(r["frame"]))
n_inc = sum(map(len, inc.values()))

data = {}
for k, rows in objs.items():
    rows.sort(key=lambda r: int(r["frame"]))
    data[k] = dict(fr=np.array([int(r["frame"]) for r in rows]),
                   cond=np.array([r["is_cond"] == "1" for r in rows]),
                   hsv=np.array([r["hsv_trigger"] == "1" for r in rows]),
                   logit=np.array([float(r["object_score_logit"]) for r in rows]),
                   area=np.array([float(r["mask_px"]) for r in rows]))


def calls(d, D, r, N, C, mode):
    """frame -> reason for one object."""
    out, last = {}, int(d["fr"][0])
    for i in range(1, len(d["fr"])):
        f = int(d["fr"][i])
        if d["cond"][i]:
            continue
        if mode == "hsv_or_unc" and d["hsv"][i]:
            out[f], last = "hsv", f
            continue
        if f - last <= C:
            continue
        j = i - 1                                   # last frame with outputs
        lo = max(0, j - W)
        why = None
        if N and j - N + 1 >= 0 and (d["area"][j - N + 1:j + 1] == 0).all():
            why = "unc_empty"
        elif D is not None and j > lo and d["logit"][j] < d["logit"][lo:j].mean() - D:
            why = "unc_logit"
        elif r is not None and j > lo and d["area"][lo:j].mean() > 0 and \
                d["area"][j] < (1 - r) * d["area"][lo:j].mean():
            why = "unc_area"
        if why:
            out[f], last = why, f
    return out


def in_win(t, onset):
    return onset - 2 <= t <= onset + 5


def evaluate(trig):
    """trig: (video,obj) -> {frame: reason}"""
    n_frames = sum((~d["cond"]).sum() for d in data.values())
    n_calls = sum(len(t) for t in trig.values())
    hit = hit_m = n_m = spur = 0
    for k, onsets in inc.items():
        hsvf = data[k]["fr"][data[k]["hsv"]]
        for o in onsets:
            h = any(in_win(t, o) for t in trig.get(k, {}))
            miss = not any(in_win(t, o) for t in hsvf)
            hit += h; n_m += miss; hit_m += h and miss
    for k, t in trig.items():
        for f in t:
            spur += not any(in_win(f, o) for o in inc.get(k, [])) and not any(abs(c - f) <= 2 for c in cuts[k[0]])
    return dict(rate=100 * n_calls / n_frames, recall=100 * hit / n_inc, recall_miss=100 * hit_m / max(n_m, 1),
                n_miss=n_m, hit_miss=hit_m, spur=100 * spur / max(n_calls, 1), n_calls=n_calls)


def main():
    os.makedirs(OUT, exist_ok=True)
    hsv_trig = {k: {int(f): "hsv" for f in d["fr"][d["hsv"] & ~d["cond"]]} for k, d in data.items()}
    base = evaluate(hsv_trig)
    res = []
    for D, r, N, C in itertools.product((None, 2, 4, 6), (None, 0.5, 0.8), (0, 1, 3), (5, 10, 20)):
        if D is None and r is None and not N:
            continue
        for mode in ("unc", "hsv_or_unc"):
            e = evaluate({k: calls(d, D, r, N, C, mode) for k, d in data.items()})
            res.append(dict(D=D or "-", r=r or "-", N=N or "-", C=C, mode=mode, **{x: round(y, 2) if isinstance(y, float) else y for x, y in e.items()}))
    with open(f"{OUT}/grid.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    ok = [x for x in res if x["mode"] == "hsv_or_unc" and x["recall_miss"] >= 30 and x["rate"] <= 17
          and x["spur"] <= base["spur"]]
    L = [f"HSV (E0r): rate {base['rate']:.2f}% | recall {base['recall']:.1f}% | incidents missed by HSV "
         f"{base['n_miss']}/{n_inc} | spurious {base['spur']:.1f}% ({base['n_calls']} calls)",
         f"configs passing (a) recall_miss>=30 (b) rate<=17 (c) spur<={base['spur']:.1f}: {len(ok)}", ""]
    hdr = "| D | r | N | C | mode | rate % | recall % | recall HSV-missed % | spurious % |\n|---|---|---|---|---|---|---|---|---|"
    def row(x):
        return (f"| {x['D']} | {x['r']} | {x['N']} | {x['C']} | {x['mode']} | {x['rate']:.2f} | {x['recall']:.1f} | "
                f"{x['recall_miss']:.1f} ({x['hit_miss']}/{x['n_miss']}) | {x['spur']:.1f} |")
    L += ["Passing (hsv_or_unc), sorted by recall HSV-missed then rate:", hdr]
    L += [row(x) for x in sorted(ok, key=lambda x: (-x["recall_miss"], x["rate"]))[:15]]
    L += ["", "Best hsv_or_unc by recall HSV-missed at rate<=17 (any spurious):", hdr]
    L += [row(x) for x in sorted([x for x in res if x["mode"] == "hsv_or_unc" and x["rate"] <= 17],
                                 key=lambda x: (-x["recall_miss"], x["spur"]))[:10]]
    open(f"{OUT}/summary.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
