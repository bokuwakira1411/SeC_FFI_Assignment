"""E11a analysis (CPU), after `analyze_frames.py E11a`. Incident definition v1 (analyze_frames.py).
- Recall on E0's 102 incidents (analysis/E0/incidents.csv): real E11a concept calls (frames/*.csv) in
  [onset-2, onset+5], vs the simulation (sim_uncertainty.py, same config); split HSV-missed.
- Recovery on E0's incidents: max J over onset+1..onset+10 >= E0 pre_mean - 0.1, with the J curve of E0 and of
  E11a (same frames); all incidents and those caught by an uncertainty call in E11a.
- Concept cosine: cosine between consecutive concepts of an object (E11a/concepts/), by reason of the later call.
Usage (from SeC/): python experiments/analyze_e11.py  -> experiments/analysis/E11a/e11.md
"""
import collections
import csv
import glob
import os
import sys

import numpy as np
import torch

sys.path.insert(0, "experiments")
import sim_uncertainty as sim  # noqa: E402

EXP = "E11a"
OUT = f"experiments/analysis/{EXP}"


def load_frames(exp):
    rows = collections.defaultdict(dict)
    for f in glob.glob(f"experiments/{exp}/frames/*.csv"):
        for r in csv.DictReader(open(f)):
            rows[(r["video"], r["obj"])][int(r["frame"])] = r
    return rows


def load_j(exp):
    J = collections.defaultdict(dict)
    for r in csv.DictReader(open(f"experiments/analysis/{exp}/j_per_frame.csv")):
        J[(r["video"], r["obj"])][int(r["frame"])] = float(r["J"])
    return J


def win(t, o):
    return o - 2 <= t <= o + 5


def main():
    L = []
    rows = load_frames(EXP)
    J0, J1 = load_j("E0"), load_j(EXP)
    inc = list(csv.DictReader(open("experiments/analysis/E0/incidents.csv")))
    calls = {k: {f: r.get("concept_reason") or "hsv" for f, r in rr.items() if r["triggered"] == "1"}
             for k, rr in rows.items()}
    nc = sum(r["is_cond"] == "0" for rr in rows.values() for r in rr.values())
    rate = 100 * sum(map(len, calls.values())) / nc
    reasons = collections.Counter(x for c in calls.values() for x in c.values())
    simtrig = {k: sim.calls(d, 2, 0.5, 1, 5, "hsv_or_unc") for k, d in sim.data.items()}
    se = sim.evaluate(simtrig)

    def recall(trig, unc_only=False):
        hit = hm = nm = 0
        for i in inc:
            k, o = (i["video"], i["obj"]), int(i["frame"])
            hsvf = [f for f, r in rows[k].items() if r.get("hsv_trigger") == "1" and r["is_cond"] == "0"]
            miss = not any(win(t, o) for t in hsvf)
            h = any(win(t, o) and (not unc_only or why.startswith("unc")) for t, why in trig.get(k, {}).items())
            hit += h; nm += miss; hm += h and miss
        return hit, hm, nm

    h, hm, nm = recall(calls)
    sh, shm, _ = recall(simtrig)
    L += ["## E11a: recall on E0's incidents (v1), real vs simulated",
          "| | % frame gọi concept | recall (102) | recall HSV-missed |", "|---|---|---|---|",
          f"| simulated (E0 trajectory) | {se['rate']:.2f} | {sh}/{len(inc)} = {100 * sh / len(inc):.1f}% | {shm}/{nm} |",
          f"| real E11a | {rate:.2f} | {h}/{len(inc)} = {100 * h / len(inc):.1f}% | {hm}/{nm} |",
          "", f"E11a concept calls by reason: {dict(reasons)}", ""]

    def rec(J, i):
        k, o, pre = (i["video"], i["obj"]), int(i["frame"]), float(i["pre_mean"])
        after = [J[k][f] for f in range(o + 1, o + 11) if f in J[k]]
        return None if not after else max(after) >= pre - 0.1

    groups = {"all E0 incidents": inc,
              "caught by an unc call in E11a": [i for i in inc if any(
                  win(t, int(i["frame"])) and why.startswith("unc")
                  for t, why in calls.get((i["video"], i["obj"]), {}).items())]}
    groups["  of which HSV-missed"] = [i for i in groups["caught by an unc call in E11a"] if not any(
        win(f, int(i["frame"])) for f, r in rows[(i["video"], i["obj"])].items()
        if r.get("hsv_trigger") == "1" and r["is_cond"] == "0")]
    L += ["## Recovery within 10 frames on E0's incidents (max J over onset+1..+10 >= E0 pre_mean - 0.1)",
          "| group | n | E0 recovered | E11a recovered | E11a J at onset < E0 pre_mean - 0.3 (incident still there) |",
          "|---|---|---|---|---|"]
    for name, g in groups.items():
        r0 = [rec(J0, i) for i in g]; r1 = [rec(J1, i) for i in g]
        ok = [a is not None for a in r0]
        still = sum(J1[(i["video"], i["obj"])][int(i["frame"])] < float(i["pre_mean"]) - 0.3 for i in g)
        L.append(f"| {name} | {len(g)} | {sum(a for a, o in zip(r0, ok) if o)}/{sum(ok)} | "
                 f"{sum(b for b, o in zip(r1, ok) if o)}/{sum(ok)} | {still} |")
    own = list(csv.DictReader(open(f"{OUT}/incidents.csv")))
    own_unc = [i for i in own if any(win(t, int(i["frame"])) and why.startswith("unc")
                                     for t, why in calls.get((i["video"], i["obj"]), {}).items())]
    okk = [i for i in own_unc if i["recovered"] != "n/a"]
    L += ["", f"E11a's own incidents (v1): {len(own)}; with an unc call in [onset-2, onset+5]: {len(own_unc)}, "
          f"recovered {sum(i['recovered'] == '1' for i in okk)}/{len(okk)}", ""]

    cos = collections.defaultdict(list)
    for k, c in calls.items():
        prev = None
        for f in sorted(c):
            p = f"experiments/{EXP}/concepts/{k[0]}_{k[1]}_{f:05d}.pt"
            if not os.path.exists(p):
                continue
            e = torch.load(p).float().flatten()
            if prev is not None:
                cos[c[f]].append(float(torch.nn.functional.cosine_similarity(prev, e, dim=0)))
                cos["unc (all)" if c[f].startswith("unc") else "hsv (all)"].append(cos[c[f]][-1])
            prev = e
    L += ["## Cosine between the previous and the new concept (same object), by reason of the new call",
          "| reason | n | mean | median | p10 | p90 |", "|---|---|---|---|---|---|"]
    for why in ("hsv (all)", "unc (all)", "unc_empty", "unc_logit", "unc_area"):
        x = np.array(cos.get(why, []))
        if len(x):
            L.append(f"| {why} | {len(x)} | {x.mean():.3f} | {np.median(x):.3f} | {np.percentile(x, 10):.3f} | "
                     f"{np.percentile(x, 90):.3f} |")
    open(f"{OUT}/e11.md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
