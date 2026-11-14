"""CPU analysis of one experiment's per-frame logs against GT and PySceneDetect cuts.

Usage (from SeC/): python experiments/analyze_frames.py E0 [CUTS]
  CUTS = msl5 (default, primary: ContentDetector min_scene_len=5 -> experiments/scenecuts_msl5.json)
       | default (ContentDetector defaults -> experiments/scenecuts.json, sensitivity analysis)
Writes experiments/analysis/<EXP>/ (msl5) or experiments/analysis/<EXP>_cutsdefault/ (default):
j_per_frame.csv, incidents.csv, summary.txt
(+ bank.csv when the log has bank_reason).

Definitions
- J(frame) = IoU(pred, GT); both empty -> 1.
- cut: a frame where PySceneDetect (ContentDetector) starts a new scene (see CUTS above).
- incident at frame f: J(f) < mean(J(f-5..f-1)) - 0.3 (needs 5 previous frames), and f-1 is not
  itself an incident frame (a run of consecutive drops counts once, at its first frame).
  * at_cut: a cut within f-2..f+2
  * triggered: this object had a concept trigger within f-2..f+2
  * recovered: max J over f+1..f+10 >= mean(J(f-5..f-1)) - 0.1 ("n/a" if f is the last frame)
- missed cut: no HSV trigger (hsv_dist > threshold) within cut-2..cut+2;
  spurious trigger: HSV-trigger frame with no cut within +-2.
"""
import collections
import csv
import glob
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from PIL import Image

EXP = sys.argv[1]
CUTS = sys.argv[2] if len(sys.argv) > 2 else "msl5"
CUTS_FILE = {"msl5": "experiments/scenecuts_msl5.json", "default": "experiments/scenecuts.json"}[CUTS]
SRC = f"experiments/{EXP}"
OUT = f"experiments/analysis/{EXP}" + ("" if CUTS == "msl5" else f"_cuts{CUTS}")
GT = "data/SeCVOS/Annotations"
W = 2

def load_rows():
    rows = collections.defaultdict(dict)
    for f in glob.glob(f"{SRC}/frames/*.csv"):
        for r in csv.DictReader(open(f)):
            rows[(r["video"], r["obj"])][int(r["frame"])] = r
    return rows

def j_curve(key):
    v, o = key
    files = sorted(os.listdir(f"{GT}/{v}/{o}"))
    out = []
    for f in files:
        g = np.array(Image.open(f"{GT}/{v}/{o}/{f}")) > 0
        p = np.array(Image.open(f"{SRC}/masks/{v}/{o}/{f}")) > 0
        u = (g | p).sum()
        out.append(1.0 if u == 0 else float((g & p).sum() / u))
    return key, out

def near(frames, f, w=W):
    return any(abs(c - f) <= w for c in frames)

def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load_rows()
    cuts = json.load(open(CUTS_FILE))
    say_cuts = f"cut set: {CUTS} ({CUTS_FILE})"
    sinfo = json.load(open("data/SeCVOS/scene_info.json"))
    keys = sorted(rows)
    with ProcessPoolExecutor(4) as ex:
        J = dict(ex.map(j_curve, keys))

    lines = []
    def say(s=""):
        print(s); lines.append(s)
    say(say_cuts)

    # ---- per-frame J ----
    with open(f"{OUT}/j_per_frame.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["video", "obj", "frame", "J", "at_cut", "triggered", "hsv_dist", "mllm_memory_len"])
        for k in keys:
            v, o = k
            for f, j in enumerate(J[k]):
                r = rows[k].get(f, {})
                w.writerow([v, o, f, round(j, 4), int(f in cuts[v]), r.get("triggered", ""),
                            r.get("hsv_dist", ""), r.get("mllm_memory_len", "")])

    # ---- cuts vs scene_info, cuts vs HSV triggers ----
    n_b = n_hit = 0
    for v in cuts:
        s = sinfo[v]
        for seg in ("second_segment", "remain_segments"):
            if s[seg]:
                n_b += 1
                n_hit += near(cuts[v], s[seg][0])
    say(f"scene_info boundaries (start of second / remain segment) matched by a detected cut (+-{W}): "
        f"{n_hit}/{n_b} = {100 * n_hit / max(n_b, 1):.1f}%")
    n_cut = sum(map(len, cuts.values()))
    hsv_frames, missed, spurious, n_trig_frames = {}, 0, 0, 0
    for v in cuts:
        rv = rows[next(k for k in keys if k[0] == v)]
        hsv_frames[v] = sorted(f for f, r in rv.items() if r.get("hsv_trigger", r.get("triggered")) == "1")
        n_trig_frames += len(hsv_frames[v])
        missed += sum(not near(hsv_frames[v], c) for c in cuts[v])
        spurious += sum(not near(cuts[v], f) for f in hsv_frames[v])
    say(f"detected cuts: {n_cut} | HSV-trigger frames (per video): {n_trig_frames}")
    say(f"cuts with no HSV trigger within +-{W} (missed): {missed}/{n_cut} = {100 * missed / max(n_cut, 1):.1f}%")
    say(f"HSV triggers with no cut within +-{W} (spurious): {spurious}/{n_trig_frames} = "
        f"{100 * spurious / max(n_trig_frames, 1):.1f}%")

    # ---- incidents ----
    inc = []
    for k in keys:
        v, o = k
        j = J[k]
        trig = [f for f, r in rows[k].items() if r.get("triggered") == "1"]
        prev_inc = False
        for f in range(5, len(j)):
            pre = float(np.mean(j[f - 5:f]))
            is_inc = j[f] < pre - 0.3
            if is_inc and not prev_inc:
                after = j[f + 1:f + 11]
                inc.append(dict(video=v, obj=o, frame=f, J=round(j[f], 3), pre_mean=round(pre, 3),
                                at_cut=int(near(cuts[v], f)), triggered=int(near(trig, f)),
                                recovered="n/a" if not after else int(max(after) >= pre - 0.1)))
            prev_inc = is_inc
    with open(f"{OUT}/incidents.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(inc[0]) if inc else ["video"])
        w.writeheader(); w.writerows(inc)
    say(f"incidents (J drop > 0.3 vs mean of previous 5 frames): {len(inc)} "
        f"in {len({(i['video'], i['obj']) for i in inc})} objects")
    say("  at_cut triggered | n | recovered within 10 frames")
    for ac in (1, 0):
        for tg in (1, 0):
            g = [i for i in inc if i["at_cut"] == ac and i["triggered"] == tg]
            rec = [i for i in g if i["recovered"] == 1]
            na = sum(i["recovered"] == "n/a" for i in g)
            say(f"  {ac:6d} {tg:9d} | {len(g):3d} | {len(rec)}/{len(g) - na}")

    # ---- bank (MLLM memory) ----
    if any("bank_reason" in r for k in keys for r in rows[k].values()):
        res = list(csv.reader(open(f"{SRC}/masks/results_overall.csv")))[2:]
        jf = {(r[0].strip(), r[1].strip()): float(r[2]) for r in res}
        reasons = collections.Counter()
        with open(f"{OUT}/bank.csv", "w", newline="") as fh:
            w = csv.writer(fh)
            w.writerow(["video", "obj", "n_triggers", "n_added", "n_empty_mask", "n_score_le1",
                        "final_bank_len", "J&F"])
            stuck, other = [], []
            for k in keys:
                rr = rows[k]
                c = collections.Counter(r["bank_reason"] for r in rr.values() if r.get("bank_reason"))
                reasons.update(c)
                last = rr[max(rr)]
                final = int(last["mllm_memory_len"]) + int(last["mem_appended"])
                w.writerow([*k, sum(r["triggered"] == "1" for r in rr.values()), c["added"] + c["added_periodic"],
                            c["empty_mask"], c["obj_score<=1"], final, jf.get(k, "")])
                (stuck if final == 1 else other).append(jf.get(k, float("nan")))
        say(f"bank decisions on trigger frames: {dict(reasons)}")
        say(f"objects whose bank stays at 1 frame for the whole video: {len(stuck)}/{len(keys)}; "
            f"mean J&F {np.nanmean(stuck) if stuck else float('nan'):.1f} vs others "
            f"{np.nanmean(other) if other else float('nan'):.1f} (n={len(other)})")

    open(f"{OUT}/summary.txt", "w").write("\n".join(lines) + "\n")

if __name__ == "__main__":
    main()
