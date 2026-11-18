"""160-video results for E0 (earlier full run), E10a, E12 (CPU).

Usage (from SeC/): python experiments/full160/report160.py
Pred dirs (symlinks, experiments/full160/eval/<run>_<set>/masks): E0 = outputs/secvos_full (original code; E0
reproduces it on subset40 with the same J&F); E10a / E12 = subset40 masks (experiments/E10a, experiments/bank/E12)
+ rest120 masks (experiments/full160/<run>_rest120). Sets: all160, rest120, subset40.
Evaluated with vos_evaluation/sav_evaluator.py --scene_info --strict. Bootstrap (objects; video clusters) as
experiments/final_report.py. Writes experiments/full160/summary160.md.
"""
import csv
import glob
import os
import subprocess
import sys

import numpy as np

sys.path.insert(0, "experiments")
from final_report import cluster_delta, paired_delta, parse_eval, per_object  # noqa: E402

F = "experiments/full160"
S40 = [l.strip() for l in open("experiments/subset40.txt") if l.strip()]
R120 = [l.strip() for l in open(f"{F}/rest120.txt") if l.strip()]
SETS = {"all160": S40 + R120, "rest120": R120, "subset40": S40}
SRC = {  # run -> (subset40 masks, rest120 masks)
    "E0": ("outputs/secvos_full", "outputs/secvos_full"),
    "E10a": ("experiments/E10a/masks", f"{F}/E10a_rest120/masks"),
    "E12": ("experiments/bank/E12/masks", f"{F}/E12_rest120/masks"),
}


def evaluate(run, s):
    d = f"{F}/eval/{run}_{s}"
    os.makedirs(f"{d}/masks", exist_ok=True)
    for v in SETS[s]:
        src = SRC[run][0] if v in S40 else SRC[run][1]
        link = f"{d}/masks/{v}"
        if not os.path.islink(link):
            os.symlink(os.path.abspath(f"{src}/{v}"), link)
    gt = f"{F}/gt_{s}" if s != "subset40" else "experiments/gt_subset40"
    if not os.path.exists(f"{d}/eval.txt"):
        with open(f"{d}/eval.tmp", "w") as fh:
            subprocess.run([sys.executable, "vos_evaluation/sav_evaluator.py", "--gt_root", gt,
                            "--pred_root", f"{d}/masks", "--scene_info", "data/SeCVOS/scene_info.json",
                            "--strict"], stdout=fh, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "PYTHONPATH": os.getcwd()})
        os.replace(f"{d}/eval.tmp", f"{d}/eval.txt")
    return parse_eval(f"{d}/eval.txt"), per_object(f"{d}/masks")


def cost(run):
    """Concept-call rate and fps on rest120 (from the run's logs), median non-call frame ms."""
    d = f"{F}/{run}_rest120"
    rows = [r for p in glob.glob(f"{d}/frames/*.csv") for r in csv.DictReader(open(p)) if r["is_cond"] == "0"]
    vids = list(csv.DictReader(open(f"{d}/videos.csv")))
    calls = sum(r["triggered"] == "1" for r in rows)
    nocall = [float(r["frame_ms"]) for r in rows if r["triggered"] == "0"]
    added = sum(r["bank_reason"].startswith("added") for r in rows)
    return (100 * calls / len(rows), sum(int(v["n_frames"]) for v in vids) / sum(float(v["video_sec"]) for v in vids),
            float(np.median(nocall)), added)


def ci(lo, hi):
    return f"[{lo:+.1f}, {hi:+.1f}]{' *' if lo > 0 or hi < 0 else ''}"


def main():
    gt = f"{F}/gt_subset40"
    if not os.path.exists(gt):
        os.symlink(os.path.abspath("experiments/gt_subset40"), gt)
    md = ["# 160-video run: E0, E10a, E12\n"]
    for s in ("all160", "rest120", "subset40"):
        res = {r: evaluate(r, s) for r in SRC}
        n_obj = len(res["E0"][1])
        md.append(f"\n## {s} ({len(SETS[s])} videos, {n_obj} objects)\n")
        md.append("| Run | J&F | J | F | J&F first / second / remain | Δ vs E0 [object CI] | Δ vs E0 video CI "
                  "| up / down vs E0 | Δ vs E10a [object CI; video CI] | up / down vs E10a |")
        md.append("|---|---|---|---|---|---|---|---|---|---|")
        for r in SRC:
            e, o = res[r]
            seg = " / ".join(f"{e[k][0]:.1f}" for k in ("first_segment", "second_segment", "remain_segments"))
            cells = [f"{e['overall'][0]:.1f}", f"{e['overall'][1]:.1f}", f"{e['overall'][2]:.1f}", seg]
            if r == "E0":
                cells += ["–"] * 5
            else:
                m, lo, hi, up, dn, _ = paired_delta(res["E0"][1], o)
                _, clo, chi = cluster_delta(res["E0"][1], o)
                cells += [f"{m:+.1f} {ci(lo, hi)}", ci(clo, chi), f"{up} / {dn}"]
                if r == "E12":
                    m, lo, hi, up, dn, _ = paired_delta(res["E10a"][1], o)
                    _, clo, chi = cluster_delta(res["E10a"][1], o)
                    cells += [f"{m:+.1f} {ci(lo, hi)}; {ci(clo, chi)}", f"{up} / {dn}"]
                else:
                    cells += ["–", "–"]
            md.append(f"| {r} | " + " | ".join(cells) + " |")
    md.append("\n## Cost on rest120 (from the new runs' logs)\n")
    md.append("| Run | % frame gọi concept | fps | median non-call frame ms | bank added |")
    md.append("|---|---|---|---|---|")
    for r in ("E10a", "E12"):
        c = cost(r)
        md.append(f"| {r} | {c[0]:.2f} | {c[1]:.2f} | {c[2]:.0f} | {c[3]} |")
    md.append("\nE0 = outputs/secvos_full (earlier 160-video run, original code; no per-frame log). "
              "Δ: paired bootstrap over objects / video clusters, 10,000 resamples, seed 0; * = 95% CI excludes 0. "
              "subset40 rows use the existing subset40 masks (E10a, bank/E12); E0 there = earlier run "
              "(same J&F as experiments/E0).")
    open(f"{F}/summary160.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
