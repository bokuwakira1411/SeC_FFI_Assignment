"""E17a report numbers on all160 / rest120 / subset40 (CPU). Prints markdown tables; report.md is written by hand.

Usage (from SeC/): python experiments/e17/report.py
References: E0, E10a = experiments/fps160/{E0,E10a} (same code, seed 0; masks bit-identical to the earlier runs,
see experiments/fps160/report.md), with per-frame logs for all 160 videos. E17a = experiments/e17/E17a.
Evaluation: sav_evaluator.py --scene_info --strict on symlinked pred dirs (experiments/e17/eval/<run>_<set>/masks).
Bootstrap as experiments/final_report.py (objects; video clusters; 10,000; seed 0).
Incidents as experiments/analyze_frames.py (J drop > 0.3 vs mean of previous 5 frames; recovered = max J over
f+1..f+10 >= previous mean - 0.1; at_cut = msl5 cut within +-2, experiments/e17/scenecuts_msl5_all160.json;
concept = a concept call for this object within +-2). fps = n_frames / (video_sec - E17 logging time).
Writes experiments/e17/lost_calls.csv (one row per concept call while lost).
"""
import csv
import glob
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "experiments")
from final_report import cluster_delta, paired_delta, parse_eval, per_object  # noqa: E402

D = "experiments/e17"
S40 = [l.strip() for l in open("experiments/subset40.txt") if l.strip()]
R120 = [l.strip() for l in open("experiments/full160/rest120.txt") if l.strip()]
SETS = {"all160": S40 + R120, "rest120": R120, "subset40": S40}
GT = {"all160": "experiments/full160/gt_all160", "rest120": "experiments/full160/gt_rest120",
      "subset40": "experiments/gt_subset40"}
RUNS = {"E0": "experiments/fps160/E0", "E10a": "experiments/fps160/E10a", "E17a": f"{D}/E17a"}
CUTS = json.load(open(f"{D}/scenecuts_msl5_all160.json"))
ANN = "data/SeCVOS/Annotations"
PY = os.environ.get("PYTHON", "python")  # evaluator env (skimage); this script needs pandas (base env)


def evaluate(run, s):
    d = f"{D}/eval/{run}_{s}"
    os.makedirs(f"{d}/masks", exist_ok=True)
    for v in SETS[s]:
        link = f"{d}/masks/{v}"
        if not os.path.islink(link):
            os.symlink(os.path.abspath(f"{RUNS[run]}/masks/{v}"), link)
    if not os.path.exists(f"{d}/eval.txt"):
        with open(f"{d}/eval.tmp", "w") as fh:
            subprocess.run([PY, "vos_evaluation/sav_evaluator.py", "--gt_root", GT[s],
                            "--pred_root", f"{d}/masks", "--scene_info", "data/SeCVOS/scene_info.json",
                            "--strict"], stdout=fh, stderr=subprocess.STDOUT, check=True,
                           env={**os.environ, "PYTHONPATH": os.getcwd()})
        os.replace(f"{d}/eval.tmp", f"{d}/eval.txt")
    return parse_eval(f"{d}/eval.txt"), per_object(f"{d}/masks")


def logs(run):
    return pd.concat([pd.read_csv(p, dtype={"obj": str}) for p in glob.glob(f"{RUNS[run]}/frames/*.csv")])


def j_curve(args):
    run, v, o = args
    out = []
    for f in sorted(os.listdir(f"{ANN}/{v}/{o}")):
        g = np.array(Image.open(f"{ANN}/{v}/{o}/{f}")) > 0
        p = np.array(Image.open(f"{RUNS[run]}/masks/{v}/{o}/{f}")) > 0
        u = (g | p).sum()
        out.append(1.0 if u == 0 else float((g & p).sum() / u))
    return (run, v, o), out


def near(frames, f, w=2):
    return any(abs(c - f) <= w for c in frames)


def incidents(run, lg, J):
    inc = []
    trig = lg[lg.triggered == 1].groupby(["video", "obj"]).frame.apply(list).to_dict()
    for (r, v, o), j in J.items():
        if r != run:
            continue
        prev = False
        for f in range(5, len(j)):
            pre = float(np.mean(j[f - 5:f]))
            is_inc = j[f] < pre - 0.3
            if is_inc and not prev:
                after = j[f + 1:f + 11]
                inc.append(dict(video=v, obj=o, frame=f, at_cut=int(near(CUTS[v], f)),
                                triggered=int(near(trig.get((v, o), []), f)),
                                recovered=None if not after else int(max(after) >= pre - 0.1)))
            prev = is_inc
    return pd.DataFrame(inc)


def ci(lo, hi):
    return f"[{lo:+.1f}, {hi:+.1f}]{' *' if lo > 0 or hi < 0 else ''}"


def main():
    L = {r: logs(r) for r in RUNS}
    e17 = L["E17a"]
    lost = e17[(e17.e17_lost == 1) & e17.e17_accept.notna()].copy()
    lost.to_csv(f"{D}/lost_calls.csv", index=False)
    keys = [(r, v, o) for r in RUNS for v in SETS["all160"] for o in sorted(os.listdir(f"{ANN}/{v}"))]
    with ProcessPoolExecutor(8) as ex:
        J = dict(ex.map(j_curve, keys, chunksize=4))
    res = {s: {r: evaluate(r, s) for r in RUNS} for s in SETS}
    E0w = sorted(SETS["all160"], key=lambda v: np.mean([x for k, x in res["all160"]["E0"][1].items() if k[0] == v]))
    for s, vids in SETS.items():
        vs = set(vids)
        print(f"\n## {s} ({len(vids)} videos, {len(res[s]['E0'][1])} objects)\n")
        # main table
        print("| Run | J&F | first / second / remain | Δ vs E0 [obj CI; video CI] | up/down vs E0 "
              "| Δ vs E10a [obj CI; video CI] | up/down vs E10a | % concept | fps (median; mean) | accepted | bank added |")
        print("|---|---|---|---|---|---|---|---|---|---|---|")
        for r in RUNS:
            e, o = res[s][r]
            seg = " / ".join(f"{e[k][0]:.1f}" for k in ("first_segment", "second_segment", "remain_segments"))
            cells = [f"{e['overall'][0]:.1f}", seg]
            for base in ("E0", "E10a"):
                if r == base or (base == "E10a" and r == "E0"):
                    cells += ["–", "–"]
                    continue
                m, lo, hi, up, dn, _ = paired_delta(res[s][base][1], o)
                _, clo, chi = cluster_delta(res[s][base][1], o)
                cells += [f"{m:+.1f} {ci(lo, hi)}; {ci(clo, chi)}", f"{up}/{dn}"]
            lg = L[r][L[r].video.isin(vs)]
            nc = lg[lg.is_cond == 0]
            vc = pd.read_csv(f"{RUNS[r]}/videos.csv")
            vc = vc[vc.video.isin(vs)]
            log_s = lg.groupby("video").e17_log_ms.sum() / 1000 if "e17_log_ms" in lg else pd.Series(dtype=float)
            fps = vc.n_frames / (vc.video_sec - vc.video.map(log_s).fillna(0))
            acc = int(nc.e17_accept.fillna(0).sum()) if "e17_accept" in nc else 0
            added = int(nc.bank_reason.fillna("").str.startswith("added").sum())
            add_c = int((nc.bank_reason == "added_concept").sum())
            cells += [f"{100 * nc.triggered.mean():.1f}", f"{fps.median():.2f}; {fps.mean():.2f}", str(acc),
                      f"{added}" + (f" ({add_c} added_concept)" if r == "E17a" else "")]
            print(f"| {r} | " + " | ".join(cells) + " |")
        # worst-10 E0 videos (by E0 all160 per-video J&F), restricted to this set
        w = [v for v in E0w if v in vs][:10]
        print(f"\n10 worst E0 videos in {s}: " + ", ".join(
            f"{r} {np.mean([x for k, x in res[s][r][1].items() if k[0] in w]):.1f}" for r in RUNS))
        # mechanism
        g = lost[lost.video.isin(vs)]
        ge = g[g.gt_empty == 0]
        print(f"\nlost calls: {len(g)} (GT empty {int((g.gt_empty == 1).sum())}); GT not empty {len(ge)}: "
              f"mean J mix {ge.J_mix.mean():.3f} con {ge.J_con.mean():.3f}; non-empty mix {(ge.mix_px > 0).mean():.1%} "
              f"con {(ge.con_px > 0).mean():.1%}; con > mix (+0.01) {(ge.J_con > ge.J_mix + .01).sum()}, "
              f"con < mix {(ge.J_con < ge.J_mix - .01).sum()}")
        a = g[g.e17_accept == 1]
        print(f"accepted {len(a)}; with GT not empty: mean J mix {a[a.gt_empty == 0].J_mix.mean():.3f} "
              f"con {a[a.gt_empty == 0].J_con.mean():.3f}")
        # misaccepts
        print("| accepted logit | n | GT empty (invented) | J_con < 0.5 (wrong object) | misaccept % | J_con median |")
        print("|---|---|---|---|---|---|")
        for name, m in [("(0, 1]", (a.con_logit > 0) & (a.con_logit <= 1)), ("> 1", a.con_logit > 1), ("all", a.con_logit > -1e9)]:
            x = a[m]
            inv = int((x.gt_empty == 1).sum())
            wr = int(((x.gt_empty == 0) & (x.J_con < 0.5)).sum())
            print(f"| {name} | {len(x)} | {inv} | {wr} | {100 * (inv + wr) / max(len(x), 1):.1f} | "
                  f"{x.J_con.median():.3f} |")
        bad = a[(a.gt_empty == 1) | (a.J_con < 0.5)]
        print("misaccept examples:\n" + bad[["video", "obj", "frame", "gt_empty", "J_mix", "J_con", "con_iou",
                                             "con_logit", "con_maxcos"]].head(8).to_string(index=False))
        # incidents
        print("\n| Run | incidents (objects) | at cut + concept | at cut, no concept | no cut + concept | "
              "no cut, no concept | recovered (all) |")
        print("|---|---|---|---|---|---|---|")
        for r in ("E10a", "E17a"):
            inc = incidents(r, L[r], {k: j for k, j in J.items() if k[1] in vs})
            cells = [f"{len(inc)} ({inc.groupby(['video', 'obj']).ngroups})"]
            for ac, tg in ((1, 1), (1, 0), (0, 1), (0, 0)):
                x = inc[(inc.at_cut == ac) & (inc.triggered == tg)]
                ok = x[x.recovered.notna()]
                cells.append(f"{len(x)} (rec. {int(ok.recovered.sum())}/{len(ok)})")
            ok = inc[inc.recovered.notna()]
            cells.append(f"{int(ok.recovered.sum())}/{len(ok)} = {100 * ok.recovered.mean():.1f}%")
            print(f"| {r} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    main()
