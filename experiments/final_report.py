"""Aggregate E0-E6 into tables, CSVs and figures (CPU only).

Usage (from SeC/): python experiments/final_report.py
Writes experiments/results/: summary.md, per_video_jf.csv, per_object_jf.csv,
e5_time_vs_objects.png, j_curves.png, cost_vs_accuracy.png

Delta J&F vs E0: per-object J&F differences against E0 on the same objects, mean and 95% CI from a
paired bootstrap over objects (10,000 resamples, numpy default_rng(0)); "*" = CI does not contain 0.
"""
import collections
import csv
import glob
import json
import os
import re
import subprocess
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = "experiments/results"
PY = sys.executable
SUBSETS = {"subset40": "experiments/gt_subset40", "short10": "experiments/gt_subset_short10",
           "multiobj": "experiments/gt_subset_multiobj"}
EXPS = [  # name, subset, description
    ("E0", "subset40", "baseline (HSV 0.35)"),
    ("E1", "subset40", "concept off"),
    ("E2", "subset40", "concept swapped (other video)"),
    ("E3a", "subset40", "HSV threshold 0.2"),
    ("E3b", "subset40", "HSV threshold 0.5"),
    ("E6", "subset40", "bank also every >=10 frames"),
    ("E0r", "subset40", "baseline re-run (extended log)"),
    ("E4", "short10", "concept on every frame"),
    ("E4b", "short10", "concept every frame, bank on HSV triggers"),
    ("E2b", "multiobj", "concept swapped (same video)"),
    ("E4b40", "subset40", "concept every frame, bank on HSV triggers"),
    ("E7", "subset40", "concept every 5 frames + HSV 0.35, bank on HSV triggers"),
    ("E8", "subset40", "E6 + HSV 0.2"),
    ("E7bd", "subset40", "concept every 5 frames + HSV 0.35, original bank rule"),
    ("E10a", "subset40", "concept on 3 frames after each HSV trigger + every >=15 frames, bank on HSV triggers"),
    ("E10b", "subset40", "concept on 3 frames after each HSV trigger, bank on HSV triggers"),
    ("E11a", "subset40", "HSV OR uncertainty (D=2, r=0.5, N=1, C=5), bank on HSV triggers"),
]
REF = {"E10a": "E7", "E10b": "E8", "E11a": "E3a"}   # extra paired comparison (same-cost reference)
N_BOOT = 10000


def paired_delta(a, b):
    """b - a over the objects of b (a = E0). Returns mean, CI low, CI high, n up, n down, n."""
    keys = sorted(b)
    d = np.array([b[k] - a[k] for k in keys])
    rng = np.random.default_rng(0)
    boots = d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return d.mean(), lo, hi, int((d > 0).sum()), int((d < 0).sum()), len(d)


def cluster_delta(a, b):
    """Like paired_delta, but resamples videos (all objects of a drawn video together); the statistic is
    the object-weighted mean difference, as in the evaluator's overall score. Returns mean, CI low, high."""
    by_vid = {}
    for k in b:
        by_vid.setdefault(k[0], []).append(b[k] - a[k])
    vids = sorted(by_vid)
    s = np.array([sum(by_vid[v]) for v in vids]); n = np.array([len(by_vid[v]) for v in vids])
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(vids), size=(N_BOOT, len(vids)))
    boots = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return s.sum() / n.sum(), lo, hi


def parse_eval(path):
    txt = open(path).read()
    out = {}
    for seg, jf, j, f in re.findall(r"(\w+) Summary:\nGlobal score: J&F: ([\d.]+) J: ([\d.]+) F: ([\d.]+)", txt):
        out[seg] = (float(jf), float(j), float(f))
    return out


def eval_on_subset(exp, subset):
    """Score `exp`'s masks on another GT subset via a symlinked pred dir (keeps the original CSVs)."""
    d = f"experiments/_eval/{exp}_on_{subset}"
    os.makedirs(f"{d}/masks", exist_ok=True)
    for v in os.listdir(SUBSETS[subset]):
        link = f"{d}/masks/{v}"
        if not os.path.islink(link):
            os.symlink(os.path.abspath(f"experiments/{exp}/masks/{v}"), link)
    if not os.path.exists(f"{d}/eval.txt"):
        with open(f"{d}/eval.txt", "w") as fh:
            subprocess.run([PY, "vos_evaluation/sav_evaluator.py", "--gt_root", SUBSETS[subset],
                            "--pred_root", f"{d}/masks", "--scene_info", "data/SeCVOS/scene_info.json",
                            "--strict"], stdout=fh, stderr=subprocess.STDOUT, check=True)
    return d


def per_object(masks_dir):
    rows = list(csv.reader(open(f"{masks_dir}/results_overall.csv")))[2:]
    return {(r[0].strip(), r[1].strip()): float(r[2]) for r in rows}


def frame_stats(exp):
    rows = [r for f in glob.glob(f"experiments/{exp}/frames/*.csv") for r in csv.DictReader(open(f))]
    nc = [r for r in rows if r["is_cond"] == "0"]
    trig = [r for r in nc if r["triggered"] == "1"]
    pf = np.array([float(r["predict_forward_ms"]) for r in trig if r["predict_forward_ms"]])
    vids = list(csv.DictReader(open(f"experiments/{exp}/videos.csv")))
    sec = sum(float(v["video_sec"]) for v in vids)
    added = sum(r.get("bank_reason", "") in ("added", "added_periodic") for r in nc) \
        if "bank_reason" in (nc[0] if nc else {}) else None
    added_p = sum(r.get("bank_reason", "") == "added_periodic" for r in nc) if added is not None else None
    return dict(
        trig_rate=100 * len(trig) / len(nc), n_trig=len(trig),
        fps=sum(int(v["n_frames"]) for v in vids) / sec, video_sec=sec,
        pf_mean=pf.mean() if len(pf) else float("nan"),
        pf_p95=np.percentile(pf, 95) if len(pf) else float("nan"),
        added=added, added_periodic=added_p,
        gpu_h=int(open(f"experiments/{exp}/gpu_time.txt").read()) / 3600,
    )


def main():
    os.makedirs(OUT, exist_ok=True)
    done = [e for e in EXPS if os.path.exists(f"experiments/{e[0]}/eval.txt")]
    md = []
    res, obj_jf, stats = {}, {}, {}
    for name, subset, _ in done:
        res[name] = parse_eval(f"experiments/{name}/eval.txt")
        obj_jf[name] = per_object(f"experiments/{name}/masks")
        stats[name] = frame_stats(name)
    # E0 on the smaller subsets, for E4/E4b/E2b
    for subset in ("short10", "multiobj"):
        if any(s == subset for _, s, _ in done):
            d = eval_on_subset("E0", subset)
            res[f"E0@{subset}"] = parse_eval(f"{d}/eval.txt")

    delta = {n: paired_delta(obj_jf["E0"], obj_jf[n]) for n in obj_jf if n != "E0"}
    sub = {n: s for n, s, _ in EXPS}
    cdelta = {n: cluster_delta(obj_jf["E0"], obj_jf[n]) for n in obj_jf
              if n != "E0" and sub[n] == "subset40"}

    def ci(lo, hi):
        return f"[{lo:+.1f}, {hi:+.1f}]{' *' if lo > 0 or hi < 0 else ''}"

    def refcol(n):
        if n not in REF or REF[n] not in obj_jf:
            return " – |"
        m, lo, hi, up, dn, k = paired_delta(obj_jf[REF[n]], obj_jf[n])
        _, clo, chi = cluster_delta(obj_jf[REF[n]], obj_jf[n])
        return f" vs {REF[n]}: {m:+.1f} {ci(lo, hi)}; video {ci(clo, chi)}; {up} / {dn} |"

    def row(label, n, r, extra=""):
        o = r["overall"]
        seg = " / ".join(f"{r[s][0]:.1f}" for s in ("first_segment", "second_segment", "remain_segments"))
        if n in delta:
            m, lo, hi, up, dn, k = delta[n]
            dcol = f" {m:+.1f} {ci(lo, hi)} | {up} / {dn} (n={k}) |"
        else:
            dcol = " – | – |"
        ccol = f" {ci(*cdelta[n][1:])} |" if n in cdelta else " – |"
        return f"| {label} | {o[0]:.1f} |{dcol} {o[1]:.1f} | {o[2]:.1f} | {seg} |{ccol}{refcol(n)}{extra}"

    md.append("## J&F / J / F\n")
    md.append("| Exp | J&F | ΔJ&F vs E0 [95% CI] | objects up / down | J | F | J&F first / second / remain "
              "| Δ video-cluster CI | Δ vs same-cost ref (E10a: E7, E10b: E8, E11a: E3a) [object CI; video CI; up / down] | % frame gọi concept | fps | predict_forward mean / p95 ms | bank added (periodic) | GPU h |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    order = ["E0", "E1", "E2", "E3a", "E3b", "E6", "E7", "E7bd", "E8", "E10a", "E10b", "E11a", "E4b40", "E0r",
             "E0@short10", "E4", "E4b", "E0@multiobj", "E2b"]
    desc = {n: d for n, _, d in EXPS}
    desc.update({"E0@short10": "E0 on the 10 shortest", "E0@multiobj": "E0 on the 3 multi-object videos"})
    for n in order:
        if n not in res:
            continue
        s = stats.get(n)
        bank = (f"{s['added']} ({s['added_periodic']})" if s and s["added"] is not None else "–")
        extra = (f" {s['trig_rate']:.2f} | {s['fps']:.2f} | {s['pf_mean']:.0f} / {s['pf_p95']:.0f} | {bank} | {s['gpu_h']:.2f} |"
                 if s else " | | | | |")
        md.append(row(f"{n} – {desc[n]}", n, res[n], extra))
    md.append("\nΔ: paired bootstrap over objects vs E0 on the same objects (10,000 resamples, seed 0); "
              "* = 95% CI excludes 0. Objects with equal J&F count as neither up nor down. "
              "Video-cluster CI: same, resampling the 40 videos (a video's objects stay together). "
              "Bank added: frames appended to the MLLM memory over all objects (non-cond frames; "
              "E0 has no bank log; see E0r).")
    if "E7bd" in obj_jf and "E7" in obj_jf:
        m, lo, hi, up, dn, k = paired_delta(obj_jf["E7"], obj_jf["E7bd"])
        cm, clo, chi = cluster_delta(obj_jf["E7"], obj_jf["E7bd"])
        md.append(f"\nE7bd vs E7 (original bank rule vs bank on HSV triggers): {m:+.1f} object CI {ci(lo, hi)}, "
                  f"video-cluster CI {ci(clo, chi)}; objects up / down {up} / {dn} (n={k})")

    if "E10a" in stats:
        md.append("\nE10a ran while another user's job shared the GPU (median non-call frame 139 ms vs 76 ms in "
                  "E0r/E7/E10b): its fps, predict_forward and GPU h are not comparable; masks/J&F are unaffected.")

    # per-video / per-object CSVs
    names = [n for n, _, _ in done]
    keys = sorted(set().union(*[obj_jf[n] for n in names]))
    with open(f"{OUT}/per_object_jf.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["video", "obj"] + names)
        for k in keys:
            w.writerow(list(k) + [obj_jf[n].get(k, "") for n in names])
    vids = sorted({k[0] for k in keys})
    per_vid = {n: {v: np.mean([x for k, x in obj_jf[n].items() if k[0] == v]) for v in vids
                   if any(k[0] == v for k in obj_jf[n])} for n in names}
    with open(f"{OUT}/per_video_jf.csv", "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["video"] + names)
        for v in vids:
            w.writerow([v] + [f"{per_vid[n][v]:.1f}" if v in per_vid[n] else "" for n in names])

    # E6: 10 worst videos of E0
    if "E6" in per_vid:
        worst = sorted(per_vid["E0"], key=per_vid["E0"].get)[:10]
        objs = [k for k in obj_jf["E0"] if k[0] in worst]
        md.append(f"\nE6 on the 10 worst E0 videos (object mean J&F): E0 {np.mean([obj_jf['E0'][k] for k in objs]):.1f}"
                  f" -> E6 {np.mean([obj_jf['E6'][k] for k in objs]):.1f}")
        md.append(f"E6 frames added to the bank: {stats['E6']['added']} "
                  f"({stats['E6']['added_periodic']} by the every-10-frames rule); "
                  f"E0r: {stats['E0r']['added'] if 'E0r' in stats else 'n/a'}")

    # E5 cost plot
    v0 = list(csv.DictReader(open("experiments/E0/videos.csv")))
    x = np.array([int(v["n_obj"]) for v in v0]); y = np.array([float(v["video_sec"]) for v in v0])
    nf = np.array([int(v["n_frames"]) for v in v0])
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    ax[0].scatter(x + np.random.default_rng(0).uniform(-.08, .08, len(x)), y, s=14)
    ax[0].set_xlabel("objects in video"); ax[0].set_ylabel("video time (s)"); ax[0].set_xticks(sorted(set(x)))
    ax[0].set_title("E0: time per video vs #objects")
    ax[1].scatter(nf * x, y, s=14); ax[1].set_xlabel("object-frames"); ax[1].set_ylabel("video time (s)")
    ax[1].set_title("E0: time vs object-frames")
    fig.tight_layout(); fig.savefig(f"{OUT}/e5_time_vs_objects.png", dpi=120)

    md.append(f"\nE5 (E0): {len(v0)} videos, time/video by #objects: " + ", ".join(
        f"{k} obj: {y[x == k].mean():.1f}s (n={int((x == k).sum())})" for k in sorted(set(x))))
    # cost vs accuracy on subset40
    pts = [n for n in ("E3b", "E0", "E7", "E7bd", "E3a", "E6", "E8", "E10a", "E10b", "E11a", "E4b40") if n in res]
    if pts:
        lab = {"E4b40": "E4b", "E7bd": "E7bd†", "E10a": "E10a†"}
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
        for a, key, xl in ((ax[0], "trig_rate", "frames with a concept call (%)"), (ax[1], "fps", "fps (frames/s)")):
            xs = [stats[n][key] for n in pts]; ys = [res[n]["overall"][0] for n in pts]
            a.scatter(xs, ys, s=40, color=["#d9480f" if n.startswith("E10") else "#2f9e44" if n.startswith("E11") else "#2a6fdb" for n in pts], zorder=3)
            for n, x_, y_ in zip(pts, xs, ys):
                a.annotate(lab.get(n, n), (x_, y_), textcoords="offset points", xytext=(6, 4), fontsize=9)
            a.set_xlabel(xl); a.grid(alpha=.25, lw=.6)
            for sp in ("top", "right"):
                a.spines[sp].set_visible(False)
        ax[0].set_ylabel("J&F (subset40)")
        ax[1].invert_xaxis()
        fig.suptitle("Cost vs accuracy, subset40 (40 videos, 44 objects)", fontsize=11)
        if "E7bd" in pts:
            fig.text(0.01, 0.01, "† E7bd, E10a ran while another job shared the GPU: their fps is not comparable.", fontsize=8)
        fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(f"{OUT}/cost_vs_accuracy.png", dpi=130)

    # incident tables (from analyze_frames.py, msl5 cuts)
    inc_exps = [n for n in ("E0", "E7", "E8", "E10a", "E10b", "E11a") if os.path.exists(f"experiments/analysis/{n}/incidents.csv")]
    if inc_exps:
        md.append("\n## Incidents (J drop > 0.3 vs mean of previous 5 frames; cuts = msl5; "
                  "'concept' = a concept call for this object within ±2 frames)\n")
        md.append("| Exp | incidents (objects) | at cut + concept | at cut, no concept | no cut + concept "
                  "| no cut, no concept |")
        md.append("|---|---|---|---|---|")
        for n in inc_exps:
            inc = list(csv.DictReader(open(f"experiments/analysis/{n}/incidents.csv")))
            cells = []
            for ac, tg in (("1", "1"), ("1", "0"), ("0", "1"), ("0", "0")):
                g = [i for i in inc if i["at_cut"] == ac and i["triggered"] == tg]
                ok = [i for i in g if i["recovered"] != "n/a"]
                cells.append(f"{len(g)} (rec. {sum(i['recovered'] == '1' for i in ok)}/{len(ok)})")
            md.append(f"| {n} | {len(inc)} ({len({(i['video'], i['obj']) for i in inc})}) | " + " | ".join(cells) + " |")
        md.append("\nrec. = recovered within 10 frames (max J over f+1..f+10 >= previous mean - 0.1).")

    open(f"{OUT}/summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


def plot_curves(examples):
    """examples: list of (title, video, obj). J per frame for E0 with cuts and triggers."""
    cuts = json.load(open("experiments/scenecuts_msl5.json"))
    sinfo = json.load(open("data/SeCVOS/scene_info.json"))
    jr = collections.defaultdict(list)
    for r in csv.DictReader(open("experiments/analysis/E0/j_per_frame.csv")):
        jr[(r["video"], r["obj"])].append(r)
    fig, axes = plt.subplots(len(examples), 1, figsize=(11, 2.6 * len(examples)), sharey=True)
    for ax, (title, v, o) in zip(np.atleast_1d(axes), examples):
        rr = jr[(v, o)]
        f = [int(r["frame"]) for r in rr]; j = [float(r["J"]) for r in rr]
        ax.plot(f, j, lw=1.2, label="J (E0)")
        for i, c in enumerate(cuts[v]):
            ax.axvline(c, color="tab:red", ls="--", lw=1, label="cut (PySceneDetect, min_scene_len=5)" if i == 0 else None)
        for i, seg in enumerate(s for s in ("second_segment", "remain_segments") if sinfo[v][s]):
            ax.axvline(sinfo[v][seg][0], color="tab:purple", ls=":", lw=1.5,
                       label="scene_info boundary" if i == 0 else None)
        t = [int(r["frame"]) for r in rr if r["triggered"] == "1"]
        ax.scatter(t, [1.04] * len(t), marker="v", color="tab:green", s=18, label="concept trigger")
        ax.set_ylim(-0.05, 1.1); ax.set_title(f"{title}: {v} obj {o}", fontsize=10); ax.set_ylabel("J")
    np.atleast_1d(axes)[0].legend(fontsize=8, loc="lower left", ncol=4)
    np.atleast_1d(axes)[-1].set_xlabel("frame")
    fig.tight_layout(); fig.savefig(f"{OUT}/j_curves.png", dpi=120)


if __name__ == "__main__":
    main()
    if len(sys.argv) > 1:
        plot_curves([tuple(x.split(":")) for x in sys.argv[1:]])   # "title:video:obj" ...
