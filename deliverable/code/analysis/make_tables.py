"""Build deliverable/results/tables/ from the per-object scores of each run, and recompute the headline numbers.

Run from the SeC repo root (CPU only, needs numpy + pandas):
    python deliverable/code/analysis/make_tables.py
Input: <run>/masks/results_overall.csv written by vos_evaluation/sav_evaluator.py (one row per object: J&F, J, F).
Every number is recomputed here from those per-object files; nothing is copied from the reports.
Bootstrap as experiments/final_report.py: paired difference over objects, 95 % percentile CI, 10,000 resamples,
numpy default_rng(0); "video" CI resamples videos (objects of a video kept together, object-weighted mean).
"""
import os

import numpy as np
import pandas as pd

E = "experiments"
OUT = "deliverable/results/tables"
N_BOOT = 10000
S40 = [l.strip() for l in open(f"{E}/subset40.txt") if l.strip()]
R120 = [l.strip() for l in open(f"{E}/full160/rest120.txt") if l.strip()]

# 160-video configurations: one or more mask folders (each with masks/results_overall.csv)
RUNS160 = {
    "E0": [f"{E}/fps160/E0"],
    "E10a": [f"{E}/fps160/E10a"],
    "E12": [f"{E}/bank/E12", f"{E}/full160/E12_rest120"],   # subset40 part + rest120 part
    "E17a": [f"{E}/e17/E17a"],
    "E18_output": [f"{E}/e18/E18a_output"],
    "E18_pixel": [f"{E}/e18/E18a_pixel"],
    "E0_secvos_full": ["outputs/secvos_full"],                # earlier original-code run (cross-check only)
}
# subset40 design runs (E2b: 7 objects of the multi-object videos)
RUNS40 = {r: f"{E}/{r}" for r in ["E0", "E1", "E2", "E2b", "E3a", "E7", "E10a", "E11a", "E4b40", "E6"]}
RUNS40.update({r: f"{E}/bank/{r}" for r in ["E12", "E14", "E13a", "E13b"]})
CONTROL40 = {"E13a": "E6", "E12": "E10a", "E14": "E12", "E13b": "E10a"}


def read(d):
    p = f"{d}/results_overall.csv" if d.startswith("outputs") else f"{d}/masks/results_overall.csv"
    t = pd.read_csv(p, skipinitialspace=True, dtype=str)
    t.columns = [c.strip() for c in t.columns]
    t["video"] = t["sequence"].str.strip()
    t = t[t.video != "Global score"].copy()
    t["object"] = t["obj"].str.strip()
    for c in ["J&F", "J", "F"]:
        t[c] = t[c].astype(float)
    return t[["video", "object", "J", "F", "J&F"]]


def split(v):
    return "subset40" if v in S40 else "rest120" if v in R120 else "other"


def boot(a, b):
    """b - a, paired on the objects of b. Returns mean, object CI, video CI, n up, n down, n."""
    k = b.index
    d = (b - a.loc[k]).to_numpy()
    rng = np.random.default_rng(0)
    lo, hi = np.percentile(d[rng.integers(0, len(d), size=(N_BOOT, len(d)))].mean(1), [2.5, 97.5])
    g = pd.Series(d, index=k).groupby(level=0)
    s, n = g.sum().to_numpy(), g.size().to_numpy()
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(s), size=(N_BOOT, len(s)))
    vlo, vhi = np.percentile(s[idx].sum(1) / n[idx].sum(1), [2.5, 97.5])
    return d.mean(), (lo, hi), (vlo, vhi), int((d > 0).sum()), int((d < 0).sum()), len(d)


def fmt_ci(ci):
    lo, hi = ci
    return f"[{lo:+.1f}, {hi:+.1f}]" + (" *" if lo > 0 or hi < 0 else "")


def main():
    os.makedirs(OUT, exist_ok=True)
    # ---- 160 videos: one CSV per configuration + a wide J&F table
    T = {}
    for name, dirs in RUNS160.items():
        t = pd.concat([read(d) for d in dirs])
        t = t[t.video.isin(S40 + R120)]
        if name == "E12":  # each part only for its own split
            t = t.drop_duplicates(["video", "object"], keep="last")
        t.insert(2, "split", t.video.map(split))
        t = t.sort_values(["video", "object"]).reset_index(drop=True)
        assert len(t) == 182, (name, len(t))
        t.to_csv(f"{OUT}/per_object_{name}.csv", index=False)
        T[name] = t.set_index(["video", "object"])["J&F"]
    wide = pd.DataFrame(T)
    wide.insert(0, "split", [split(v) for v, _ in wide.index])
    wide.to_csv(f"{OUT}/per_object_jf_160_wide.csv")

    rows = []
    for s, vids in [("rest120", R120), ("all160", S40 + R120), ("subset40", S40)]:
        sub = wide[wide.index.get_level_values(0).isin(vids)]
        for name in RUNS160:
            r = dict(set=s, run=name, n_obj=len(sub), JF=sub[name].mean())
            for ref in ["E0", "E10a"]:
                if name not in (ref, "E0_secvos_full"):
                    m, oci, vci, up, dn, _ = boot(sub[ref], sub[name])
                    r.update({f"d_vs_{ref}": m, f"ci_obj_vs_{ref}": fmt_ci(oci),
                              f"ci_vid_vs_{ref}": fmt_ci(vci), f"up_down_vs_{ref}": f"{up}/{dn}"})
            rows.append(r)
    pd.DataFrame(rows).round(2).to_csv(f"{OUT}/summary_160.csv", index=False)

    # E18: split G = E17a - E10a into output (output - E10a), pixel memory (pixel - output), bank (E17a - pixel)
    rows = []
    for s, vids in [("rest120", R120), ("all160", S40 + R120), ("subset40", S40)]:
        sub = wide[wide.index.get_level_values(0).isin(vids)]
        g = (sub["E17a"] - sub["E10a"]).mean()
        for part, a, b in [("output", "E10a", "E18_output"), ("pixel memory", "E18_output", "E18_pixel"),
                           ("bank", "E18_pixel", "E17a"), ("total G", "E10a", "E17a")]:
            m, oci, vci, up, dn, _ = boot(sub[a], sub[b])
            rows.append(dict(set=s, part=part, delta=m, ci_obj=fmt_ci(oci), ci_vid=fmt_ci(vci), share_of_G=m / g))
    pd.DataFrame(rows).round(2).to_csv(f"{OUT}/e18_decomposition.csv", index=False)

    # 10 worst videos of E0 on all160 (per-video mean over objects), mean J&F over their objects
    vmean = wide["E0"].groupby(level=0).mean().sort_values(kind="stable")
    worst = vmean.index[:10]
    w = wide[wide.index.get_level_values(0).isin(worst)]
    pd.DataFrame({"video": list(worst), "E0_video_JF": vmean.loc[worst].round(1).to_numpy()}).to_csv(
        f"{OUT}/worst10_E0_videos.csv", index=False)
    worst_means = {k: w[k].mean() for k in ["E0", "E10a", "E17a", "E18_pixel"]}

    # ---- subset40 design runs
    P = {n: read(d).set_index(["video", "object"]) for n, d in RUNS40.items()}
    long = pd.concat([P[n].assign(run=n) for n in P]).reset_index()
    long.insert(2, "split", "subset40")
    long[["run", "video", "object", "split", "J", "F", "J&F"]].to_csv(
        f"{OUT}/per_object_subset40_all_runs.csv", index=False)
    rows = []
    for n in RUNS40:
        b = P[n]["J&F"]
        r = dict(run=n, n_obj=len(b), JF=b.mean())
        if n != "E0":
            m, oci, vci, up, dn, _ = boot(P["E0"]["J&F"], b)
            r.update(d_vs_E0=m, ci_obj_vs_E0=fmt_ci(oci), ci_vid_vs_E0=fmt_ci(vci), up_down_vs_E0=f"{up}/{dn}")
        if n in CONTROL40:
            c = CONTROL40[n]
            m, oci, vci, up, dn, _ = boot(P[c]["J&F"], b)
            r.update(control=c, d_vs_control=m, ci_obj_vs_control=fmt_ci(oci), ci_vid_vs_control=fmt_ci(vci),
                     up_down_vs_control=f"{up}/{dn}")
        rows.append(r)
    pd.DataFrame(rows).round(2).to_csv(f"{OUT}/summary_subset40.csv", index=False)

    # ---- number check against the values quoted in the slides / reports
    S160 = pd.read_csv(f"{OUT}/summary_160.csv").set_index(["set", "run"])["JF"]
    S40t = pd.read_csv(f"{OUT}/summary_subset40.csv").set_index("run")["JF"]
    exp = [("rest120", r, v) for r, v in [("E0", 70.0), ("E10a", 72.2), ("E12", 70.8), ("E17a", 77.4),
                                          ("E18_output", 72.7), ("E18_pixel", 77.1)]]
    exp += [("all160", r, v) for r, v in [("E0", 69.9), ("E10a", 72.6), ("E12", 71.9), ("E17a", 78.6)]]
    chk = [dict(set=s, run=r, expected=v, recomputed=S160[(s, r)]) for s, r, v in exp]
    chk += [dict(set="subset40 (design runs)", run=r, expected=v, recomputed=S40t[r])
            for r, v in [("E0", 69.8), ("E10a", 73.9), ("E13b", 81.1)]]
    chk += [dict(set="subset40 (160-video runs)", run=r, expected=v, recomputed=S160[("subset40", r)])
            for r, v in [("E0", 69.8), ("E10a", 73.9)]]
    chk += [dict(set="10 worst E0 videos (all160)", run=r, expected=v, recomputed=worst_means[r])
            for r, v in [("E0", 19.1), ("E10a", 28.7), ("E17a", 48.8)]]
    chk += [dict(set=f"{s} (E0 cross-check)", run="E0_secvos_full", expected=S160[(s, "E0")],
                 recomputed=S160[(s, "E0_secvos_full")]) for s in ["rest120", "all160", "subset40"]]
    c = pd.DataFrame(chk)
    c["diff"] = c.recomputed - c.expected
    c["flag_gt_0.1"] = c["diff"].abs() > 0.1 + 1e-9
    c.round(3).to_csv(f"{OUT}/number_check.csv", index=False)
    print(c.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
