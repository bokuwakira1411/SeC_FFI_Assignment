"""E17c step 2 numbers (CPU): w* = 1.0 = E17a (no re-run) + sensitivity run E17c_w08 (w = 0.8, off the pre-set rule).
Prints markdown tables; report.md is written by hand.

Usage (from SeC/): python3 experiments/e17c/report.py > experiments/e17c/report_numbers.txt
Reuses experiments/e17/report.py (same evaluation, bootstrap, incidents, fps); adds E17c_w08 and Δ vs E17a.
Eval dirs: experiments/e17c/eval/<run>_<set> (E0 / E10a / E17a symlinked to experiments/e17/eval).
Misaccepts: accepted frames (e17_accept = 1) with GT empty (invented) or J of the kept mask < 0.5 (wrong object);
the kept mask is the f_w mask ("con" columns: w = 1.0 for E17a, 0.8 for E17c_w08).
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, "experiments/e17")
import report as R  # noqa: E402

D = "experiments/e17c"
R.RUNS["E17c_w08"] = f"{D}/E17c_w08"
os.makedirs(f"{D}/eval", exist_ok=True)
for run in ("E0", "E10a", "E17a"):
    for s in R.SETS:
        link = f"{D}/eval/{run}_{s}"
        if not os.path.islink(link):
            os.symlink(os.path.abspath(f"experiments/e17/eval/{run}_{s}"), link)
R.D = D
NEW = ("E17a", "E17c_w08")


def main():
    L = {r: R.logs(r) for r in R.RUNS}
    keys = [(r, v, o) for r in ("E10a",) + NEW for v in R.SETS["all160"] for o in sorted(os.listdir(f"{R.ANN}/{v}"))]
    with ProcessPoolExecutor(8) as ex:
        J = dict(ex.map(R.j_curve, keys, chunksize=4))
    res = {s: {r: R.evaluate(r, s) for r in R.RUNS} for s in R.SETS}
    E0w = sorted(R.SETS["all160"], key=lambda v: np.mean([x for k, x in res["all160"]["E0"][1].items() if k[0] == v]))
    for s in ("rest120", "all160", "subset40"):
        vids = R.SETS[s]
        vs = set(vids)
        print(f"\n## {s} ({len(vids)} videos, {len(res[s]['E0'][1])} objects)\n")
        print("| Run | J&F | first / second / remain | Δ vs E0 [obj CI; video CI] | Δ vs E10a [obj CI; video CI] | up/down vs E10a "
              "| Δ vs E17a [obj CI; video CI] | up/down vs E17a | % concept | fps (median; mean) | accepted | bank added |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in R.RUNS:
            e, o = res[s][r]
            seg = " / ".join(f"{e[k][0]:.1f}" for k in ("first_segment", "second_segment", "remain_segments"))
            cells = [f"{e['overall'][0]:.1f}", seg]
            for base, show_ud in (("E0", False), ("E10a", True), ("E17a", True)):
                order = list(R.RUNS)
                if order.index(r) <= order.index(base):
                    cells += ["–"] + (["–"] if show_ud else [])
                    continue
                m, lo, hi, up, dn, _ = R.paired_delta(res[s][base][1], o)
                _, clo, chi = R.cluster_delta(res[s][base][1], o)
                cells += [f"{m:+.1f} {R.ci(lo, hi)}; {R.ci(clo, chi)}"] + ([f"{up}/{dn}"] if show_ud else [])
            lg = L[r][L[r].video.isin(vs)]
            nc = lg[lg.is_cond == 0]
            vc = pd.read_csv(f"{R.RUNS[r]}/videos.csv")
            vc = vc[vc.video.isin(vs)]
            log_s = lg.groupby("video").e17_log_ms.sum() / 1000 if "e17_log_ms" in lg else pd.Series(dtype=float)
            fps = vc.n_frames / (vc.video_sec - vc.video.map(log_s).fillna(0))
            acc = int(nc.e17_accept.fillna(0).sum()) if "e17_accept" in nc else 0
            added = int(nc.bank_reason.fillna("").str.startswith("added").sum())
            add_c = int((nc.bank_reason == "added_concept").sum())
            cells += [f"{100 * nc.triggered.mean():.1f}", f"{fps.median():.2f}; {fps.mean():.2f}", str(acc),
                      f"{added}" + (f" ({add_c} added_concept)" if r in NEW else "")]
            print(f"| {r} | " + " | ".join(cells) + " |")
        w = [v for v in E0w if v in vs][:10]
        print(f"\n10 worst E0 videos in {s}: " + ", ".join(
            f"{r} {np.mean([x for k, x in res[s][r][1].items() if k[0] in w]):.1f}" for r in R.RUNS))

        print("\n| Run | lost calls (GT not empty) | J mix → kept-branch mask (GT not empty) | accepted | invented (GT empty) "
              "| wrong object (J < 0.5) | misaccept % | of which logit (0, 1]: n, misaccept % |")
        print("|---|---|---|---|---|---|---|---|")
        for r in NEW:
            g = L[r][L[r].video.isin(vs) & (L[r].e17_lost == 1) & L[r].e17_accept.notna()]
            ge = g[g.gt_empty == 0]
            a = g[g.e17_accept == 1]
            inv = int((a.gt_empty == 1).sum())
            wr = int(((a.gt_empty == 0) & (a.J_con < 0.5)).sum())
            lo = a[(a.con_logit > 0) & (a.con_logit <= 1)]
            lo_bad = int(((lo.gt_empty == 1) | (lo.J_con < 0.5)).sum())
            print(f"| {r} | {len(g)} ({len(ge)}) | {ge.J_mix.mean():.3f} → {ge.J_con.mean():.3f} | {len(a)} | {inv} | {wr} | "
                  f"{100 * (inv + wr) / max(len(a), 1):.1f} | {len(lo)}, {100 * lo_bad / max(len(lo), 1):.1f} |")
            if r == "E17c_w08":
                bad = a[(a.gt_empty == 1) | (a.J_con < 0.5)]
                print("misaccept examples (E17c_w08):\n" + bad[["video", "obj", "frame", "gt_empty", "J_mix", "J_con",
                                                                "con_iou", "con_logit"]].head(8).to_string(index=False))

        print("\n| Run | incidents (objects) | at cut + concept | at cut, no concept | no cut + concept | "
              "no cut, no concept | recovered (all) |")
        print("|---|---|---|---|---|---|---|")
        for r in ("E10a",) + NEW:
            inc = R.incidents(r, L[r], {k: j for k, j in J.items() if k[1] in vs})
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
