"""E18 report numbers (CPU). Prints markdown; report_e18a.md / report_memory.md are written by hand.

Usage (from SeC/): python3 experiments/e18/report.py RUN [RUN ...] > experiments/e18/<name>_numbers.txt
  RUN = output | pixel | E18b (directories experiments/e18/E18a_output, E18a_pixel, E18b).
Reuses experiments/e17/report.py (evaluation, bootstrap over objects / video clusters, incidents, fps) with references
E0, E10a (experiments/fps160) and E17a (= --e17_commit full). Eval dirs in experiments/e18/eval (E0 / E10a / E17a
symlinked to experiments/e17/eval).
Misaccepts: accepted frames (e17_accept = 1) with GT empty (invented) or J of the kept mask < 0.5 (wrong object).
"Misaccepts into the bank": misaccepted frames that entered the MLLM memory (full: bank_reason added / added_concept on
the accept frame; pixel: bank rule on the accept frame; E18b: also bank_confirm add:<frame>).
Decomposition (only when output and pixel are given): output - E10a, pixel - output, full - pixel on each set, with
paired object / video-cluster CIs, and % of G = full - E10a.
"""
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd

sys.path.insert(0, "experiments/e17")
import report as R  # noqa: E402

D = "experiments/e18"
DIRS = {"output": f"{D}/E18a_output", "pixel": f"{D}/E18a_pixel", "E18b": f"{D}/E18b"}
EXTRA = sys.argv[1:]
for r in EXTRA:
    R.RUNS[r] = DIRS[r]
os.makedirs(f"{D}/eval", exist_ok=True)
for run in ("E0", "E10a", "E17a"):
    for s in R.SETS:
        link = f"{D}/eval/{run}_{s}"
        if not os.path.islink(link):
            os.symlink(os.path.abspath(f"experiments/e17/eval/{run}_{s}"), link)
R.D = D
ACC_RUNS = ["E17a"] + EXTRA


def misaccepts(lg, vs):
    g = lg[lg.video.isin(vs) & (lg.e17_accept == 1)]
    inv = g.gt_empty == 1
    wr = (g.gt_empty == 0) & (g.J_con < 0.5)
    bad = g[inv | wr]
    # into the bank: the accept frame is in the MLLM memory at the end of that frame or later (bank_frames of the
    # object's last row lists the final bank; frames dropped by trimming still count, so use any later row)
    added = set()
    for (v, o), x in lg[lg.video.isin(vs)].groupby(["video", "obj"]):
        for b in x.bank_frames.dropna().astype(str):
            added.update((v, o, int(f)) for f in b.split())
    n_bank = sum((r.video, r.obj, int(r.frame)) in added for r in bad.itertuples())
    return len(g), int(inv.sum()), int(wr.sum()), n_bank


def main():
    L = {r: R.logs(r) for r in R.RUNS}
    keys = [(r, v, o) for r in ["E10a"] + ACC_RUNS for v in R.SETS["all160"] for o in sorted(os.listdir(f"{R.ANN}/{v}"))]
    with ProcessPoolExecutor(8) as ex:
        J = dict(ex.map(R.j_curve, keys, chunksize=4))
    res = {s: {r: R.evaluate(r, s) for r in R.RUNS} for s in R.SETS}
    E0w = sorted(R.SETS["all160"], key=lambda v: np.mean([x for k, x in res["all160"]["E0"][1].items() if k[0] == v]))
    for s in ("rest120", "all160", "subset40"):
        vs = set(R.SETS[s])
        print(f"\n## {s} ({len(vs)} videos, {len(res[s]['E0'][1])} objects)\n")
        print("| Run | J&F | first / second / remain | Δ vs E10a [obj CI; video CI] | up/down vs E10a "
              "| Δ vs E17a [obj CI; video CI] | up/down vs E17a | % concept | fps (median; mean) | accepted "
              "| misaccept: invented / wrong / % | misaccepts into bank | bank added |")
        print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for r in R.RUNS:
            if r == "E0":
                continue
            e, o = res[s][r]
            seg = " / ".join(f"{e[k][0]:.1f}" for k in ("first_segment", "second_segment", "remain_segments"))
            cells = [f"{e['overall'][0]:.1f}", seg]
            for base in ("E10a", "E17a"):
                if r == base or (base == "E17a" and r == "E10a"):
                    cells += ["–", "–"]
                    continue
                m, lo, hi, up, dn, _ = R.paired_delta(res[s][base][1], o)
                _, clo, chi = R.cluster_delta(res[s][base][1], o)
                cells += [f"{m:+.1f} {R.ci(lo, hi)}; {R.ci(clo, chi)}", f"{up}/{dn}"]
            lg = L[r][L[r].video.isin(vs)]
            nc = lg[lg.is_cond == 0]
            vc = pd.read_csv(f"{R.RUNS[r]}/videos.csv")
            vc = vc[vc.video.isin(vs)]
            log_s = lg.groupby("video").e17_log_ms.sum() / 1000 if "e17_log_ms" in lg else pd.Series(dtype=float)
            fps = vc.n_frames / (vc.video_sec - vc.video.map(log_s).fillna(0))
            added = int(nc.bank_reason.fillna("").str.startswith("added").sum())
            if "bank_confirm" in nc:
                added += int(nc.bank_confirm.fillna("").str.count("add:").sum())
            cells += [f"{100 * nc.triggered.mean():.1f}", f"{fps.median():.2f}; {fps.mean():.2f}"]
            if r in ACC_RUNS:
                n, inv, wr, nb = misaccepts(L[r], vs)
                # output: the kept mask never reaches the bank (the bank holds the mix mask of that frame, if any)
                cells += [str(n), f"{inv} / {wr} / {100 * (inv + wr) / max(n, 1):.1f}%", "–" if r == "output" else str(nb)]
            else:
                cells += ["–", "–", "–"]
            cells.append(str(added))
            print(f"| {r} | " + " | ".join(cells) + " |")
        w = [v for v in E0w if v in vs][:10]
        print(f"\n10 worst E0 videos in {s}: " + ", ".join(
            f"{r} {np.mean([x for k, x in res[s][r][1].items() if k[0] in w]):.1f}" for r in R.RUNS))
        if "E18b" in EXTRA:
            c = L["E18b"][L["E18b"].video.isin(vs)].bank_confirm.fillna("")
            print(f"E18b bank_confirm: pending {int(c.str.count('pend:').sum())}, added {int(c.str.count('add:').sum())}, "
                  f"dropped {int(c.str.count('drop:').sum())}")

        if "output" in EXTRA and "pixel" in EXTRA:
            G = res[s]["E17a"][0]["overall"][0] - res[s]["E10a"][0]["overall"][0]
            print(f"\nDecomposition ({s}), G = J&F(E17a) - J&F(E10a) = {G:+.1f}\n")
            print("| part | Δ | object CI | video CI | % of G |")
            print("|---|---|---|---|---|")
            for name, a, b in (("output", "E10a", "output"), ("pixel memory", "output", "pixel"),
                               ("bank", "pixel", "E17a")):
                m, lo, hi, _, _, _ = R.paired_delta(res[s][a][1], res[s][b][1])
                _, clo, chi = R.cluster_delta(res[s][a][1], res[s][b][1])
                d = res[s][b][0]["overall"][0] - res[s][a][0]["overall"][0]
                print(f"| {name} ({b} − {a}) | {m:+.2f} (J&F {d:+.1f}) | {R.ci(lo, hi)} | {R.ci(clo, chi)} | "
                      f"{100 * m / G:.0f}% |")

        print("\n| Run | incidents (objects) | at cut + concept | at cut, no concept | no cut + concept | "
              "no cut, no concept | recovered (all) |")
        print("|---|---|---|---|---|---|---|")
        for r in ["E10a"] + ACC_RUNS:
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
