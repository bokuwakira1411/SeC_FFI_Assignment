"""E17c step 1: offline w sweep on subset40 (CPU). Prints markdown tables; sweep_report.md is written by hand.

Usage (from SeC/): python experiments/e17c/sweep_analysis.py
Input: experiments/e17c/sweep/frames/*.csv (E10a trajectory, --fusion_sweep 0.5,...,1.0), and sweep_smoke for norms.
A lost call = a concept-call frame while lost (row with sw050_iou set). Columns per w: sw<100w>_{iou,logit,px,gate},
J_sw<100w>; sw050 = the (a+b)/2 output (mix).
J_eff(w) = J of the w mask if it passes the gate (not empty, pred IoU >= 0.78, logit > 0), else J of the mix; for
w = 0.5 J_eff = J_mix. Main set for J_eff and the w* rule: all lost calls with a GT file (GT empty -> J = 1 for an
empty mask, 0 otherwise, as the evaluator); the GT-not-empty version is printed too.
Misaccept(w) = gated masks with GT empty (invented) or J < 0.5 (wrong object); rate = / gated.
Bootstrap: Δ J_eff(w) - J_eff(0.5), objects resampled (10,000, seed 0).
Writes experiments/e17c/sweep_calls.csv and sweep_plot.png.
"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

D = "experiments/e17c"
WS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
K = {w: f"sw{round(w * 100):03d}" for w in WS}
MIS_MAX = 0.20


def load(run):
    df = pd.concat([pd.read_csv(p, dtype={"obj": str}) for p in glob.glob(f"{D}/{run}/frames/*.csv")])
    return df[df.sw050_iou.notna()].copy()


def boot_obj(c, col_a, col_b, n=10000, seed=0):
    g = c.groupby(["video", "obj"])
    s = (g[col_a].sum() - g[col_b].sum()).values
    cnt = g.size().values
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(s), (n, len(s)))
    d = s[idx].sum(1) / cnt[idx].sum(1)
    return np.percentile(d, 2.5), np.percentile(d, 97.5)


def main():
    c = load("sweep")
    c = c[c.gt_empty.notna()].copy()
    assert np.allclose(c.J_sw050, c.J_mix), "sw050 must be the mix output"
    for w in WS:
        k = K[w]
        c[f"eff_{k}"] = c.J_mix if w == 0.5 else np.where(c[f"{k}_gate"] == 1, c[f"J_{k}"], c.J_mix)
        c[f"mis_{k}"] = (c[f"{k}_gate"] == 1) & ((c.gt_empty == 1) | (c[f"J_{k}"] < 0.5))
    c.to_csv(f"{D}/sweep_calls.csv", index=False)
    ge = c[c.gt_empty == 0]
    print(f"lost calls with GT: {len(c)} ({c.groupby(['video', 'obj']).ngroups} objects, "
          f"{c.video.nunique()} videos); GT empty {int((c.gt_empty == 1).sum())}, GT not empty {len(ge)}\n")

    print("### 1. GT not empty (per-frame, fixed E10a trajectory)\n")
    print("| w | mean J | non-empty % | gate % | gated n | mean J of gated |")
    print("|---|---|---|---|---|---|")
    for w in WS:
        k = K[w]
        gt = ge[ge[f"{k}_gate"] == 1]
        print(f"| {w:.1f} | {ge[f'J_{k}'].mean():.3f} | {100 * (ge[f'{k}_px'] > 0).mean():.1f} | "
              f"{100 * (ge[f'{k}_gate'] == 1).mean():.1f} | {len(gt)} | "
              f"{gt[f'J_{k}'].mean() if len(gt) else float('nan'):.3f} |")

    print("\n### 2-3. J_eff and misaccepts (w = 0.5: gate not used, J_eff = J mix)\n")
    print("| w | J_eff all calls | Δ vs 0.5 [object CI] | J_eff GT not empty | Δ vs 0.5 | accepted | "
          "invented (GT empty) | wrong (J < 0.5) | misaccept % | accepted J≥0.5 |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    rows = []
    for w in WS:
        k = K[w]
        e_all, e_ge = c[f"eff_{k}"].mean(), ge[f"eff_{k}"].mean()
        if w == 0.5:
            print(f"| 0.5 | {e_all:.3f} | – | {e_ge:.3f} | – | – | – | – | – | – |")
            rows.append(dict(w=w, eff=e_all, eff_ge=e_ge, mis=0.0, n_acc=0))
            continue
        a = c[c[f"{k}_gate"] == 1]
        inv = int((a.gt_empty == 1).sum())
        wr = int(((a.gt_empty == 0) & (a[f"J_{k}"] < 0.5)).sum())
        mis = (inv + wr) / max(len(a), 1)
        lo, hi = boot_obj(c, f"eff_{k}", "eff_sw050")
        print(f"| {w:.1f} | {e_all:.3f} | {e_all - c.eff_sw050.mean():+.3f} [{lo:+.3f}, {hi:+.3f}] | {e_ge:.3f} | "
              f"{e_ge - ge.eff_sw050.mean():+.3f} | {len(a)} | {inv} | {wr} | {100 * mis:.1f} | {len(a) - inv - wr} |")
        rows.append(dict(w=w, eff=e_all, eff_ge=e_ge, mis=mis, n_acc=len(a)))
    R = pd.DataFrame(rows)

    # w* rule (fixed before the run): misaccept < 20%; max J_eff; within 0.01 -> smaller w;
    # no w > 0.5 with J_eff >= J_eff(0.5) + 0.02 -> negative result
    base = R.eff[R.w == 0.5].iloc[0]
    ok = R[(R.w == 0.5) | (R.mis < MIS_MAX)]
    best = ok.eff.max()
    w_star = ok[ok.eff >= best - 0.01].w.min()
    gain = ok[ok.w > 0.5].eff.max() - base if (ok.w > 0.5).any() else float("nan")
    print(f"\nw*: eligible {list(ok.w)}; best J_eff {best:.3f}; w* = {w_star}; "
          f"best gain of w > 0.5 over 0.5 = {gain:+.3f} -> {'GO' if gain >= 0.02 else 'STOP (negative)'}")

    # examples of misaccepts per w
    for w in WS[1:]:
        k = K[w]
        m = c[c[f"mis_{k}"]]
        print(f"misaccepts w={w}: " + "; ".join(
            f"{r.video}/{r.obj} f{int(r.frame)} (GT empty {int(r.gt_empty)}, J {r[f'J_{k}']:.2f}, "
            f"logit {r[f'{k}_logit']:.2f})" for _, r in m.iterrows()))

    # norms on lost calls (smoke + subset40)
    nm = pd.concat([load("sweep_smoke"), c])
    nm = nm[nm.norm_a.notna()]
    print(f"\n### norms on lost calls (n = {len(nm)}): mean per-pixel L2 norm, ratio to |a| [min, max]\n")
    print("| feature | norm | / norm a |")
    print("|---|---|---|")
    for name, col in [("a (memory)", "norm_a"), ("b (concept)", "norm_b")] + \
            [(f"f_w, w={w}", f"norm_f{round(w * 100):03d}") for w in WS]:
        r = nm[col] / nm.norm_a
        print(f"| {name} | {nm[col].mean():.2f} | {r.mean():.3f} [{r.min():.3f}, {r.max():.3f}] |")
    print(f"cos(a, b) mean {nm.cos_ab.mean():.3f}")

    # figure: two panels sharing w (no dual axis)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=150)
    for ax in (ax1, ax2):
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="#e5e4e0", lw=0.8)
        ax.set_axisbelow(True)
        ax.set_xticks(WS)
        ax.set_xlabel("w (trọng số nhánh concept)")
    ax1.plot(R.w, R.eff, color="#2a78d6", lw=2, marker="o", ms=6)
    ax1.set_title("J hiệu dụng (mọi lần gọi lúc mất dấu)", fontsize=10, loc="left")
    m = R[R.w > 0.5]
    ax2.plot(m.w, 100 * m.mis, color="#2a78d6", lw=2, marker="o", ms=6)
    ax2.axhline(100 * MIS_MAX, color="#52514e", lw=1, ls="--")
    ax2.text(0.6, 100 * MIS_MAX + 1, "ngưỡng 20%", fontsize=8, color="#52514e")
    ax2.set_title("Tỷ lệ nhận nhầm (% số mask qua cổng)", fontsize=10, loc="left")
    ax2.set_xlim(0.45, 1.05)
    ax2.set_ylim(0, max(30, 100 * m.mis.max() + 5))
    for ax in (ax1, ax2):
        ax.axvline(w_star, color="#eb6834", lw=1, ls=":")
    fig.tight_layout()
    fig.savefig(f"{D}/sweep_plot.png")
    R.to_csv(f"{D}/sweep_by_w.csv", index=False)


if __name__ == "__main__":
    main()
