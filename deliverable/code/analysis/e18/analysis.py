"""E18 step 1 (CPU only): analysis of existing E17a / E10a logs and masks. Prints markdown; analysis_report.md is
written by hand.

Usage (from SeC/): python3 experiments/e18/analysis.py > experiments/e18/analysis_numbers.txt
Inputs: E17a = experiments/e17/E17a, E10a = experiments/fps160/E10a (frames/*.csv, masks/), GT data/SeCVOS/Annotations
(GT files from the object's first annotated frame on; earlier frames are not scored and skipped here; J of an
empty GT + empty mask = 1). Per-object J&F from experiments/e17/eval (report.py).

A1. Per accept (E17a, e17_accept = 1): window = accept frame .. frame before the next accept of the same object
    (or video end). ΔJ = sum over the window of J(E17a) - J(E10a) (unit: frame-J). Damage length = frames from the
    accept until J(E17a) >= J(E10a) again (censored at video end). Type: correct / invented (GT empty) /
    wrong object (GT not empty, J of the kept mask < 0.5).
A2. Signals on accepts (correct vs misaccept; and correct vs invented): con_maxcos (max cosine of the kept mask's
    descriptor to the bank frames given to the MLLM; the mean cosine was not logged), con_iou, con_logit,
    con_px / frame pixels. AUC = P(signal of a correct accept > signal of a misaccept), ties 0.5.
    A signal with AUC >= 0.70 on subset40: threshold on subset40 keeping >= 90% of correct accepts, applied to rest120.
A3. (for Q4 / E18c) E10a pixel memory on lost frames (lost as E17: the 2 previous tracked frames have an empty mask or
    logit <= 0): number of memory frames (sam_mem_frames, start frame excluded) that are bad.
Writes experiments/e18/accepts.csv (one row per E17a accept).
"""
import glob
import os
import sys
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, "experiments/e17")
import report as R  # noqa: E402

D = "experiments/e18"
RUN = {"E17a": "experiments/e17/E17a", "E10a": "experiments/fps160/E10a"}
ANN = "data/SeCVOS/Annotations"
SETS = {"rest120": R.SETS["rest120"], "subset40": R.SETS["subset40"], "all160": R.SETS["all160"]}
SIGNALS = [("con_maxcos", "cos max tới bank"), ("con_iou", "pred_iou"), ("con_logit", "object_score_logit"),
           ("area", "diện tích mask / frame")]


def j_pair(args):
    v, o = args
    # indexed by frame number; NaN on frames without a GT file (before the object's first annotation)
    n = len(os.listdir(f"{RUN['E17a']}/masks/{v}/{o}"))
    j = {r: np.full(n, np.nan) for r in RUN}
    for f in sorted(os.listdir(f"{ANN}/{v}/{o}")):
        g = np.array(Image.open(f"{ANN}/{v}/{o}/{f}")) > 0
        for r, d in RUN.items():
            p = np.array(Image.open(f"{d}/masks/{v}/{o}/{f}")) > 0
            u = (g | p).sum()
            j[r][int(f[:-4])] = 1.0 if u == 0 else float((g & p).sum() / u)
    return (v, o), (j["E17a"], j["E10a"], g.size)


def auc(pos, neg):
    """P(pos > neg) + 0.5 P(tie)."""
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    gt = (pos[:, None] > neg[None, :]).mean()
    eq = (pos[:, None] == neg[None, :]).mean()
    return gt + 0.5 * eq


def main():
    lg = {r: pd.concat([pd.read_csv(p, dtype={"obj": str}) for p in glob.glob(f"{d}/frames/*.csv")])
          for r, d in RUN.items()}
    keys = [(v, o) for v in SETS["all160"] for o in sorted(os.listdir(f"{ANN}/{v}"))]
    with ProcessPoolExecutor(8) as ex:
        J = dict(ex.map(j_pair, keys, chunksize=4))

    # ---------- A1 ----------
    a = lg["E17a"][lg["E17a"].e17_accept == 1].sort_values(["video", "obj", "frame"]).copy()
    a["type"] = np.where(a.gt_empty == 1, "invented", np.where(a.J_con < 0.5, "wrong", "correct"))
    rows = []
    for (v, o), g in a.groupby(["video", "obj"]):
        j17, j10, npx = J[(v, o)]
        fr = list(g.frame.astype(int))
        for i, (f, r) in enumerate(zip(fr, g.itertuples())):
            end = fr[i + 1] if i + 1 < len(fr) else len(j17)
            d = j17[f:end] - j10[f:end]
            rec = np.nonzero(j17[f:] >= j10[f:])[0]  # NaN frames compare False
            rows.append(dict(video=v, obj=o, frame=f, type=r.type, win_len=end - f, dJ=np.nansum(d),
                             dJ_frame=d[0], dmg_len=int(rec[0]) if len(rec) else len(j17) - f,
                             dmg_censored=int(len(rec) == 0), con_maxcos=r.con_maxcos, con_iou=r.con_iou,
                             con_logit=r.con_logit, area=r.con_px / npx, gt_empty=r.gt_empty, J_con=r.J_con,
                             J_mix=r.J_mix))
    A = pd.DataFrame(rows)
    A.to_csv(f"{D}/accepts.csv", index=False)

    for s in ("rest120", "all160", "subset40"):
        x = A[A.video.isin(SETS[s])]
        print(f"\n## A1 – {s}: {len(x)} lần nhận\n")
        print("| loại | số lần | ΔJ TB / lần | ΔJ trung vị / lần | tổng ΔJ | ΔJ tại frame nhận (TB) | độ dài cửa sổ TB "
              "| độ dài thiệt hại: trung vị; TB (bị cắt ở cuối video) |")
        print("|---|---|---|---|---|---|---|---|")
        for t in ("correct", "invented", "wrong", "all"):
            y = x if t == "all" else x[x.type == t]
            print(f"| {t} | {len(y)} | {y.dJ.mean():+.2f} | {y.dJ.median():+.2f} | {y.dJ.sum():+.1f} | "
                  f"{y.dJ_frame.mean():+.3f} | {y.win_len.mean():.1f} | {y.dmg_len.median():.0f}; {y.dmg_len.mean():.1f} "
                  f"({int(y.dmg_censored.sum())}) |")

    # decreased objects (rest120, per-object J&F E17a < E10a)
    o17 = R.per_object("experiments/e17/eval/E17a_rest120/masks")
    o10 = R.per_object("experiments/e17/eval/E10a_rest120/masks")
    down = [k for k in o17 if o17[k] < o10[k]]
    mis = A[A.type != "correct"]
    n_mis_obj = sum(1 for k in down if ((mis.video == k[0]) & (mis.obj == k[1])).any())
    tot, inwin = 0.0, 0.0
    for k in down:
        j17, j10, _ = J[k]
        d = j17 - j10
        neg = np.nan_to_num(np.minimum(d, 0))
        tot += neg.sum()
        m = np.zeros(len(d), bool)
        for r in A[(A.video == k[0]) & (A.obj == k[1])].itertuples():
            if r.type != "correct":
                m[r.frame:r.frame + r.win_len] = True
        inwin += neg[m].sum()
    print(f"\nrest120: {len(down)} object giảm (J&F E17a < E10a); {n_mis_obj} có >= 1 lần nhận nhầm. "
          f"Tổng phần J giảm (chỉ cộng frame có ΔJ < 0) {tot:.1f} frame-J; trong cửa sổ sau nhận nhầm {inwin:.1f} "
          f"= {100 * inwin / tot:.1f}%.")
    no_acc = sum(1 for k in down if not ((A.video == k[0]) & (A.obj == k[1])).any())
    print(f"  object giảm mà không có lần nhận nào: {no_acc}")

    # ---------- A2 ----------
    print("\n## A2 – AUC (đúng > nhầm; 0.5 = ngẫu nhiên, < 0.5 = ngược chiều)\n")
    print("| tín hiệu | subset40: đúng vs nhầm | subset40: đúng vs bịa | rest120: đúng vs nhầm | rest120: đúng vs bịa |")
    print("|---|---|---|---|---|")
    res = {}
    for col, name in SIGNALS:
        cells = []
        for s in ("subset40", "rest120"):
            x = A[A.video.isin(SETS[s])]
            c, m, i = x[x.type == "correct"][col], x[x.type != "correct"][col], x[x.type == "invented"][col]
            res[(col, s)] = auc(c, m)
            cells += [f"{auc(c, m):.3f}", f"{auc(c, i):.3f}"]
        print(f"| {name} (`{col}`) | " + " | ".join(cells) + " |")
    for s in ("subset40", "rest120"):
        x = A[A.video.isin(SETS[s])]
        print(f"{s}: correct {int((x.type == 'correct').sum())}, invented {int((x.type == 'invented').sum())}, "
              f"wrong {int((x.type == 'wrong').sum())}")
    good = [col for col, _ in SIGNALS if res[(col, "subset40")] >= 0.70]
    print(f"\nTín hiệu có AUC subset40 >= 0.70 (đúng vs nhầm): {good or 'không có'}")
    s40 = A[A.video.isin(SETS["subset40"])]
    r120 = A[A.video.isin(SETS["rest120"])]
    for col in good:
        thr = np.quantile(s40[s40.type == "correct"][col], 0.10)  # keep >= 90% of correct accepts (higher = keep)
        k = s40[col] >= thr
        print(f"- {col} >= {thr:.4f} (chọn trên subset40): subset40 giữ {int(k[s40.type == 'correct'].sum())}"
              f"/{int((s40.type == 'correct').sum())} đúng, chặn {int((~k)[s40.type != 'correct'].sum())}"
              f"/{int((s40.type != 'correct').sum())} nhầm")
        k = r120[col] >= thr
        print(f"  rest120: giữ {int(k[r120.type == 'correct'].sum())}/{int((r120.type == 'correct').sum())} đúng; "
              f"chặn {int((~k)[r120.type != 'correct'].sum())}/{int((r120.type != 'correct').sum())} nhầm "
              f"(bịa {int((~k)[r120.type == 'invented'].sum())}/{int((r120.type == 'invented').sum())}, "
              f"bắt nhầm {int((~k)[r120.type == 'wrong'].sum())}/{int((r120.type == 'wrong').sum())}); "
              f"tỷ lệ nhận nhầm còn lại {100 * k[r120.type != 'correct'].sum() / max(k.sum(), 1):.1f}%")

    # ---------- A3 ----------
    e = lg["E10a"].sort_values(["video", "obj", "frame"])
    tot_f = n_bad_slots = n_slots = n_with_bad = 0
    bad_pos = []
    for (v, o), g in e.groupby(["video", "obj"]):
        g = g[g.is_cond == 0]
        bad = dict(zip(g.frame, (g.mask_px == 0) | (g.object_score_logit <= 0)))
        fr = list(g.frame)
        for i in range(2, len(fr)):
            if not (bad[fr[i - 1]] and bad[fr[i - 2]]):
                continue
            mem = [int(x) for x in str(g.iloc[i].sam_mem_frames).split() if x != "nan"]
            mem = [m for m in mem if m in bad]  # start frame excluded (not in bad)
            nb = sum(bad[m] for m in mem)
            tot_f += 1
            n_slots += len(mem)
            n_bad_slots += nb
            n_with_bad += nb > 0
            bad_pos += [fr[i] - m for m in mem if bad[m]]
    print(f"\n## A3 – pixel memory của E10a ở frame mất dấu (all160)\n\nframe mất dấu: {tot_f}; số frame memory TB "
          f"(không tính frame đầu): {n_slots / tot_f:.1f}; frame memory xấu TB: {n_bad_slots / tot_f:.2f}; "
          f"frame có >= 1 memory xấu: {100 * n_with_bad / tot_f:.1f}%; khoảng cách tới frame xấu trong memory: "
          f"{pd.Series(bad_pos).value_counts().sort_index().head(5).to_dict()}")


if __name__ == "__main__":
    main()
