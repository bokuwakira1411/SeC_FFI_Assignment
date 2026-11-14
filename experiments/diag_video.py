"""Per-frame diagnosis of SeC runs (CPU only): metrics, timeline PNG, diagnostic mp4, summary table.

Usage (from SeC/):
  python experiments/diag_video.py --exp experiments/D0 --videos experiments/subset_diag5.txt \
      --out experiments/D0/diag [--fps 10]

Inputs: <exp>/frames/<video>.csv (per-frame log written by vos_inference.py --exp_log_dir),
        <exp>/masks/<video>/<obj>/*.png, data/SeCVOS/JPEGImages, data/SeCVOS/Annotations.
Outputs in --out: <video>_<obj>_metrics.csv, <video>_<obj>_timeline.png, <video>_<obj>.mp4,
                  summary.csv, summary.md
J and F per frame use the Evaluator of vos_evaluation/utils/sav_benchmark.py; the video J&F is the mean
over frames[1:-1] as in sav_evaluator (overall).
"""
import argparse
import csv
import os
import subprocess
import sys

import cv2
import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from vos_evaluation.utils.sav_benchmark import Evaluator  # noqa: E402

HSV_THRESH = 0.35
BANK_SCORE_THRESH = 1.0
RED, GREEN, YELLOW, WHITE, ORANGE = (0, 0, 255), (0, 200, 0), (0, 255, 255), (255, 255, 255), (0, 165, 255)


def load_mask(path):
    return np.array(Image.open(path)) > 0


def frame_metrics(gt_masks, pred_masks):
    """Per-frame J, F (fractions) with the official Evaluator; the object is pre-registered so that
    frames where GT and prediction are both empty count as J = F = 1, as in the official code once the
    object has been seen."""
    ev = Evaluator()
    ev.objects_in_gt.add(True)
    for gt, pred in zip(gt_masks, pred_masks):
        ev.feed_frame(mask=pred, gt=gt)
    return np.array(ev.object_iou[True]), np.array(ev.boundary_f[True])


def decision(row):
    if row["is_cond"] == "1":
        return "init frame (GT mask)", ""
    if row["triggered"] != "1":
        return "rejected", "no_trigger"
    r = row["bank_reason"]
    if r == "added":
        return "added", ""
    if r.startswith("obj_score<="):
        return "rejected", f"{r} ({float(row['object_score_logit']):.2f})"
    return "rejected", r


def worst_window(jf, win):
    """Start/end frame (inclusive) of the window of `win` frames with the lowest mean J&F."""
    if len(jf) <= win:
        return 0, len(jf) - 1, float(np.mean(jf))
    m = np.convolve(jf, np.ones(win) / win, mode="valid")
    s = int(np.argmin(m))
    return s, s + win - 1, float(m[s])


def low_runs(jf, thresh=0.5, min_len=1):
    """Contiguous runs (start, end) with J&F < thresh."""
    runs, s = [], None
    for i, v in enumerate(list(jf) + [1.0]):
        if v < thresh and s is None:
            s = i
        elif v >= thresh and s is not None:
            if i - s >= min_len:
                runs.append((s, i - 1))
            s = None
    return runs


def plot_timeline(rows, j, f, jf, title, size, dpi, bank_thresh=BANK_SCORE_THRESH):
    n = len(rows)
    x = np.arange(n)
    hsv = np.array([float(r["hsv_dist"]) if r["hsv_dist"] != "" else np.nan for r in rows])
    score = np.array([float(r["object_score_logit"]) for r in rows])
    piou = np.array([float(r["pred_iou"]) if r["pred_iou"] != "" else np.nan for r in rows])
    trig = [i for i, r in enumerate(rows) if r["triggered"] == "1"]
    added = [i for i, r in enumerate(rows) if r["bank_reason"] == "added"]
    fn = [i for i, r in enumerate(rows) if r["_fn"]]
    fp = [i for i, r in enumerate(rows) if r["_fp"]]

    fig, axes = plt.subplots(3, 1, figsize=size, dpi=dpi, sharex=True,
                             gridspec_kw=dict(height_ratios=[1.3, 1, 1], hspace=0.08))
    a0, a1, a2 = axes
    a0.plot(x, jf, color="tab:blue", lw=1.6, label="J&F")
    a0.plot(x, j, color="tab:cyan", lw=0.8, alpha=0.8, label="J")
    a0.plot(x, piou, color="gray", lw=0.8, ls="--", label="pred IoU")
    for i in fn:
        a0.axvspan(i - 0.5, i + 0.5, ymin=0, ymax=0.08, color="red", lw=0)
    for i in fp:
        a0.axvspan(i - 0.5, i + 0.5, ymin=0, ymax=0.08, color="purple", lw=0)
    a0.set_ylim(-0.02, 1.05)
    a0.set_ylabel("J&F / J")
    a0.set_title(title, fontsize=9)

    a1.plot(x, hsv, color="tab:orange", lw=1.2, label="hsv_dist")
    a1.axhline(HSV_THRESH, color="red", lw=0.9, ls="--", label=f"threshold {HSV_THRESH}")
    a1.set_ylabel("hsv_dist")
    a1.set_ylim(0, max(0.6, np.nanmax(hsv) * 1.1 if np.isfinite(hsv).any() else 0.6))
    a1.legend(loc="upper right", fontsize=7, framealpha=0.7)

    a2.plot(x, score, color="tab:green", lw=1.2, label="object_score_logit")
    a2.axhline(bank_thresh, color="red", lw=0.9, ls="--", label=f"bank threshold {bank_thresh:g}")
    a2.axhline(0, color="black", lw=0.5)
    a2.set_ylabel("obj score")
    a2.set_xlabel("frame")
    a2.legend(loc="upper right", fontsize=7, framealpha=0.7)

    for ax in axes:
        for i in trig:
            ax.axvline(i, color="orange", lw=0.9, ls=":", alpha=0.9)
        for i in added:
            ax.axvline(i, color="green", lw=1.4, alpha=0.8)
        ax.set_xlim(-0.5, n - 0.5)
        ax.grid(alpha=0.25)
    # legend for the markers
    marks = [plt.Line2D([], [], color="orange", ls=":", label="trigger"),
             plt.Line2D([], [], color="green", label="added to bank"),
             plt.Line2D([], [], color="red", lw=5, label="FN (GT obj, empty pred)"),
             plt.Line2D([], [], color="purple", lw=5, label="FP (no GT, pred)")]
    a0.legend(handles=a0.get_legend_handles_labels()[0] + marks, loc="upper left", bbox_to_anchor=(0, 1.0),
              fontsize=7, ncol=7, framealpha=0.8)
    fig.subplots_adjust(left=0.06, right=0.99, top=0.93, bottom=0.13)
    return fig, axes


def fig_to_bgr(fig):
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[:, :, :3]
    return cv2.cvtColor(buf, cv2.COLOR_RGB2BGR)


def overlay(img, pred, gt, scale):
    out = img.copy()
    red = np.zeros_like(out)
    red[:] = RED
    out[pred] = (0.55 * out[pred] + 0.45 * red[pred]).astype(np.uint8)
    out = cv2.resize(out, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    gt_big = cv2.resize(gt.astype(np.uint8), (out.shape[1], out.shape[0]), interpolation=cv2.INTER_NEAREST)
    cnts, _ = cv2.findContours(gt_big, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(out, cnts, -1, GREEN, 2)
    return out


def put_lines(img, lines, org=(10, 10), scale=0.6):
    """lines: list of (text, color). Draws on a translucent black box."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    lh = int(28 * scale / 0.6)
    w = max(cv2.getTextSize(t, font, scale, 2)[0][0] for t, _ in lines) + 16
    h = lh * len(lines) + 10
    x0, y0 = org
    sub = img[y0:y0 + h, x0:x0 + w]
    img[y0:y0 + h, x0:x0 + w] = (sub * 0.35).astype(np.uint8)
    for k, (t, c) in enumerate(lines):
        cv2.putText(img, t, (x0 + 8, y0 + lh * (k + 1) - 4), font, scale, c, 2, cv2.LINE_AA)


def thumb(img, mask, label, h, color):
    t = img.copy()
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(t, cnts, -1, RED, max(2, img.shape[0] // 120))
    w = int(round(img.shape[1] * h / img.shape[0]))
    t = cv2.resize(t, (w, h), interpolation=cv2.INTER_AREA)
    cv2.rectangle(t, (0, 0), (w - 1, h - 1), color, 3)
    cv2.putText(t, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(t, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHITE, 1, cv2.LINE_AA)
    return t


def process(exp, video, obj, rows, out_dir, fps, data_root, mllm_memory_size=7, oracle_frames=(),
            bank_thresh=BANK_SCORE_THRESH, after=None):
    frames_dir = os.path.join(data_root, "JPEGImages", video)
    gt_dir = os.path.join(data_root, "Annotations", video, obj)
    pred_dir = os.path.join(exp, "masks", video, obj)
    names = sorted(os.listdir(frames_dir), key=lambda p: int(os.path.splitext(p)[0]))
    n = len(names)
    rows = sorted(rows, key=lambda r: int(r["frame"]))
    assert [int(r["frame"]) for r in rows] == list(range(int(rows[0]["frame"]), n)), "frame rows incomplete"
    start = int(rows[0]["frame"])
    # pad frames before the object's start frame (not tracked) so that index == frame_idx
    pad = [dict(frame=str(i), is_cond="0", triggered="0", bank_reason="", hsv_dist="", pred_iou="",
                object_score_logit="nan", bank_frames="", ctx_frames="", sam_mem_frames="", _pad=True)
           for i in range(start)]
    rows = pad + rows

    gts = [load_mask(os.path.join(gt_dir, f"{i:05d}.png")) for i in range(n)]
    preds = [load_mask(os.path.join(pred_dir, f"{i:05d}.png")) for i in range(n)]
    j, f = frame_metrics(gts, preds)
    jf = (j + f) / 2
    for i, r in enumerate(rows):
        r["_fn"] = bool(gts[i].any() and not preds[i].any())
        r["_fp"] = bool(preds[i].any() and not gts[i].any())
    ev_slice = slice(1, n - 1)  # sav_evaluator skips the first and the last frame
    video_j, video_f = j[ev_slice].mean() * 100, f[ev_slice].mean() * 100
    video_jf = (video_j + video_f) / 2

    # --- per-frame metrics csv ---
    tag = f"{video}_{obj}"
    with open(os.path.join(out_dir, f"{tag}_metrics.csv"), "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["frame", "J", "F", "JF", "gt_px", "pred_px", "fn_empty_pred", "fp_no_gt", "hsv_dist",
                    "triggered", "bank_decision", "bank_reason", "pred_iou", "object_score_logit",
                    "bank_frames", "ctx_frames", "sam_mem_frames"])
        for i, r in enumerate(rows):
            d, why = decision(r) if not r.get("_pad") else ("not tracked", "")
            w.writerow([i, f"{j[i]:.4f}", f"{f[i]:.4f}", f"{jf[i]:.4f}", int(gts[i].sum()), int(preds[i].sum()),
                        int(r["_fn"]), int(r["_fp"]), r["hsv_dist"], r["triggered"], d, why, r["pred_iou"],
                        r["object_score_logit"], r["bank_frames"], r["ctx_frames"], r["sam_mem_frames"]])

    # --- static timeline ---
    title = f"{video} obj {obj}   J&F {video_jf:.1f}  (J {video_j:.1f}, F {video_f:.1f})"
    fig, _ = plot_timeline(rows, j, f, jf, title, size=(14, 6.5), dpi=110, bank_thresh=bank_thresh)
    fig.savefig(os.path.join(out_dir, f"{tag}_timeline.png"))
    plt.close(fig)

    # --- diagnostic video ---
    W = 1280
    img0 = cv2.imread(os.path.join(frames_dir, names[0]))
    scale = W / img0.shape[1]
    main_h = int(round(img0.shape[0] * scale)) // 2 * 2
    thumb_h = 120
    tl_h = 380
    fig, axes = plot_timeline(rows, j, f, jf, title, size=(W / 100, tl_h / 100), dpi=100, bank_thresh=bank_thresh)
    tl = fig_to_bgr(fig)
    tl = cv2.resize(tl, (W, tl_h))
    to_px = lambda xd: int(round(axes[0].transData.transform((xd, 0))[0] * W / fig.bbox.width))
    y_top = int(round((1 - axes[0].get_position().y1) * tl_h))
    y_bot = int(round((1 - axes[2].get_position().y0) * tl_h))
    plt.close(fig)

    bank_cache = {}

    def bank_thumb(fi, in_ctx):
        key = (fi, in_ctx)
        if key not in bank_cache:
            im = cv2.imread(os.path.join(frames_dir, names[fi]))
            # MLLM memory masks: GT mask for the start frame and oracle keyframes, the predicted mask otherwise
            m = gts[fi] if fi == start or fi in oracle_frames else preds[fi]
            bank_cache[key] = thumb(im, m, f"#{fi}", thumb_h - 8, YELLOW if in_ctx else (120, 120, 120))
        return bank_cache[key]

    H = main_h + thumb_h + tl_h
    mp4 = os.path.join(out_dir, f"{tag}.mp4")
    proc = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{W}x{H}",
         "-r", str(fps), "-i", "-", "-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "23", mp4],
        stdin=subprocess.PIPE)
    for i, r in enumerate(rows):
        canvas = np.zeros((H, W, 3), np.uint8)
        img = cv2.imread(os.path.join(frames_dir, names[i]))
        canvas[:main_h] = cv2.resize(overlay(img, preds[i], gts[i], scale), (W, main_h))

        if r.get("_pad"):
            lines = [(f"frame {i}   (before object start frame {start}, not tracked)", WHITE)]
        else:
            d, why = decision(r)
            hsv_txt = r["hsv_dist"] if r["hsv_dist"] == "" else f"{float(r['hsv_dist']):.3f}"
            hsv_red = r["hsv_dist"] != "" and float(r["hsv_dist"]) > HSV_THRESH
            trig = r["triggered"] == "1"
            sc = float(r["object_score_logit"])
            flags = ("  FN: GT has object, pred empty" if r["_fn"] else "") + \
                    ("  FP: no GT, pred non-empty" if r["_fp"] else "")
            lines = [
                (f"frame {i}/{n - 1}", WHITE),
                (f"hsv_dist {hsv_txt}", RED if hsv_red else WHITE),
                (f"TRIGGER {'Y' if trig else 'N'}", ORANGE if trig else WHITE),
                (f"bank: {d}" + (f" - {why}" if why else ""), GREEN if d == "added" else WHITE),
                (f"pred IoU {r['pred_iou'] or '-'}   obj score {sc:.2f}", RED if sc <= bank_thresh else WHITE),
                (f"J {j[i]:.3f}  F {f[i]:.3f}  J&F {jf[i]:.3f}", RED if jf[i] < 0.5 else WHITE),
            ]
            if flags:
                lines.append((flags.strip(), RED))
        put_lines(canvas, lines, org=(10, 10), scale=0.6)
        put_lines(canvas, [("red fill: prediction   green outline: GT", WHITE)],
                  org=(10, main_h - 40), scale=0.5)

        # bank thumbnails (state after this frame's update); yellow border = frames the MLLM would see
        bank = [int(v) for v in r["bank_frames"].split()] if r["bank_frames"] else []
        ctx = set(bank if len(bank) <= mllm_memory_size else bank[:1] + bank[-(mllm_memory_size - 1):])
        x = 8
        label_w = 150
        cv2.putText(canvas, f"MLLM bank ({len(bank)})", (8, main_h + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    WHITE, 1, cv2.LINE_AA)
        cv2.putText(canvas, "yellow = in context", (8, main_h + 55), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    YELLOW, 1, cv2.LINE_AA)
        if r.get("sam_mem_frames"):
            sm = r["sam_mem_frames"].split()
            cv2.putText(canvas, f"SAM mem: {len(sm)} fr", (8, main_h + 80), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        WHITE, 1, cv2.LINE_AA)
            cv2.putText(canvas, f"{sm[0]}..{sm[-1]}" if len(sm) > 1 else sm[0], (8, main_h + 102),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, WHITE, 1, cv2.LINE_AA)
        x = label_w
        shown = bank
        th_w = bank_thumb(bank[0], True).shape[1] + 6 if bank else 1
        max_n = (W - label_w) // th_w
        if len(bank) > max_n:  # first frame + the most recent ones
            shown = bank[:1] + bank[-(max_n - 1):]
        for k, fi in enumerate(shown):
            t = bank_thumb(fi, fi in ctx)
            canvas[main_h + 4:main_h + 4 + t.shape[0], x:x + t.shape[1]] = t
            x += t.shape[1] + 6
            if k == 0 and len(shown) < len(bank):
                cv2.putText(canvas, "...", (x, main_h + 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, WHITE, 2)
                x += 30

        # timeline with cursor
        t_img = tl.copy()
        cx = to_px(i)
        cv2.line(t_img, (cx, y_top), (cx, y_bot), (0, 0, 0), 2)
        canvas[main_h + thumb_h:] = t_img
        proc.stdin.write(canvas.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0, f"ffmpeg failed for {mp4}"

    # --- summary row ---
    tr = [r for r in rows if not r.get("_pad")]
    dec = [decision(r) for r in tr]
    reasons = lambda key: sum(1 for d, why in dec if d == "rejected" and why.startswith(key))
    ws, we, wm = worst_window(jf, 10)
    runs = low_runs(jf, 0.5)
    longest = max(runs, key=lambda s: s[1] - s[0]) if runs else None
    final_bank = tr[-1]["bank_frames"].split()
    return dict(
        video=video, obj=obj, n_frames=n, start_frame=start,
        n_triggers=sum(r["triggered"] == "1" for r in tr),
        n_added=sum(d == "added" for d, _ in dec),
        rej_no_trigger=reasons("no_trigger"),
        rej_empty_mask=reasons("empty_mask"),
        rej_obj_score=reasons("obj_score"),
        final_bank_size=len(final_bank), final_bank=" ".join(final_bank),
        JF=round(video_jf, 1), J=round(video_j, 1), F=round(video_f, 1),
        n_fn=sum(r["_fn"] for r in tr), n_fp=sum(r["_fp"] for r in tr),
        worst10=f"{ws}-{we} ({wm * 100:.1f})",
        longest_low=f"{longest[0]}-{longest[1]} ({longest[1] - longest[0] + 1} fr)" if longest else "",
        **({f"JF_after_{after}": round(float(jf[after + 1:n - 1].mean() * 100), 1)} if after is not None else {}),
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp", required=True)
    ap.add_argument("--videos", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--data_root", default="data/SeCVOS")
    ap.add_argument("--fps", type=int, default=10)
    ap.add_argument("--oracle_frames", type=str, default="", help="frames whose bank mask is the GT, e.g. '10'")
    ap.add_argument("--bank_score_thresh", type=float, default=BANK_SCORE_THRESH)
    ap.add_argument("--after", type=int, default=None,
                    help="also report J&F over frames after+1 .. n-2 (last frame skipped as in sav_evaluator)")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    videos = [v.strip() for v in open(args.videos) if v.strip()]
    summary = []
    for v in videos:
        with open(os.path.join(args.exp, "frames", f"{v}.csv")) as fh:
            all_rows = list(csv.DictReader(fh))
        for obj in sorted({r["obj"] for r in all_rows}):
            print(f"{v} obj {obj} ...", flush=True)
            summary.append(process(args.exp, v, obj, [r for r in all_rows if r["obj"] == obj],
                                   args.out, args.fps, args.data_root,
                                   oracle_frames={int(x) for x in args.oracle_frames.split(",") if x},
                                   bank_thresh=args.bank_score_thresh, after=args.after))
    keys = list(summary[0])
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        w.writerows(summary)
    cols = ["video", "n_frames", "n_triggers", "n_added", "rej_no_trigger", "rej_empty_mask", "rej_obj_score",
            "final_bank_size", "JF", "J", "F", "n_fn", "n_fp", "worst10", "longest_low"] + \
           [k for k in keys if k.startswith("JF_after_")]
    with open(os.path.join(args.out, "summary.md"), "w") as fh:
        fh.write("| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n")
        for s in summary:
            fh.write("| " + " | ".join(str(s[c]) for c in cols) + " |\n")
    print(open(os.path.join(args.out, "summary.md")).read())


if __name__ == "__main__":
    main()
