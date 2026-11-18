"""Render the illustration videos and keyframes (CPU only; reads saved masks and frame logs, no inference).

Run from the SeC repo root:  python deliverable/code/render/make_videos.py [--only NAME]
Each video: 3x2 grid, GT | E0 | E10a (top), E17a | E18 pixel | legend (bottom), same mask colour in all panels.
Markers (from the run's per-frame log experiments/<run>/frames/<video>.csv):
  - white dot (top right)  : a concept call for this object on this frame (column `triggered`);
  - red border             : the object is "lost" (mask empty or object score logit <= 0 on the 2 previous
                             tracked frames; same rule as E17a's `--lost_n 2`, computed from the log);
  - green border           : the concept-only mask was accepted on this frame (column `e17_accept`; E17a and E18 only).
Top-left label: configuration, frame number, J of this frame.
Output: deliverable/videos/<name>.mp4 (H.264, libx264 -preset veryfast, 1920 px wide), deliverable/keyframes/<name>_fNNN.png.
"""
import argparse
import os
import subprocess

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

E = "experiments"
IMG, ANN = "data/SeCVOS/JPEGImages", "data/SeCVOS/Annotations"
# (panel label, run folder)
RUNS = [("E0: original SeC (baseline)", f"{E}/fps160/E0"),
        ("E10a: denser concept calls", f"{E}/fps160/E10a"),
        ("E17a: + concept-only mask when lost (best)", f"{E}/e17/E17a"),
        ("E18: E17a, accepted mask not put in bank", f"{E}/e18/E18a_pixel")]
LEGEND = ["Mask = orange.",
          "White dot    = concept (MLLM) call.",
          "Red border   = tracker lost.",
          "Green border = concept-only mask accepted.",
          "",
          "E10a = E0 + more concept calls after a scene",
          "       change (burst 3) and every 15 frames.",
          "E17a/E18 = E10a + when lost, the decoder is also",
          "       run on the concept branch alone; a mask",
          "       passing the gate replaces the output.",
          "E17a writes it to output + SAM 2 pixel memory",
          "       + MLLM keyframe bank;",
          "E18 ('pixel') to output + pixel memory only."]
LFONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)
OUT_V, OUT_K = "deliverable/videos", "deliverable/keyframes"
COLOR = np.array([255, 140, 0], dtype=np.float32)  # mask colour (orange), alpha 0.5
FONT = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", 15)

# name: (video, object, first frame, last frame, keyframes, render video?)
CLIPS = {
    "1_recovery_09trCkd99Os": ("09trCkd99Os", "000", 0, 127, [18, 25, 27, 82], True),
    "2_bank_lock_aP2up9N6H-g": ("aP2up9N6H-g", "000", 0, 150, [5, 12, 30, 72, 145], True),
    "3_invented_E_UbFKdWFl4": ("E_UbFKdWFl4", "000", 71, 160, [128, 131, 138, 148], True),
    "4_wrong_object_CKmkgzeNKZw": ("CKmkgzeNKZw", "000", 0, 118, [50, 58, 66, 74], True),
    "5_cut_both_fail_GMnti60Gxgc": ("GMnti60Gxgc", "000", 0, 108, [45, 48, 60, 90], True),
    "6_harm_cIX8usnGj7s": ("cIX8usnGj7s", "000", 0, 162, [30, 34, 52, 60, 125], True),
    # 7-10: next 4 largest E17a - E10a gains on rest120 (objects without a video yet)
    "7_gain_ccj3faULpKw": ("ccj3faULpKw", "000", 0, 141, [45, 56, 105, 109, 112], True),
    "8_gain_H7hkYP9CeSk": ("H7hkYP9CeSk", "000", 0, 178, [50, 54, 70, 114, 150], True),
    "9_gain_aYO2eRIFjQs": ("aYO2eRIFjQs", "000", 0, 167, [4, 40, 105, 139], True),
    "10_gain_cw5P5iAJ3Xw": ("cw5P5iAJ3Xw", "000", 0, 140, [45, 51, 55, 100], True),
}


def mask(path):
    return np.array(Image.open(path)) > 0 if os.path.exists(path) else None


def lost_flags(log):
    """lost[f]: mask empty or logit <= 0 on the 2 previous tracked (non-start) frames."""
    bad = {int(r.frame): (r.mask_px == 0 or r.object_score_logit <= 0) for r in log.itertuples() if r.is_cond == 0}
    return {f: bad.get(f - 1, False) and bad.get(f - 2, False) for f in log.frame.astype(int)}


def panel(img, m, label, j, call=False, lost=False, accept=False):
    a = img.astype(np.float32)
    if m is not None and m.any():
        a[m] = 0.5 * a[m] + 0.5 * COLOR
    p = Image.fromarray(a.astype(np.uint8))
    d = ImageDraw.Draw(p)
    w, h = p.size
    if accept:
        d.rectangle([0, 0, w - 1, h - 1], outline=(0, 220, 0), width=7)
    elif lost:
        d.rectangle([0, 0, w - 1, h - 1], outline=(230, 0, 0), width=7)
    for k, txt in enumerate(label.split("\n")):  # line 1: configuration; line 2: frame, J
        d.rectangle([8, 8 + 22 * k, 16 + d.textlength(txt, font=FONT), 30 + 22 * k], fill=(0, 0, 0))
        d.text((12, 10 + 22 * k), txt, fill=(255, 255, 255), font=FONT)
    if call:
        d.ellipse([w - 26, 12, w - 12, 26], fill=(255, 255, 255), outline=(0, 0, 0))
    return p


def jac(g, m):
    u = (g | m).sum()
    return 1.0 if u == 0 else float((g & m).sum() / u)


def render(name, v, o, a, b, keys, video):
    logs = {r: pd.read_csv(f"{d}/frames/{v}.csv", dtype={"obj": str}) for r, d in RUNS}
    logs = {r: l[l.obj == o].set_index("frame", drop=False) for r, l in logs.items()}
    lost = {r: lost_flags(l) for r, l in logs.items()}
    tmp = f"{OUT_V}/_tmp_{name}"
    os.makedirs(tmp, exist_ok=True)
    os.makedirs(OUT_K, exist_ok=True)
    frames = range(a, b + 1)
    for i, f in enumerate(frames):
        img = np.array(Image.open(f"{IMG}/{v}/{f:05d}.jpg").convert("RGB"))
        g = mask(f"{ANN}/{v}/{o}/{f:05d}.png")
        ps = [panel(img, g, f"GT (ground truth)\nf{f}" + ("  object not visible" if not g.any() else ""), None)]
        for r, d in RUNS:
            m = mask(f"{d}/masks/{v}/{o}/{f:05d}.png")
            row = logs[r].loc[f] if f in logs[r].index else None
            call = row is not None and row.triggered == 1 and row.is_cond == 0
            acc = row is not None and "e17_accept" in row and row.e17_accept == 1
            ps.append(panel(img, m, f"{r}\nf{f}  J={jac(g, m):.2f}", None, call, lost[r].get(f, False), acc))
        W, H = ps[0].size
        canvas = Image.new("RGB", (3 * W + 8, 2 * H + 4), (255, 255, 255))
        for k, p in enumerate(ps):
            canvas.paste(p, ((k % 3) * (W + 4), (k // 3) * (H + 4)))
        d = ImageDraw.Draw(canvas)  # legend in the 6th cell
        for li, line in enumerate(LEGEND):
            d.text((2 * (W + 4) + 12, H + 16 + 22 * li), line, fill=(0, 0, 0), font=LFONT)
        if f in keys:
            canvas.save(f"{OUT_K}/{name}_f{f:03d}.png", optimize=True)
        if video:
            s = 1920 / canvas.size[0]
            canvas.resize((1920, int(round(canvas.size[1] * s / 2)) * 2), Image.LANCZOS).save(f"{tmp}/{i:05d}.png")
    if video:
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "6", "-i", f"{tmp}/%05d.png",
                        "-c:v", "libx264", "-preset", "veryfast", "-crf", "23", "-pix_fmt", "yuv420p",
                        f"{OUT_V}/{name}.mp4"], check=True)
    for fn in os.listdir(tmp):
        os.remove(f"{tmp}/{fn}")
    os.rmdir(tmp)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    args = ap.parse_args()
    for n, c in CLIPS.items():
        if args.only in (None, n):
            render(n, *c)
            print("done", n)
