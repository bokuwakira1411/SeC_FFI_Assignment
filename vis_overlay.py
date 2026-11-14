"""Overlay SeC predicted masks (and optionally GT) on video frames, save as MP4.

Usage (from SeC/):
  python vis_overlay.py --video -klPo9pilc4
  python vis_overlay.py --video -klPo9pilc4 --show_gt     # pred | GT side by side
"""
import argparse
import os

import cv2
import numpy as np
from PIL import Image

COLORS = [(255, 64, 64), (64, 200, 255), (80, 255, 80), (255, 200, 0),
          (200, 80, 255), (255, 128, 200)]


def overlay(frame, mask_root, fname, alpha=0.5):
    out = frame.copy()
    if not os.path.isdir(mask_root):
        return out
    for i, obj in enumerate(sorted(os.listdir(mask_root))):
        p = os.path.join(mask_root, obj, fname + ".png")
        if not os.path.exists(p):
            continue
        m = np.array(Image.open(p)) > 0
        c = np.array(COLORS[i % len(COLORS)], dtype=np.float32)
        out[m] = (1 - alpha) * out[m] + alpha * c
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(out, cnts, -1, tuple(int(x) for x in c), 2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--frames", default="data/SeCVOS/JPEGImages")
    ap.add_argument("--pred", default="outputs/secvos_full")
    ap.add_argument("--gt", default="data/SeCVOS/Annotations")
    ap.add_argument("--show_gt", action="store_true")
    ap.add_argument("--out_dir", default="outputs/vis")
    ap.add_argument("--fps", type=int, default=6)
    a = ap.parse_args()

    fdir = os.path.join(a.frames, a.video)
    names = sorted(os.path.splitext(f)[0] for f in os.listdir(fdir)
                   if f.lower().endswith((".jpg", ".jpeg")))
    os.makedirs(a.out_dir, exist_ok=True)
    out_path = os.path.join(a.out_dir, f"{a.video}.mp4")
    writer = None
    for n in names:
        img = np.array(Image.open(os.path.join(fdir, n + ".jpg")).convert("RGB")).astype(np.float32)
        vis = overlay(img, os.path.join(a.pred, a.video), n)
        cv2.putText(vis, f"pred {n}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        if a.show_gt:
            g = overlay(img, os.path.join(a.gt, a.video), n)
            cv2.putText(g, "GT", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            vis = np.concatenate([vis, g], axis=1)
        vis = cv2.cvtColor(vis.clip(0, 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
        if writer is None:
            h, w = vis.shape[:2]
            writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
        writer.write(vis)
    writer.release()
    print("saved", out_path)


if __name__ == "__main__":
    main()
