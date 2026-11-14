"""List SeCVOS videos that still need inference (and clean up half-written outputs).

A video counts as done when its output folder has at least one object folder
and every object folder holds one PNG per input frame.

Usage (from SeC/):
  python make_rest_list.py            # writes data/SeCVOS/video_id_rest.txt
"""
import argparse
import os
import shutil

ap = argparse.ArgumentParser()
ap.add_argument("--list", default="data/SeCVOS/video_id.txt")
ap.add_argument("--frames", default="data/SeCVOS/JPEGImages")
ap.add_argument("--pred", default="outputs/secvos_full")
ap.add_argument("--out", default="data/SeCVOS/video_id_rest.txt")
a = ap.parse_args()

videos = [v.strip() for v in open(a.list) if v.strip()]
done, rest, cleaned = [], [], []
for v in videos:
    n_frames = sum(f.lower().endswith((".jpg", ".jpeg"))
                   for f in os.listdir(os.path.join(a.frames, v)))
    vdir = os.path.join(a.pred, v)
    objs = [o for o in os.listdir(vdir)] if os.path.isdir(vdir) else []
    ok = bool(objs) and all(
        len([f for f in os.listdir(os.path.join(vdir, o)) if f.endswith(".png")]) == n_frames
        for o in objs)
    if ok:
        done.append(v)
    else:
        if os.path.isdir(vdir):
            shutil.rmtree(vdir)
            cleaned.append(v)
        rest.append(v)

with open(a.out, "w") as f:
    f.write("\n".join(rest) + ("\n" if rest else ""))
print(f"done: {len(done)}/{len(videos)} | remaining: {len(rest)} -> {a.out}")
if cleaned:
    print("removed incomplete outputs:", cleaned)
