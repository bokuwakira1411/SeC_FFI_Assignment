"""Scene cuts (PySceneDetect ContentDetector, min_scene_len=5, as experiments/scenecuts_msl5.json) for all 160 videos (CPU).

Usage (from SeC/): python experiments/e17/detect_cuts_msl5.py
subset40 cuts are copied from experiments/scenecuts_msl5.json (not recomputed); the other videos are detected.
Writes experiments/e17/scenecuts_msl5_all160.json.
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

from scenedetect import ContentDetector, detect


def cuts(video):
    pattern = os.path.join("data/SeCVOS/JPEGImages", video, "%05d.jpg")
    scenes = detect(pattern, ContentDetector(min_scene_len=5))
    return video, [s[0].get_frames() for s in scenes[1:]]


if __name__ == "__main__":
    old = json.load(open("experiments/scenecuts_msl5.json"))
    videos = [l.strip() for l in open("experiments/e17/all160.txt") if l.strip()]
    todo = [v for v in videos if v not in old]
    with ProcessPoolExecutor(4) as ex:
        new = dict(ex.map(cuts, todo))
    res = {v: old[v] if v in old else new[v] for v in videos}
    json.dump(res, open("experiments/e17/scenecuts_msl5_all160.json", "w"), indent=1)
    print("videos", len(res), "detected", len(new), "cuts", sum(map(len, res.values())))
