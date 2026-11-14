"""Independent scene-cut detection with PySceneDetect (ContentDetector, default params) on JPEG sequences.

Writes experiments/scenecuts.json: {video: [frame index where a new scene starts, ...]} (frame 0 excluded).
"""
import json, os, sys
from concurrent.futures import ProcessPoolExecutor
from scenedetect import ContentDetector, detect

def cuts(video):
    pattern = os.path.join("data/SeCVOS/JPEGImages", video, "%05d.jpg")
    scenes = detect(pattern, ContentDetector())
    return video, [s[0].get_frames() for s in scenes[1:]]

if __name__ == "__main__":
    videos = [l.strip() for l in open(sys.argv[1]) if l.strip()]
    with ProcessPoolExecutor(4) as ex:
        res = dict(ex.map(cuts, videos))
    json.dump(res, open("experiments/scenecuts.json", "w"), indent=1)
    print("videos", len(res), "cuts", sum(map(len, res.values())),
          "scenes/video %.2f" % (sum(len(c) + 1 for c in res.values()) / len(res)))
