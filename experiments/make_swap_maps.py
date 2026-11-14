"""Build fixed concept-swap maps from E0's saved concepts (seed 0).

E2  (swap_cross.json): each object -> a random object of a DIFFERENT video.
E2b (swap_same.json):  each object of a video with >= 2 objects -> a random OTHER object of the SAME video.
Value = that source object's E0 concept files sorted by frame; the k-th trigger uses entry k mod n.
Also writes the video lists / GT symlink dirs for E2b (multi-object videos) and E4/E4b (10 shortest).
"""
import collections, csv, glob, json, os, random

E0 = "experiments/E0"
concepts = collections.defaultdict(list)
for f in sorted(glob.glob(f"{E0}/concepts/*.pt")):
    key = os.path.basename(f)[:-len("_00000.pt")]          # <video>_<obj>
    concepts[key].append(f)
videos = [l.strip() for l in open("experiments/subset40.txt") if l.strip()]
objs = sorted((v, o) for v in videos for o in os.listdir(f"data/SeCVOS/Annotations/{v}"))
assert all(f"{v}_{o}" in concepts for v, o in objs), "every object needs >= 1 E0 concept"

rng = random.Random(0)
cross, cross_src = {}, {}
for v, o in objs:
    src = rng.choice([(v2, o2) for v2, o2 in objs if v2 != v])
    cross[f"{v}_{o}"] = concepts[f"{src[0]}_{src[1]}"]
    cross_src[f"{v}_{o}"] = f"{src[0]}_{src[1]}"
json.dump(cross, open("experiments/swap_cross.json", "w"), indent=1)

rng = random.Random(0)
same, same_src = {}, {}
by_video = collections.defaultdict(list)
for v, o in objs:
    by_video[v].append(o)
multi = [v for v in videos if len(by_video[v]) >= 2]
for v in multi:
    for o in by_video[v]:
        o2 = rng.choice([x for x in by_video[v] if x != o])
        same[f"{v}_{o}"] = concepts[f"{v}_{o2}"]
        same_src[f"{v}_{o}"] = f"{v}_{o2}"
json.dump(same, open("experiments/swap_same.json", "w"), indent=1)
json.dump({"cross": cross_src, "same": same_src}, open("experiments/swap_sources.json", "w"), indent=1)

def write_subset(name, vids):
    open(f"experiments/{name}.txt", "w").write("".join(v + "\n" for v in vids))
    gt = f"experiments/gt_{name}"
    os.makedirs(gt, exist_ok=True)
    for v in vids:
        if not os.path.islink(f"{gt}/{v}"):
            os.symlink(os.path.abspath(f"data/SeCVOS/Annotations/{v}"), f"{gt}/{v}")

write_subset("subset_multiobj", multi)
rows = list(csv.DictReader(open(f"{E0}/videos.csv")))
short = [r["video"] for r in sorted(rows, key=lambda r: (int(r["n_frames"]), r["video"]))[:10]]
write_subset("subset_short10", short)
print("objects", len(objs), "| cross map", len(cross), "| same map", len(same), "on", multi)
print("short10", short)
print("same-video sources:", same_src)
