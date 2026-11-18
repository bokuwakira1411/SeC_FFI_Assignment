"""I1 step 1: test = subset40; the other 120 SeCVOS videos -> 90% train / 10% val (by video, seed 0)."""
import json, os, random
root = os.path.join(os.path.dirname(__file__), "..", "..")
test = [v.strip() for v in open(os.path.join(root, "experiments/subset40.txt")) if v.strip()]
rest = sorted(v for v in os.listdir(os.path.join(root, "data/SeCVOS/JPEGImages")) if v not in set(test))
random.Random(0).shuffle(rest)
n_val = round(0.1 * len(rest))
split = {"train": sorted(rest[n_val:]), "val": sorted(rest[:n_val]), "test": test}
json.dump(split, open(os.path.join(os.path.dirname(__file__), "split.json"), "w"), indent=1)
print({k: len(v) for k, v in split.items()})
