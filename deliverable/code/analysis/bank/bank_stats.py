"""Bank statistics per run (CPU): frames added, objects whose bank stays at 1 frame, bank diversity.

Usage (from SeC/): python experiments/bank/bank_stats.py RUN [RUN ...]   (e.g. E0r E6 E10a bank/E13a)
Needs experiments/bank/desc/<RUN>/ and desc/GT/ (descriptors.py). Writes experiments/bank/bank_stats.csv.
- added: non-cond frames with bank_reason added* (added, added_periodic, added_gt, added_div, added_div_replace).
- stuck: objects whose bank still holds only the start frame on the last frame.
- diversity: on each frame, the bank as given to the MLLM (bank_frames; if > 7 frames, the start frame + the
  last 6, as in propagate_in_video); mean cosine over the pairs of its descriptors; mean over the frames
  with >= 2 bank frames of an object, then over objects (lower = more diverse). Start frame and frames added
  with a GT mask (added_gt) use GT descriptors, other frames the run's predicted mask (= the mask put in the bank).
"""
import csv
import glob
import os
import sys

import numpy as np
import torch

SIZE = 7


def stats(run):
    added = stuck = n_obj = 0
    div = []
    for p in sorted(glob.glob(f"experiments/{run}/frames/*.csv")):
        rows = list(csv.DictReader(open(p)))
        for o in sorted({r["obj"] for r in rows}):
            rs = sorted((r for r in rows if r["obj"] == o), key=lambda r: int(r["frame"]))
            v = rs[0]["video"]
            n_obj += 1
            added += sum(r["is_cond"] == "0" and r["bank_reason"].startswith("added") for r in rs)
            stuck += len(rs[-1]["bank_frames"].split()) == 1
            pred = torch.load(f"experiments/bank/desc/{run}/{v}_{o}.pt").float().numpy()
            gt = torch.load(f"experiments/bank/desc/GT/{v}_{o}.pt").float().numpy()
            gt_frames = {int(rs[0]["frame"])} | {int(r["frame"]) for r in rs if r["bank_reason"] == "added_gt"}
            per_frame = []
            for r in rs:
                b = [int(x) for x in r["bank_frames"].split()]
                if len(b) > SIZE:
                    b = [b[0]] + b[-(SIZE - 1):]
                d = np.stack([gt[f] if f in gt_frames else pred[f] for f in b])
                d = d[~np.isnan(d).any(1)]
                if len(d) < 2:
                    continue
                c = d @ d.T
                per_frame.append(c[np.triu_indices(len(d), 1)].mean())
            if per_frame:
                div.append(np.mean(per_frame))
    return dict(run=run, objects=n_obj, added=added, stuck=stuck,
                diversity=round(float(np.mean(div)), 4), objects_with_pairs=len(div))


def main():
    out = "experiments/bank/bank_stats.csv"
    old = {r["run"]: r for r in csv.DictReader(open(out))} if os.path.exists(out) else {}
    for run in sys.argv[1:]:
        old[run] = stats(run)
        print(old[run])
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["run", "objects", "added", "stuck", "diversity", "objects_with_pairs"])
        w.writeheader()
        w.writerows(old.values())


if __name__ == "__main__":
    main()
