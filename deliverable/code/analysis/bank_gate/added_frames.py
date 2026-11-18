"""Per added-bank-frame J vs GT + candidate gate signals. Run from SeC/."""
import csv, glob, json, collections, sys
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
RUNS = {"E0r": "experiments/E0r", "E10a": "experiments/E10a", "E12": "experiments/bank/E12", "E14": "experiments/bank/E14"}
GT = "experiments/gt_subset40"
CUTS = json.load(open("experiments/scenecuts_msl5.json"))

def m(path):
    try: return np.array(Image.open(path)) > 0
    except FileNotFoundError: return None

def iou(a, b):
    if a is None or b is None: return np.nan
    u = (a | b).sum(); return 1.0 if u == 0 else (a & b).sum() / u

def job(args):
    run, p = args
    rows = list(csv.DictReader(open(p)))
    by = collections.defaultdict(dict)
    for r in rows: by[r["obj"]][int(r["frame"])] = r
    out = []
    for o, fr in by.items():
        for f, r in fr.items():
            if r["is_cond"] != "0" or not r["bank_reason"].startswith("added"): continue
            v = r["video"]; d = f"{RUNS[run]}/masks/{v}/{o}"
            pm = m(f"{d}/{f:05d}.png"); g = m(f"{GT}/{v}/{o}/{f:05d}.png")
            prev = [float(fr[k]["mask_px"]) for k in range(f - 5, f) if k in fr]
            area = float(r["mask_px"]); ar = area / np.mean(prev) if prev and np.mean(prev) > 0 else np.nan
            st = [iou(pm, m(f"{d}/{f+k:05d}.png")) for k in (1, 2, 3)]
            cuts = CUTS.get(v, [])
            dc = min([abs(f - c) for c in cuts], default=999)
            since = min([f - c for c in cuts if c <= f], default=f)
            out.append(dict(run=run, video=v, obj=o, frame=f, reason=r["bank_reason"], J=iou(pm, g),
                logit=float(r["object_score_logit"]), pred_iou=float(r["pred_iou"] or "nan"),
                area_ratio=ar, log_area_dev=abs(np.log(ar)) if ar and ar > 0 else np.nan,
                stab1=st[0], stab2=st[1], stab3=st[2], stab_min=np.nanmin(st) if not all(np.isnan(st)) else np.nan,
                cut_dist=dc, since_cut=since, hsv_dist=float(r.get("hsv_dist") or "nan"),
                div_maxcos=float(r.get("div_maxcos") or "nan"), area_px=area))
    return out

if __name__ == "__main__":
    jobs = [(run, p) for run, d in RUNS.items() for p in sorted(glob.glob(f"{d}/frames/*.csv"))]
    with ProcessPoolExecutor(8) as ex: res = [x for l in ex.map(job, jobs) for x in l]
    with open(sys.argv[1], "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(res[0])); w.writeheader(); w.writerows(res)
    print(len(res))
