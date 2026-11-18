"""I1 diagnostic (CPU): where does the teacher-vector variance come from? Relative MSE (vs train-mean constant) of
oracle baselines on val: leave-one-out object mean; object mean split by 'object visible in current frame'."""
import json, torch
R = torch.load("experiments/i1_distill/data/teacher.pt")
tr = [r for r in R if r["split"] == "train"]; va = [r for r in R if r["split"] == "val"]
mu = torch.stack([r["vec"] for r in tr]).mean(0)
masks = {}
def vis(r):
    if r["video"] not in masks:
        masks[r["video"]] = torch.load(f"experiments/i1_distill/data/masks/{r['video']}.pt")
    return bool(masks[r["video"]][r["obj"]][r["cur"]].float().sum() > 0)
def rel(pred_fn, rs):
    V = torch.stack([r["vec"] for r in rs]); P = torch.stack([pred_fn(r) for r in rs])
    return float(((P - V) ** 2).mean() / ((mu - V) ** 2).mean())
out = {}
for name, rs in (("val", va), ("train", tr)):
    groups = {}
    for r in rs:
        groups.setdefault((r["video"], r["obj"]), []).append(r)
    def loo(r, key=lambda x: (x["video"], x["obj"])):
        g = [x["vec"] for x in rs if key(x) == key(r) and x is not r]
        return torch.stack(g).mean(0) if g else mu
    out[name] = dict(
        obj_mean_loo=rel(loo, rs),
        obj_x_curvisible_loo=rel(lambda r: loo(r, key=lambda x: (x["video"], x["obj"], vis(x))), rs),
        curvisible_only=rel(lambda r: loo(r, key=lambda x: vis(x)), rs),
        nkeys_only=rel(lambda r: loo(r, key=lambda x: len(x["keys"])), rs),
        frac_cur_visible=sum(vis(r) for r in rs) / len(rs),
    )
json.dump(out, open("experiments/i1_distill/variance_check.json", "w"), indent=1)
print(json.dumps(out, indent=1))
