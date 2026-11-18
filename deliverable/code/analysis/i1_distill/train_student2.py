"""I1 fixes: train ObjStudent (--mode obj, data/teacher.pt) or the hybrid student (--mode hybrid, data/hybrid.pt).

Same loss / optimiser / schedule / selection as train_student.py: (1 - cos) + lambda * MSE in standardised space,
AdamW lr 3e-4 wd 0.05, batch 64, bf16 autocast, warmup 100 + cosine, best val loss, seed 0.
Hybrid baselines reported too: "copy anchor" (reuse the last LLM vector) besides the train-mean constant.
Usage (from SeC/): python experiments/i1_distill/train_student2.py --mode obj|hybrid [--epochs 50]
Outputs: experiments/i1_distill/student_{mode}.pt, train_log_{mode}.csv, val_report_{mode}.json
"""
import argparse, json, math, os, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from experiments.i1_distill.student import slots_for
from experiments.i1_distill.student2 import ObjStudent, N_KEY

D = "experiments/i1_distill"
DATA = f"{D}/data"


def build(records, fidx, masks, hybrid):
    n = len(records)
    krow = torch.zeros(n, N_KEY, dtype=torch.long)
    km = torch.zeros(n, N_KEY, 16, 16, dtype=torch.float16)
    ks = torch.zeros(n, N_KEY, dtype=torch.long)
    kv = torch.zeros(n, N_KEY, dtype=torch.bool)
    crow = torch.zeros(n, dtype=torch.long)
    arow = torch.zeros(n, dtype=torch.long)
    for i, r in enumerate(records):
        b = fidx[r["video"]]
        s = slots_for(len(r["keys"]))[:-1]
        for p, (f, sl) in enumerate(zip(r["keys"], s)):
            krow[i, p], ks[i, p], kv[i, p] = b + f, sl, True
            km[i, p] = masks[r["video"]][r["obj"]][f]
        crow[i] = b + r["cur"]
        if hybrid:
            arow[i] = b + r["anc"]
    v = torch.stack([r["vec"] for r in records])
    av = torch.stack([r["anc_vec"] for r in records]) if hybrid else torch.zeros_like(v)
    return [krow, km, ks, kv, crow, arow, av, v]


def metrics(v_hat, v, mu, anc=None):
    def one(p):
        return dict(cos=float(F.cosine_similarity(p, v, dim=1).mean()), mse=float(((p - v) ** 2).mean()))
    s, c = one(v_hat), one(mu.expand_as(v))
    out = dict(cos_student=s["cos"], cos_const=c["cos"], rel_mse=s["mse"] / c["mse"],
               cos_student_p10=float(F.cosine_similarity(v_hat, v, dim=1).quantile(0.1)),
               cos_centered_student=float(F.cosine_similarity(v_hat - mu, v - mu, dim=1).mean()))
    if anc is not None:
        a = one(anc)
        out.update(cos_copy_anchor=a["cos"], rel_mse_copy_anchor=a["mse"] / c["mse"],
                   rel_mse_vs_copy_anchor=s["mse"] / a["mse"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["obj", "hybrid"], required=True)
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    args = ap.parse_args()
    hyb = args.mode == "hybrid"
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    dev = "cuda"

    records = torch.load(f"{DATA}/{'hybrid' if hyb else 'teacher'}.pt")
    videos = sorted({r["video"] for r in records})
    fidx, feats, masks, off = {}, [], {}, 0
    for v in videos:
        x = torch.load(f"{DATA}/feats/{v}.pt")
        fidx[v] = off; off += len(x); feats.append(x)
        masks[v] = torch.load(f"{DATA}/masks/{v}.pt")
    FEAT = torch.cat(feats).to(dev)
    tr = [r for r in records if r["split"] == "train"]
    va = [r for r in records if r["split"] == "val"]
    Tr = [t.to(dev) for t in build(tr, fidx, masks, hyb)]
    Va = [t.to(dev) for t in build(va, fidx, masks, hyb)]
    allv = torch.cat([Tr[7], Tr[6]]) if hyb else Tr[7]
    mu, sigma = allv.mean(0), allv.std(0).clamp_min(1e-6)
    lam = float((1 - F.cosine_similarity(mu.expand_as(Tr[7]), Tr[7], dim=1)).mean())
    print(f"[{args.mode}] train {len(tr)} samples / {len({(r['video'], r['obj']) for r in tr})} objects, "
          f"val {len(va)} / {len({(r['video'], r['obj']) for r in va})}; lambda {lam:.4f}")

    model = ObjStudent(hybrid=hyb).to(dev)
    model.mu.copy_(mu); model.sigma.copy_(sigma)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"student params {n_param / 1e6:.3f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total = math.ceil(len(tr) / args.bs) * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    def run(T, idx):
        krow, km, ks, kv, crow, arow, av, v = (t[idx] for t in T)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_hat = model(FEAT[krow], km, ks, kv, FEAT[crow],
                          FEAT[arow] if hyb else None, ((av - mu) / sigma) if hyb else None)
        z_hat = z_hat.float()
        v_hat = z_hat * sigma + mu
        cos_l = (1 - F.cosine_similarity(v_hat, v, dim=1)).mean()
        mse_l = F.mse_loss(z_hat, (v - mu) / sigma)
        return cos_l + lam * mse_l, v_hat

    @torch.no_grad()
    def evaluate(T):
        model.eval()
        out, ls = [], 0.0
        for i in range(0, len(T[0]), 256):
            idx = torch.arange(i, min(i + 256, len(T[0])), device=dev)
            l, vh = run(T, idx)
            out.append(vh); ls += float(l) * len(idx)
        model.train()
        return ls / len(T[0]), torch.cat(out)

    log = open(f"{D}/train_log_{args.mode}.csv", "w")
    log.write("epoch,train_loss,val_loss,val_cos,val_rel_mse,sec\n")
    best, g, t0 = float("inf"), torch.Generator().manual_seed(0), time.time()
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(len(tr), generator=g).to(dev)
        tl = 0.0
        for i in range(0, len(tr), args.bs):
            idx = perm[i:i + args.bs]
            loss, _ = run(Tr, idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += float(loss) * len(idx)
        vl, vh = evaluate(Va)
        mt = metrics(vh, Va[7], mu)
        log.write(f"{ep},{tl / len(tr):.5f},{vl:.5f},{mt['cos_student']:.5f},{mt['rel_mse']:.5f},"
                  f"{time.time() - t0:.0f}\n"); log.flush()
        print(f"ep {ep} train {tl / len(tr):.4f} val {vl:.4f} cos {mt['cos_student']:.4f} "
              f"relMSE {mt['rel_mse']:.4f}", flush=True)
        if vl < best:
            best = vl
            torch.save(dict(state_dict=model.state_dict(), epoch=ep, val_loss=vl, lam=lam, args=vars(args)),
                       f"{D}/student_{args.mode}.pt")
    ck = torch.load(f"{D}/student_{args.mode}.pt")
    model.load_state_dict(ck["state_dict"])
    _, vh = evaluate(Va)
    _, th = evaluate(Tr)
    rep = dict(best_epoch=ck["epoch"], val_loss=ck["val_loss"], lam=lam, params_M=n_param / 1e6,
               train_sec=time.time() - t0, n_train=len(tr), n_val=len(va),
               val=metrics(vh, Va[7], mu, Va[6] if hyb else None),
               train=metrics(th, Tr[7], mu, Tr[6] if hyb else None))
    rep["pass"] = rep["val"]["cos_student"] >= 0.8 and rep["val"]["rel_mse"] <= 0.5
    json.dump(rep, open(f"{D}/val_report_{args.mode}.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
