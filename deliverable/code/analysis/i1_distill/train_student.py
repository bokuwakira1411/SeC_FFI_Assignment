"""I1 step 3: train the concept student on the teacher vectors (collect_teacher.py output).

Loss = (1 - cos(v_hat, v)) + lambda * MSE(z_hat, z), z = (v - mu) / sigma with train teacher statistics,
v_hat = z_hat * sigma + mu. lambda = mean(1 - cos(mu, v)) on train, so both terms equal 1-cos / 1 for the constant
(mean vector) baseline. AdamW lr 3e-4, wd 0.05, batch 64, bf16 autocast, warmup 100 steps + cosine decay,
checkpoint chosen by val loss. Seed 0.
Usage (from SeC/): python experiments/i1_distill/train_student.py [--epochs 50]
Outputs: experiments/i1_distill/student_best.pt, train_log.csv, val_report.json
"""
import argparse, json, math, os, random, sys, time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.getcwd())
from experiments.i1_distill.student import ConceptStudent, N_KEY, slots_for

D = "experiments/i1_distill"
DATA = f"{D}/data"


def build(records, feat_index, masks):
    """Per sample: frame rows [8] (7 keyframe slots + current), masks [8,16,16], slots [8], valid [8], target."""
    n = len(records)
    rows = torch.zeros(n, N_KEY + 1, dtype=torch.long)
    m = torch.zeros(n, N_KEY + 1, 16, 16, dtype=torch.float16)
    slots = torch.zeros(n, N_KEY + 1, dtype=torch.long)
    valid = torch.zeros(n, N_KEY + 1, dtype=torch.bool)
    for i, r in enumerate(records):
        base = feat_index[r["video"]]
        frames = r["keys"] + [r["cur"]]
        s = slots_for(len(r["keys"]))
        pos = list(range(len(r["keys"]))) + [N_KEY]
        for p, f, sl in zip(pos, frames, s):
            rows[i, p] = base + f
            slots[i, p] = sl
            valid[i, p] = True
        for p, f in zip(pos[:-1], r["keys"]):
            m[i, p] = masks[r["video"]][r["obj"]][f]
    v = torch.stack([r["vec"] for r in records])
    return rows, m, slots, valid, v


def metrics(v_hat, v, mu):
    const = mu.expand_as(v)
    cos_s = F.cosine_similarity(v_hat, v, dim=1)
    cos_c = F.cosine_similarity(const, v, dim=1)
    mse_s = ((v_hat - v) ** 2).mean()
    mse_c = ((const - v) ** 2).mean()
    cc = F.cosine_similarity(v_hat - mu, v - mu, dim=1)
    return dict(cos_student=float(cos_s.mean()), cos_const=float(cos_c.mean()),
                cos_student_p10=float(cos_s.quantile(0.1)), cos_student_min=float(cos_s.min()),
                rel_mse=float(mse_s / mse_c), mse_student=float(mse_s), mse_const=float(mse_c),
                cos_centered_student=float(cc.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=50)
    ap.add_argument("--bs", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    dev = "cuda"

    records = torch.load(f"{DATA}/teacher.pt")
    videos = sorted({r["video"] for r in records})
    feat_index, feats, masks, off = {}, [], {}, 0
    for v in videos:
        x = torch.load(f"{DATA}/feats/{v}.pt")
        feat_index[v] = off
        off += len(x)
        feats.append(x)
        masks[v] = torch.load(f"{DATA}/masks/{v}.pt")
    FEAT = torch.cat(feats).to(dev)                                     # [N, 256, 16, 16] fp16
    tr = [r for r in records if r["split"] == "train"]
    va = [r for r in records if r["split"] == "val"]
    Tr = [t.to(dev) for t in build(tr, feat_index, masks)]
    Va = [t.to(dev) for t in build(va, feat_index, masks)]
    mu, sigma = Tr[4].mean(0), Tr[4].std(0).clamp_min(1e-6)
    lam = float((1 - F.cosine_similarity(mu.expand_as(Tr[4]), Tr[4], dim=1)).mean())
    print(f"train {len(tr)} samples / {len({(r['video'], r['obj']) for r in tr})} objects, "
          f"val {len(va)} / {len({(r['video'], r['obj']) for r in va})}; frames {len(FEAT)}; lambda {lam:.4f}")

    model = ConceptStudent().to(dev)
    model.mu.copy_(mu); model.sigma.copy_(sigma)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"student params {n_param / 1e6:.2f} M")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = math.ceil(len(tr) / args.bs)
    total = steps_per_epoch * args.epochs
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / 100) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    def run(T, idx):
        rows, m, slots, valid, v = (t[idx] for t in T)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            z_hat = model(FEAT[rows], m, slots, valid)
        z_hat = z_hat.float()
        v_hat = z_hat * sigma + mu
        cos_l = (1 - F.cosine_similarity(v_hat, v, dim=1)).mean()
        mse_l = F.mse_loss(z_hat, (v - mu) / sigma)
        return cos_l + lam * mse_l, cos_l, mse_l, v_hat

    @torch.no_grad()
    def evaluate(T):
        model.eval()
        out, ls = [], []
        for i in range(0, len(T[0]), 256):
            idx = torch.arange(i, min(i + 256, len(T[0])), device=dev)
            l, _, _, vh = run(T, idx)
            out.append(vh); ls.append(float(l) * len(idx))
        model.train()
        return sum(ls) / len(T[0]), torch.cat(out)

    log = open(f"{D}/train_log{args.tag}.csv", "w")
    log.write("epoch,train_loss,train_cos_loss,train_mse_z,val_loss,val_cos,val_rel_mse,sec\n")
    best, g = float("inf"), torch.Generator(device="cpu").manual_seed(0)
    t0 = time.time()
    for ep in range(1, args.epochs + 1):
        perm = torch.randperm(len(tr), generator=g).to(dev)
        tl = tc = tm = 0.0
        for i in range(0, len(tr), args.bs):
            idx = perm[i:i + args.bs]
            loss, cl, ml, _ = run(Tr, idx)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step()
            tl += float(loss) * len(idx); tc += float(cl) * len(idx); tm += float(ml) * len(idx)
        vl, vh = evaluate(Va)
        mt = metrics(vh, Va[4], mu)
        log.write(f"{ep},{tl / len(tr):.5f},{tc / len(tr):.5f},{tm / len(tr):.5f},{vl:.5f},"
                  f"{mt['cos_student']:.5f},{mt['rel_mse']:.5f},{time.time() - t0:.0f}\n"); log.flush()
        print(f"ep {ep} train {tl / len(tr):.4f} val {vl:.4f} cos {mt['cos_student']:.4f} "
              f"relMSE {mt['rel_mse']:.4f}", flush=True)
        if vl < best:
            best = vl
            torch.save(dict(state_dict=model.state_dict(), epoch=ep, val_loss=vl, lam=lam, args=vars(args)),
                       f"{D}/student_best{args.tag}.pt")
    ck = torch.load(f"{D}/student_best{args.tag}.pt")
    model.load_state_dict(ck["state_dict"])
    _, vh = evaluate(Va)
    _, th = evaluate(Tr)
    rep = dict(best_epoch=ck["epoch"], val_loss=ck["val_loss"], lam=lam, params_M=n_param / 1e6,
               train_sec=time.time() - t0, val=metrics(vh, Va[4], mu), train=metrics(th, Tr[4], mu),
               n_train=len(tr), n_val=len(va))
    rep["pass"] = rep["val"]["cos_student"] >= 0.8 and rep["val"]["rel_mse"] <= 0.5
    json.dump(rep, open(f"{D}/val_report{args.tag}.json", "w"), indent=1)
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
