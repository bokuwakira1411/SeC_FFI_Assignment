"""Choose tau for E14 (--bank_diverse_tau) without J&F (CPU).

Usage (from SeC/): python experiments/bank/sim_diverse.py [TARGET=480]
Replays the E14 bank rule (inference/modeling_sec.py, bank_diverse_tau) on E10a's logs (same concept
schedule as E14): candidate = tracked frame with mask_px > 0 and object_score_logit > 1, if HSV trigger or
>= 5 frames after the last candidate; added if max cosine to the bank < tau; bank of 7 keeps the start frame
and drops the frame most similar to the candidate. Descriptors: experiments/bank/desc/E10a (pred masks),
start frame from desc/GT (init mask = GT). Approximation: E14's own masks differ from E10a's.
Picks the tau whose total number of added frames is closest to TARGET (E6: 480).
Writes experiments/bank/tau.json.
"""
import csv
import glob
import json
import os
import sys

import numpy as np
import torch

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 480
GAP, SIZE, SCORE = 5, 7, 1.0
TAUS = np.round(np.arange(0.50, 1.0001, 0.01), 2)


def load_objects():
    objs = []
    for p in sorted(glob.glob("experiments/E10a/frames/*.csv")):
        rows = list(csv.DictReader(open(p)))
        for o in sorted({r["obj"] for r in rows}):
            rs = sorted((r for r in rows if r["obj"] == o), key=lambda r: int(r["frame"]))
            v = rs[0]["video"]
            pred = torch.load(f"experiments/bank/desc/E10a/{v}_{o}.pt").float().numpy()
            gt = torch.load(f"experiments/bank/desc/GT/{v}_{o}.pt").float().numpy()
            start = int(rs[0]["frame"])
            assert rs[0]["is_cond"] == "1"
            cands = []  # (frame, hsv_trigger, descriptor) of trusted tracked frames
            for r in rs[1:]:
                if int(r["mask_px"]) > 0 and float(r["object_score_logit"]) > SCORE:
                    cands.append((int(r["frame"]), r["hsv_trigger"] == "1", pred[int(r["frame"])]))
            objs.append((v, o, start, gt[start], cands))
    return objs


def simulate(obj, tau):
    _, _, start, d0, cands = obj
    bank, last, added, maxcos = [d0], start, 0, []
    for f, hsv, d in cands:
        if not (hsv or f - last >= GAP):
            continue
        last = f
        cos = [float(d @ e) for e in bank]
        maxcos.append(max(cos))
        if max(cos) < tau:
            if len(bank) >= SIZE:
                del bank[1 + int(np.argmax(cos[1:]))]
            bank.append(d)
            added += 1
    return added, maxcos


def main():
    objs = load_objects()
    totals = {float(t): sum(simulate(o, t)[0] for o in objs) for t in TAUS}
    tau = min(totals, key=lambda t: (abs(totals[t] - TARGET), t))
    mc = np.concatenate([simulate(o, 2.0)[1] for o in objs])  # tau > 1: every candidate added
    out = dict(target=TARGET, tau=tau, sim_added=totals[tau], n_objects=len(objs), grid=totals,
               candidate_maxcos_quantiles_all_added={q: float(np.quantile(mc, q)) for q in (.1, .25, .5, .75, .9)})
    json.dump(out, open("experiments/bank/tau.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in out.items() if k != "grid"}, indent=1))
    for t in TAUS[::5]:
        print(f"tau {t:.2f}: added {totals[float(t)]}")


if __name__ == "__main__":
    main()
