"""I1 fix (3): teacher pairs for the hybrid student (LLM on HSV triggers, student on the other frames).

Per object N_BANKS draws: anchor frame a = start+1 (first tracked frame, 1/3), an HSV-trigger frame (1/3) or a uniform
frame after the start (1/3); keyframes = start + k (0..6) GT-non-empty frames in (start, a); targets = N_TGT uniform
frames in (a, n-1]. With bank_rule hsv the bank does not change between triggers, so anchor and targets use the same
keyframes. Teacher calls exactly as collect_teacher.py (GT contours, same prompt, bf16 autocast).
Output: data/hybrid.pt, list of {video, obj, split, start, keys, anc, anc_vec, cur, vec}.
Features / masks / hsv are reused from collect_teacher.py.
Usage (from SeC/): python experiments/i1_distill/collect_hybrid.py [--n_banks 10 --n_tgt 4]
"""
import argparse, json, os, random, sys, time

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
from experiments.i1_distill.collect_teacher import ANN, OUT, frame_paths, load_gt, teacher


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # as vos_separate_inference_per_object
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_banks", type=int, default=10)
    ap.add_argument("--n_tgt", type=int, default=4)
    args = ap.parse_args()
    split = json.load(open("experiments/i1_distill/split.json"))
    videos = [(v, s) for s in ("train", "val") for v in split[s]]

    from inference.configuration_sec import SeCConfig
    from inference.modeling_sec import SeCModel
    from transformers import AutoTokenizer
    mp = "saved_models/SeC-4B"
    config = SeCConfig.from_pretrained(mp)
    config.hydra_overrides_extra = ["++model.non_overlap_masks=false"]
    model = SeCModel.from_pretrained(mp, config=config, torch_dtype=torch.bfloat16, use_flash_attn=True).eval().to("cuda")
    model.preparing_for_generation(tokenizer=AutoTokenizer.from_pretrained(mp, trust_remote_code=True))

    path = os.path.join(OUT, "hybrid.pt")
    records = torch.load(path) if os.path.exists(path) else []
    done = {r["video"] for r in records}
    for i, (video, sp) in enumerate(videos):
        if video in done:
            continue
        tv = time.time()
        ps = frame_paths(video)
        n = len(ps)
        imgs = [Image.open(p).convert("RGB") for p in ps]
        hsv = np.load(os.path.join(OUT, "hsv", f"{video}.npy"))
        new = []
        for obj in sorted(os.listdir(os.path.join(ANN, video))):
            gts = load_gt(video, obj, n)
            nonempty = [bool(m.any()) for m in gts]
            if not any(nonempty):
                continue
            start = nonempty.index(True)
            if start >= n - 2:
                continue
            rng = random.Random(f"hybrid_{video}_{obj}")
            trig = [f for f in range(start + 1, n - 1) if hsv[f] > 0.35]
            for _ in range(args.n_banks):
                u = rng.random()
                if u < 1 / 3:
                    anc = start + 1
                elif u < 2 / 3 and trig:
                    anc = rng.choice(trig)
                else:
                    anc = rng.randint(start + 1, n - 2)
                cand = [f for f in range(start + 1, anc) if nonempty[f]]
                keys = [start] + sorted(rng.sample(cand, rng.randint(0, min(6, len(cand)))))
                anc_vec = teacher(model, imgs, gts, keys, anc)
                for cur in sorted(rng.sample(range(anc + 1, n), min(args.n_tgt, n - 1 - anc))):
                    new.append(dict(video=video, obj=obj, split=sp, start=start, keys=keys, anc=anc,
                                    anc_vec=anc_vec, cur=cur, vec=teacher(model, imgs, gts, keys, cur)))
        records += new
        torch.save(records, path)
        print(f"[{i + 1}/{len(videos)}] {video} {sp} pairs={len(new)} {time.time() - tv:.0f}s", flush=True)


if __name__ == "__main__":
    main()
