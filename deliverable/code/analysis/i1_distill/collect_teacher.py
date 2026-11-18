"""I1 step 2: teacher concept vectors (LLM, GT masks) + student inputs (Hiera features, GT masks at 16x16).

For every object of the train/val videos (split.json), N_SETS random keyframe sets are built:
  current frame c: 50% a frame with an HSV trigger (hsv_dist > 0.35) after the start frame, otherwise (or if
  there is none) a uniform frame after the start frame; keyframes = start frame + k (uniform 0..6) frames
  with a non-empty GT mask in (start, c), in time order. The teacher input is the same as at inference
  (modeling_sec.py:727-733): keyframes with their mask contour drawn, then the current frame, same prompt.
Outputs (experiments/i1_distill/data/):
  feats/<video>.pt   fp16 [n_frames, 256, 16, 16]  (current_vision_feats[-1], 64x64 avg-pooled 4x4)
  masks/<video>.pt   {obj: fp16 [n_frames, 16, 16]}  (GT mask area fraction, cv2 INTER_AREA)
  hsv/<video>.npy    hsv_dist(frame f, f-1), f >= 1 (index 0 = 0)
  teacher.pt         list of {video, obj, split, start, keys, cur, vec (fp32 [256])}
  checks.json        repeat test, norm stats, cosine between objects
Usage (from SeC/): python experiments/i1_distill/collect_teacher.py [--n_sets 30] [--limit_videos N]
"""
import argparse, json, os, random, sys, time
from multiprocessing import Pool

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.getcwd())
from inference.modeling_sec import hsv_distance, label_img_with_mask

VID = "data/SeCVOS/JPEGImages"
ANN = "data/SeCVOS/Annotations"
OUT = "experiments/i1_distill/data"
PROMPT = "<image>Please segment the object in the last frame based on the object labeled in the first several images."


def frame_paths(video):
    names = sorted(os.listdir(os.path.join(VID, video)), key=lambda p: int(os.path.splitext(p)[0]))
    return [os.path.join(VID, video, n) for n in names]


def video_hsv(video):
    path = os.path.join(OUT, "hsv", f"{video}.npy")
    if os.path.exists(path):
        return
    ps = frame_paths(video)
    imgs = [Image.open(p).convert("RGB") for p in ps]
    d = np.zeros(len(ps), np.float32)
    for f in range(1, len(ps)):
        d[f] = hsv_distance(imgs[f], imgs[f - 1])
    np.save(path, d)


def load_gt(video, obj, n):
    """GT masks of one object: list of bool arrays (missing PNG = empty)."""
    out = []
    for f in range(n):
        p = os.path.join(ANN, video, obj, f"{f:05d}.png")
        out.append(np.array(Image.open(p)) > 0 if os.path.exists(p) else None)
    shape = next(m.shape for m in out if m is not None)
    return [m if m is not None else np.zeros(shape, bool) for m in out]


def mask16(m):
    return cv2.resize(m.astype(np.float32), (16, 16), interpolation=cv2.INTER_AREA)


def sample_sets(rng, gt_nonempty, hsv, start, n_frames, n_sets):
    after = list(range(start + 1, n_frames))
    trig = [f for f in after if hsv[f] > 0.35]
    sets = []
    for _ in range(n_sets):
        cur = rng.choice(trig) if (trig and rng.random() < 0.5) else rng.choice(after)
        cand = [f for f in range(start + 1, cur) if gt_nonempty[f]]
        k = rng.randint(0, min(6, len(cand)))
        sets.append(([start] + sorted(rng.sample(cand, k)), cur))
    return sets


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # as vos_separate_inference_per_object
def teacher(model, imgs, gts, keys, cur):
    video = [label_img_with_mask(imgs[f], gts[f]) for f in keys] + [imgs[cur]]
    return model.predict_forward(video=video, text=PROMPT)[0].float().cpu()  # [256], before unsqueeze(0)


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # as vos_separate_inference_per_object
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n_sets", type=int, default=30)
    ap.add_argument("--limit_videos", type=int, default=0)
    args = ap.parse_args()
    for d in ("feats", "masks", "hsv"):
        os.makedirs(os.path.join(OUT, d), exist_ok=True)
    split = json.load(open("experiments/i1_distill/split.json"))
    videos = [(v, s) for s in ("train", "val") for v in split[s]]
    if args.limit_videos:
        videos = videos[:args.limit_videos]

    t0 = time.time()
    with Pool(16) as p:
        p.map(video_hsv, [v for v, _ in videos])
    print(f"hsv done in {time.time() - t0:.0f}s", flush=True)

    from inference.configuration_sec import SeCConfig
    from inference.modeling_sec import SeCModel
    from transformers import AutoTokenizer
    mp = "saved_models/SeC-4B"
    config = SeCConfig.from_pretrained(mp)
    config.hydra_overrides_extra = ["++model.non_overlap_masks=false"]  # as vos_inference with --per_obj_png_file
    model = SeCModel.from_pretrained(mp, config=config, torch_dtype=torch.bfloat16, use_flash_attn=True).eval().to("cuda")
    tok = AutoTokenizer.from_pretrained(mp, trust_remote_code=True)
    model.preparing_for_generation(tokenizer=tok)
    enc = model.grounding_encoder

    rec_path = os.path.join(OUT, "teacher.pt")
    records = torch.load(rec_path) if os.path.exists(rec_path) else []
    done = {r["video"] for r in records}
    t_llm, n_llm = 0.0, 0
    for i, (video, sp) in enumerate(videos):
        if video in done:
            continue
        tv = time.time()
        ps = frame_paths(video)
        n = len(ps)
        imgs = [Image.open(p).convert("RGB") for p in ps]
        # Hiera features of every frame, same path as tracking (_get_image_feature)
        fpath = os.path.join(OUT, "feats", f"{video}.pt")
        if not os.path.exists(fpath):
            state = enc.init_state(video_path=os.path.join(VID, video), async_loading_frames=False,
                                   offload_video_to_cpu=True)
            feats = torch.empty(n, 256, 16, 16, dtype=torch.float16)
            for f in range(n):
                vf = enc._get_image_feature(state, f, 1)[2][-1]          # [4096, 1, 256]
                x = vf.permute(1, 2, 0).reshape(1, 256, 64, 64).float()
                feats[f] = F.avg_pool2d(x, 4)[0].half().cpu()
            torch.save(feats, fpath)
            del state
            torch.cuda.empty_cache()
        hsv = np.load(os.path.join(OUT, "hsv", f"{video}.npy"))
        objs = sorted(os.listdir(os.path.join(ANN, video)))
        m16 = {}
        new = []
        for obj in objs:
            gts = load_gt(video, obj, n)
            nonempty = [bool(m.any()) for m in gts]
            m16[obj] = torch.from_numpy(np.stack([mask16(m) for m in gts])).half()
            if not any(nonempty):
                continue
            start = nonempty.index(True)
            if start >= n - 1:
                continue
            rng = random.Random(f"{video}_{obj}")
            for keys, cur in sample_sets(rng, nonempty, hsv, start, n, args.n_sets):
                torch.cuda.synchronize(); tt = time.perf_counter()
                vec = teacher(model, imgs, gts, keys, cur)
                torch.cuda.synchronize(); t_llm += time.perf_counter() - tt; n_llm += 1
                new.append(dict(video=video, obj=obj, split=sp, start=start, keys=keys, cur=cur, vec=vec))
        torch.save(m16, os.path.join(OUT, "masks", f"{video}.pt"))
        records += new
        torch.save(records, rec_path)
        print(f"[{i + 1}/{len(videos)}] {video} {sp} frames={n} objs={len(objs)} sets={len(new)} "
              f"{time.time() - tv:.0f}s llm={1000 * t_llm / max(n_llm, 1):.0f}ms/call", flush=True)

    # checks
    r0 = records[0]
    ps = frame_paths(r0["video"])
    imgs = [Image.open(p).convert("RGB") for p in ps]
    gts = load_gt(r0["video"], r0["obj"], len(ps))
    rep = [teacher(model, imgs, gts, r0["keys"], r0["cur"]) for _ in range(2)]
    V = torch.stack([r["vec"] for r in records])
    norms = V.norm(dim=1)
    obj_mean = {}
    for r in records:
        obj_mean.setdefault((r["video"], r["obj"]), []).append(r["vec"])
    M = F.normalize(torch.stack([torch.stack(v).mean(0) for v in obj_mean.values()]), dim=1)
    C = M @ M.T
    off = C[~torch.eye(len(M), dtype=bool)]
    Vn = F.normalize(V, dim=1)
    within = []
    for v in obj_mean.values():
        x = F.normalize(torch.stack(v), dim=1)
        c = x @ x.T
        within.append(c[~torch.eye(len(x), dtype=bool)].mean())
    gm = F.normalize(V.mean(0, keepdim=True), dim=1)
    checks = dict(
        n_records=len(records), n_objects=len(obj_mean), llm_ms_per_call=1000 * t_llm / max(n_llm, 1),
        repeat_max_abs_diff=float((rep[0] - rep[1]).abs().max()),
        repeat_vs_stored_max_abs_diff=float((rep[0] - r0["vec"]).abs().max()),
        repeat_cos=float(F.cosine_similarity(rep[0], rep[1], dim=0)),
        norm=dict(mean=float(norms.mean()), std=float(norms.std()), min=float(norms.min()), max=float(norms.max())),
        cos_between_objects_mean=float(off.mean()), cos_between_objects_min=float(off.min()),
        cos_within_object_mean=float(torch.stack(within).mean()),
        cos_to_global_mean=float((Vn @ gm.T).mean()),
    )
    json.dump(checks, open(os.path.join(OUT, "checks.json"), "w"), indent=1)
    print(json.dumps(checks, indent=1))


if __name__ == "__main__":
    main()
