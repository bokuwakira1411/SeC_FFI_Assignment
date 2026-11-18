"""Bank descriptors of every frame, per run and object (GPU, SAM 2 image encoder only; no tracking, no LLM).

Usage (from SeC/): python experiments/bank/descriptors.py RUN [RUN ...]
  RUN = GT | an experiment dir under experiments/ (e.g. E10a, bank/E13a)
Writes experiments/bank/desc/<RUN>/<video>_<obj>.pt: float16 [n_frames, 256], NaN rows where the mask is
empty / missing. Mask = GT PNG (RUN = GT) or the run's predicted PNG; with --per_obj_png_file the saved
PNG is exactly the mask the run put into the MLLM memory (pred > 0, no non-overlap step).
Descriptor = inference.modeling_sec.mask_pooled_descriptor (same as --bank_diverse_tau): Hiera feature
current_vision_feats[-1] from _get_image_feature (bf16 autocast, as vos_inference), averaged in the mask.
"""
import os
import sys

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.getcwd())
from inference.modeling_sec import mask_pooled_descriptor  # noqa: E402

VID = "data/SeCVOS/JPEGImages"
ANN = "data/SeCVOS/Annotations"
OUT = "experiments/bank/desc"


def load_mask(path):
    return np.array(Image.open(path)) > 0 if os.path.exists(path) else None


@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)  # as vos_separate_inference_per_object
def main():
    runs = sys.argv[1:]
    videos = [l.strip() for l in open("experiments/subset40.txt") if l.strip()]
    todo = [(r, v) for v in videos for r in runs
            if not all(os.path.exists(f"{OUT}/{r}/{v}_{o}.pt") for o in sorted(os.listdir(f"{ANN}/{v}")))]
    if not todo:
        print("nothing to do"); return
    from inference.configuration_sec import SeCConfig
    from inference.modeling_sec import SeCModel
    mp = "saved_models/SeC-4B"
    config = SeCConfig.from_pretrained(mp)
    config.hydra_overrides_extra = ["++model.non_overlap_masks=false"]  # as vos_inference --per_obj_png_file
    model = SeCModel.from_pretrained(mp, config=config, torch_dtype=torch.bfloat16, use_flash_attn=True)
    enc = model.grounding_encoder.eval().to("cuda")
    del model
    for r in runs:
        os.makedirs(f"{OUT}/{r}", exist_ok=True)
    for v in videos:
        rs = [r for r, vv in todo if vv == v]
        if not rs:
            continue
        frames = sorted(os.listdir(f"{VID}/{v}"), key=lambda p: int(os.path.splitext(p)[0]))
        names = [os.path.splitext(p)[0] for p in frames]
        state = enc.init_state(video_path=f"{VID}/{v}", async_loading_frames=False, offload_video_to_cpu=True)
        feats = [enc._get_image_feature(state, f, 1)[2][-1].clone() for f in range(len(frames))]  # [4096,1,256]
        for r in rs:
            src = ANN if r == "GT" else f"experiments/{r}/masks"
            for o in sorted(os.listdir(f"{ANN}/{v}")):
                d = torch.full((len(frames), 256), float("nan"))
                for f, n in enumerate(names):
                    m = load_mask(f"{src}/{v}/{o}/{n}.png")
                    if m is not None and m.any():
                        d[f] = mask_pooled_descriptor(feats[f], m).float().cpu()
                torch.save(d.half(), f"{OUT}/{r}/{v}_{o}.pt")
        del state, feats
        torch.cuda.empty_cache()
        print(v, rs, flush=True)


if __name__ == "__main__":
    main()
