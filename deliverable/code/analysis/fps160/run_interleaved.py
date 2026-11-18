"""fps160: E0, E10a, E14 on all 160 SeCVOS videos, interleaved per video, one process (model loaded once).

Usage (from SeC/, PYTHONPATH=.): python experiments/fps160/run_interleaved.py
Per video: E0 -> E10a -> E14, each timed by vos_inference's own code path (perf_counter with cuda synchronize
at both ends, model load excluded). Before the 160 videos, one warm-up video runs all three configs into
experiments/fps160/warmup/ (not counted). Outputs: experiments/fps160/<run>/{masks,frames,videos.csv,progress.txt}
and experiments/fps160/gpu_apps.csv (foreign GPU processes seen by a 2 s nvidia-smi sampler during each run of
each video, plus one snapshot after each video). Resumable at video level (a video is done when E14 logged it).
"""
import csv
import os
import subprocess
import sys
import threading
import time

import numpy as np
import torch

sys.path.insert(0, "vos_evaluation")
import vos_inference as vi  # noqa: E402

OUT = "experiments/fps160"
E10A = ["--burst_after_trigger", "3", "--concept_every_sparse", "15", "--bank_rule", "hsv"]
RUNS = [("E0", []), ("E10a", E10A), ("E14", E10A + ["--bank_diverse_tau", "0.97"])]
WARMUP = "_H00-2zCavQ"
IGNORE = {"gnome-remote-desktop-daemon"}  # persistent 225 MiB daemon of another user (agreed in step 0)
VIDEOS = [l.strip() for f in ("experiments/subset40.txt", "experiments/full160/rest120.txt")
          for l in open(f) if l.strip()]


def run_args(name, extra, root):
    return vi.build_parser().parse_args([
        "--model_path", "saved_models/SeC-4B", "--base_video_dir", "data/SeCVOS/JPEGImages",
        "--input_mask_dir", "data/SeCVOS/Annotations", "--output_mask_dir", f"{root}/{name}/masks",
        "--per_obj_png_file", "--track_object_appearing_later_in_video",
        "--exp_log_dir", f"{root}/{name}", "--seed", "0", *extra])


class GpuSampler(threading.Thread):
    """Every 2 s: compute apps other than this process and IGNORE -> list of (t, 'pid:MiB:name')."""

    def __init__(self):
        super().__init__(daemon=True)
        self.samples = []

    @staticmethod
    def foreign():
        out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory,process_name",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
        apps = []
        for line in out.strip().splitlines():
            pid, mem, name = [x.strip() for x in line.split(",", 2)]
            if int(pid) != os.getpid() and os.path.basename(name) not in IGNORE:
                apps.append(f"{pid}:{mem}MiB:{os.path.basename(name)}")
        return apps

    def run(self):
        while True:
            self.samples.append((time.time(), self.foreign()))
            time.sleep(2)

    def seen(self, t0, t1):
        return sorted({a for t, apps in self.samples if t0 <= t <= t1 for a in apps})


def drop_video(path, video):
    """Remove a partial video's line from a videos.csv (resume after a crash)."""
    if os.path.exists(path):
        lines = open(path).read().splitlines(keepends=True)
        open(path, "w").writelines(l for l in lines if not l.startswith(video + ","))


def done_videos(root):
    p = f"{root}/E14/videos.csv"
    return {r["video"] for r in csv.DictReader(open(p))} if os.path.exists(p) else set()


def run_video(model, tokenizer, video, root, sampler, n, total):
    times = {}
    for name, extra in RUNS:
        args = run_args(name, extra, root)
        os.makedirs(f"{args.exp_log_dir}/frames", exist_ok=True)
        drop_video(f"{args.exp_log_dir}/videos.csv", video)
        exp_cfg = vi.make_exp_cfg(args, None, set())
        torch.cuda.synchronize()
        t_wall = time.time()
        t_video = time.perf_counter()
        vi.vos_separate_inference_per_object(
            predictor=model, tokenizer=tokenizer, base_video_dir=args.base_video_dir,
            input_mask_dir=args.input_mask_dir, output_mask_dir=args.output_mask_dir, video_name=video,
            score_thresh=args.score_thresh, use_all_masks=args.use_all_masks,
            per_obj_png_file=args.per_obj_png_file, exp_cfg=exp_cfg)
        torch.cuda.synchronize()
        video_sec = time.perf_counter() - t_video
        vi.write_exp_logs(args.exp_log_dir, video, exp_cfg["rows"], video_sec, n, total)
        times[name] = (t_wall, time.time())
    snap = GpuSampler.foreign()
    p = f"{root}/gpu_apps.csv"
    new = not os.path.exists(p)
    with open(p, "a") as f:
        if new:
            f.write("video,run,t_start,t_end,foreign_seen,foreign_after_video\n")
        for name, (t0, t1) in times.items():
            f.write(f"{video},{name},{t0:.1f},{t1:.1f},{' '.join(sampler.seen(t0, t1))},{' '.join(snap)}\n")
    print(f"[{time.strftime('%F %T')}] {n}/{total} {video} done", flush=True)


def main():
    import random
    random.seed(0); np.random.seed(0); torch.manual_seed(0)
    from inference.configuration_sec import SeCConfig
    from inference.modeling_sec import SeCModel
    from transformers import AutoTokenizer
    config = SeCConfig.from_pretrained("saved_models/SeC-4B")
    config.apply_postprocessing = False
    config.hydra_overrides_extra = ["++model.non_overlap_masks=false"]
    model = SeCModel.from_pretrained("saved_models/SeC-4B", config=config, torch_dtype=torch.bfloat16,
                                     use_flash_attn=True).eval().to("cuda")
    tokenizer = AutoTokenizer.from_pretrained("saved_models/SeC-4B", trust_remote_code=True)
    sampler = GpuSampler()
    sampler.start()
    if WARMUP not in done_videos(f"{OUT}/warmup"):
        run_video(model, tokenizer, WARMUP, f"{OUT}/warmup", sampler, 0, 0)
    done = done_videos(OUT)
    for n, v in enumerate(VIDEOS, 1):
        if v not in done:
            run_video(model, tokenizer, v, OUT, sampler, n, len(VIDEOS))
    print("all videos done", flush=True)


if __name__ == "__main__":
    main()
