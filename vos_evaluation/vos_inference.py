# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import csv
import json
import os
import time
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
# from sam2.build_sam import build_sam2_video_predictor


# the PNG palette for DAVIS 2017 dataset
DAVIS_PALETTE = b"\x00\x00\x00\x80\x00\x00\x00\x80\x00\x80\x80\x00\x00\x00\x80\x80\x00\x80\x00\x80\x80\x80\x80\x80@\x00\x00\xc0\x00\x00@\x80\x00\xc0\x80\x00@\x00\x80\xc0\x00\x80@\x80\x80\xc0\x80\x80\x00@\x00\x80@\x00\x00\xc0\x00\x80\xc0\x00\x00@\x80\x80@\x80\x00\xc0\x80\x80\xc0\x80@@\x00\xc0@\x00@\xc0\x00\xc0\xc0\x00@@\x80\xc0@\x80@\xc0\x80\xc0\xc0\x80\x00\x00@\x80\x00@\x00\x80@\x80\x80@\x00\x00\xc0\x80\x00\xc0\x00\x80\xc0\x80\x80\xc0@\x00@\xc0\x00@@\x80@\xc0\x80@@\x00\xc0\xc0\x00\xc0@\x80\xc0\xc0\x80\xc0\x00@@\x80@@\x00\xc0@\x80\xc0@\x00@\xc0\x80@\xc0\x00\xc0\xc0\x80\xc0\xc0@@@\xc0@@@\xc0@\xc0\xc0@@@\xc0\xc0@\xc0@\xc0\xc0\xc0\xc0\xc0 \x00\x00\xa0\x00\x00 \x80\x00\xa0\x80\x00 \x00\x80\xa0\x00\x80 \x80\x80\xa0\x80\x80`\x00\x00\xe0\x00\x00`\x80\x00\xe0\x80\x00`\x00\x80\xe0\x00\x80`\x80\x80\xe0\x80\x80 @\x00\xa0@\x00 \xc0\x00\xa0\xc0\x00 @\x80\xa0@\x80 \xc0\x80\xa0\xc0\x80`@\x00\xe0@\x00`\xc0\x00\xe0\xc0\x00`@\x80\xe0@\x80`\xc0\x80\xe0\xc0\x80 \x00@\xa0\x00@ \x80@\xa0\x80@ \x00\xc0\xa0\x00\xc0 \x80\xc0\xa0\x80\xc0`\x00@\xe0\x00@`\x80@\xe0\x80@`\x00\xc0\xe0\x00\xc0`\x80\xc0\xe0\x80\xc0 @@\xa0@@ \xc0@\xa0\xc0@ @\xc0\xa0@\xc0 \xc0\xc0\xa0\xc0\xc0`@@\xe0@@`\xc0@\xe0\xc0@`@\xc0\xe0@\xc0`\xc0\xc0\xe0\xc0\xc0\x00 \x00\x80 \x00\x00\xa0\x00\x80\xa0\x00\x00 \x80\x80 \x80\x00\xa0\x80\x80\xa0\x80@ \x00\xc0 \x00@\xa0\x00\xc0\xa0\x00@ \x80\xc0 \x80@\xa0\x80\xc0\xa0\x80\x00`\x00\x80`\x00\x00\xe0\x00\x80\xe0\x00\x00`\x80\x80`\x80\x00\xe0\x80\x80\xe0\x80@`\x00\xc0`\x00@\xe0\x00\xc0\xe0\x00@`\x80\xc0`\x80@\xe0\x80\xc0\xe0\x80\x00 @\x80 @\x00\xa0@\x80\xa0@\x00 \xc0\x80 \xc0\x00\xa0\xc0\x80\xa0\xc0@ @\xc0 @@\xa0@\xc0\xa0@@ \xc0\xc0 \xc0@\xa0\xc0\xc0\xa0\xc0\x00`@\x80`@\x00\xe0@\x80\xe0@\x00`\xc0\x80`\xc0\x00\xe0\xc0\x80\xe0\xc0@`@\xc0`@@\xe0@\xc0\xe0@@`\xc0\xc0`\xc0@\xe0\xc0\xc0\xe0\xc0  \x00\xa0 \x00 \xa0\x00\xa0\xa0\x00  \x80\xa0 \x80 \xa0\x80\xa0\xa0\x80` \x00\xe0 \x00`\xa0\x00\xe0\xa0\x00` \x80\xe0 \x80`\xa0\x80\xe0\xa0\x80 `\x00\xa0`\x00 \xe0\x00\xa0\xe0\x00 `\x80\xa0`\x80 \xe0\x80\xa0\xe0\x80``\x00\xe0`\x00`\xe0\x00\xe0\xe0\x00``\x80\xe0`\x80`\xe0\x80\xe0\xe0\x80  @\xa0 @ \xa0@\xa0\xa0@  \xc0\xa0 \xc0 \xa0\xc0\xa0\xa0\xc0` @\xe0 @`\xa0@\xe0\xa0@` \xc0\xe0 \xc0`\xa0\xc0\xe0\xa0\xc0 `@\xa0`@ \xe0@\xa0\xe0@ `\xc0\xa0`\xc0 \xe0\xc0\xa0\xe0\xc0``@\xe0`@`\xe0@\xe0\xe0@``\xc0\xe0`\xc0`\xe0\xc0\xe0\xe0\xc0"


def split_list(lst, n):
    """Split a list into n (roughly) equal-sized non-empty chunks."""
    # Compute sizes of the chunks
    avg = len(lst) // n
    remainder = len(lst) % n

    # Create the chunks
    chunks = []
    start = 0
    for i in range(n):
        chunk_size = avg + (1 if i < remainder else 0)
        chunks.append(lst[start:start + chunk_size])
        start += chunk_size

    return chunks

def get_chunk(lst, n, k):
    chunks = split_list(lst, n)
    return chunks[k]
 
    
def load_ann_png(path):
    """Load a PNG file as a mask and its palette."""
    mask = Image.open(path)
    palette = mask.getpalette()
    mask = np.array(mask).astype(np.uint8)
    return mask, palette


def save_ann_png(path, mask, palette):
    """Save a mask as a PNG file with the given palette."""
    assert mask.dtype == np.uint8
    assert mask.ndim == 2
    output_mask = Image.fromarray(mask)
    output_mask.putpalette(palette)
    output_mask.save(path)


def get_per_obj_mask(mask):
    """Split a mask into per-object masks."""
    object_ids = np.unique(mask)
    object_ids = object_ids[object_ids > 0].tolist()
    per_obj_mask = {object_id: (mask == object_id) for object_id in object_ids}
    return per_obj_mask


def put_per_obj_mask(per_obj_mask, height, width):
    """Combine per-object masks into a single mask."""
    mask = np.zeros((height, width), dtype=np.uint8)
    object_ids = sorted(per_obj_mask)[::-1]
    for object_id in object_ids:
        object_mask = per_obj_mask[object_id]
        object_mask = object_mask.reshape(height, width)
        mask[object_mask] = object_id
    return mask


def load_masks_from_dir(
    input_mask_dir, video_name, frame_name, per_obj_png_file, allow_missing=False
):
    """Load masks from a directory as a dict of per-object masks."""
    if not per_obj_png_file:
        input_mask_path = os.path.join(input_mask_dir, video_name, f"{frame_name}.png")
        if allow_missing and not os.path.exists(input_mask_path):
            return {}, None
        input_mask, input_palette = load_ann_png(input_mask_path)
        per_obj_input_mask = get_per_obj_mask(input_mask)
    else:
        per_obj_input_mask = {}
        input_palette = None
        # each object is a directory in "{object_id:%03d}" format
        for object_name in os.listdir(os.path.join(input_mask_dir, video_name)):
            object_id = int(object_name)
            input_mask_path = os.path.join(
                input_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            if allow_missing and not os.path.exists(input_mask_path):
                continue
            input_mask, input_palette = load_ann_png(input_mask_path)
            per_obj_input_mask[object_id] = input_mask > 0

    return per_obj_input_mask, input_palette


def save_masks_to_dir(
    output_mask_dir,
    video_name,
    frame_name,
    per_obj_output_mask,
    height,
    width,
    per_obj_png_file,
    output_palette,
):
    """Save masks to a directory as PNG files."""
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    if not per_obj_png_file:
        output_mask = put_per_obj_mask(per_obj_output_mask, height, width)
        output_mask_path = os.path.join(
            output_mask_dir, video_name, f"{frame_name}.png"
        )
        save_ann_png(output_mask_path, output_mask, output_palette)
    else:
        for object_id, object_mask in per_obj_output_mask.items():
            object_name = f"{object_id:03d}"
            os.makedirs(
                os.path.join(output_mask_dir, video_name, object_name),
                exist_ok=True,
            )
            output_mask = object_mask.reshape(height, width).astype(np.uint8)
            output_mask_path = os.path.join(
                output_mask_dir, video_name, object_name, f"{frame_name}.png"
            )
            save_ann_png(output_mask_path, output_mask, output_palette)

@torch.inference_mode()
@torch.autocast(device_type="cuda", dtype=torch.bfloat16)
def vos_separate_inference_per_object(
    predictor,
    tokenizer,
    base_video_dir,
    input_mask_dir,
    output_mask_dir,
    video_name,
    score_thresh=0.0,
    use_all_masks=False,
    per_obj_png_file=False,
    exp_cfg=None,
):
    """
    Run VOS inference on a single video with the given predictor.

    exp_cfg (experiments, see CHANGES.md): None = original behaviour. Otherwise a dict with
    hsv_thresh, concept_mode, swap_map, concept_dir, rows; a per-object exp_ctx is built from it.

    Unlike `vos_inference`, this function run inference separately for each object
    in a video, which could be applied to datasets like LVOS or YouTube-VOS that
    don't have all objects to track appearing in the first frame (i.e. some objects
    might appear only later in the video).
    """
    # load the video frames and initialize the inference state on this video
    video_dir = os.path.join(base_video_dir, video_name)
    frame_names = [
        os.path.splitext(p)[0]
        for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1] in [".jpg", ".jpeg", ".JPG", ".JPEG"]
    ]
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))
    inference_state = predictor.grounding_encoder.init_state(
        video_path=video_dir, async_loading_frames=False, offload_video_to_cpu=True
    )
    height = inference_state["video_height"]
    width = inference_state["video_width"]
    input_palette = None

    # collect all the object ids and their input masks
    inputs_per_object = defaultdict(dict)
    for idx, name in enumerate(frame_names):
        if per_obj_png_file or os.path.exists(
            os.path.join(input_mask_dir, video_name, f"{name}.png")
        ):
            per_obj_input_mask, input_palette = load_masks_from_dir(
                input_mask_dir=input_mask_dir,
                video_name=video_name,
                frame_name=frame_names[idx],
                per_obj_png_file=per_obj_png_file,
                allow_missing=True,
            )
            for object_id, object_mask in per_obj_input_mask.items():
                # skip empty masks
                if not np.any(object_mask):
                    continue
                # if `use_all_masks=False`, we only use the first mask for each object
                if len(inputs_per_object[object_id]) > 0 and not use_all_masks:
                    continue
                print(f"adding mask from frame {idx} as input for {object_id=}")
                inputs_per_object[object_id][idx] = object_mask

    # run inference separately for each object in the video
    object_ids = sorted(inputs_per_object)
    output_scores_per_object = defaultdict(dict)
    for object_id in object_ids:
        # add those input masks to SAM 2 inference state before propagation
        input_frame_inds = sorted(inputs_per_object[object_id])
        predictor.grounding_encoder.reset_state(inference_state)
        for input_frame_idx in input_frame_inds:
            predictor.grounding_encoder.add_new_mask(
                inference_state=inference_state,
                frame_idx=input_frame_idx,
                obj_id=object_id,
                mask=inputs_per_object[object_id][input_frame_idx],
            )
        exp_ctx = None
        if exp_cfg is not None:
            obj_name = f"{object_id:03d}" if per_obj_png_file else str(object_id)
            exp_ctx = dict(
                video=video_name, obj=obj_name,
                hsv_thresh=exp_cfg["hsv_thresh"], concept_mode=exp_cfg["concept_mode"],
                rows=exp_cfg["rows"], concept_dir=exp_cfg["concept_dir"],
                bank_rule=exp_cfg["bank_rule"], bank_extra_gap=exp_cfg["bank_extra_gap"],
                concept_every=exp_cfg["concept_every"],
                burst_after_trigger=exp_cfg["burst_after_trigger"],
                concept_every_sparse=exp_cfg["concept_every_sparse"],
                trigger_mode=exp_cfg["trigger_mode"], unc_logit_drop=exp_cfg["unc_logit_drop"],
                unc_area_drop=exp_cfg["unc_area_drop"], unc_empty_n=exp_cfg["unc_empty_n"],
                unc_window=exp_cfg["unc_window"], unc_cooldown=exp_cfg["unc_cooldown"],
                force_trigger=exp_cfg["force_trigger"], bank_score_thresh=exp_cfg["bank_score_thresh"],
            )
            if exp_cfg["oracle_keyframe"] is not None:
                k = exp_cfg["oracle_keyframe"]
                k_masks, _ = load_masks_from_dir(
                    input_mask_dir=input_mask_dir, video_name=video_name, frame_name=frame_names[k],
                    per_obj_png_file=per_obj_png_file, allow_missing=True,
                )
                assert np.any(k_masks.get(object_id)), f"oracle keyframe {k}: empty GT mask for {object_id}"
                exp_ctx["oracle_keyframes"] = [(k, k_masks[object_id])]
            if exp_cfg["concept_mode"] == "swap":
                exp_ctx["swap_concepts"] = [
                    torch.load(f) for f in exp_cfg["swap_map"][f"{video_name}_{obj_name}"]
                ]
        # run propagation throughout the video and collect the results in a dict
        for out_frame_idx, _, out_mask_logits in predictor.propagate_in_video(
            inference_state,
            start_frame_idx=min(input_frame_inds),
            reverse=False,
            init_mask=inputs_per_object[object_id][min(input_frame_inds)],
            tokenizer=tokenizer,
            exp_ctx=exp_ctx,
        ):
            obj_scores = out_mask_logits.cpu().numpy()
            output_scores_per_object[object_id][out_frame_idx] = obj_scores

    # post-processing: consolidate the per-object scores into per-frame masks
    os.makedirs(os.path.join(output_mask_dir, video_name), exist_ok=True)
    output_palette = input_palette or DAVIS_PALETTE
    video_segments = {}  # video_segments contains the per-frame segmentation results
    for frame_idx in range(len(frame_names)):
        scores = torch.full(
            size=(len(object_ids), 1, height, width),
            fill_value=-1024.0,
            dtype=torch.float32,
        )
        for i, object_id in enumerate(object_ids):
            if frame_idx in output_scores_per_object[object_id]:
                scores[i] = torch.from_numpy(
                    output_scores_per_object[object_id][frame_idx]
                )

        if not per_obj_png_file:
            scores = predictor.grounding_encoder._apply_non_overlapping_constraints(scores)
        per_obj_output_mask = {
            object_id: (scores[i] > score_thresh).cpu().numpy()
            for i, object_id in enumerate(object_ids)
        }
        video_segments[frame_idx] = per_obj_output_mask

    # write the output masks as palette PNG files to output_mask_dir
    for frame_idx, per_obj_output_mask in video_segments.items():
        save_masks_to_dir(
            output_mask_dir=output_mask_dir,
            video_name=video_name,
            frame_name=frame_names[frame_idx],
            per_obj_output_mask=per_obj_output_mask,
            height=height,
            width=width,
            per_obj_png_file=per_obj_png_file,
            output_palette=output_palette,
        )

def video_output_complete(base_video_dir, input_mask_dir, output_mask_dir, video_name):
    """True if every GT object of the video has one output PNG per input frame."""
    n_frames = sum(f.lower().endswith((".jpg", ".jpeg"))
                   for f in os.listdir(os.path.join(base_video_dir, video_name)))
    out_dir = os.path.join(output_mask_dir, video_name)
    gt_dir = os.path.join(input_mask_dir, video_name)
    if not os.path.isdir(out_dir):
        return False
    gt_objs = [o for o in os.listdir(gt_dir) if os.path.isdir(os.path.join(gt_dir, o))]
    return all(
        os.path.isdir(os.path.join(out_dir, o))
        and sum(f.endswith(".png") for f in os.listdir(os.path.join(out_dir, o))) == n_frames
        for o in gt_objs
    )


def write_exp_logs(exp_log_dir, video_name, rows, video_sec, n_done, n_total):
    """Per-frame CSV for one video, one line in videos.csv, one line in progress.txt."""
    fields = ["video", "obj", "frame", "is_cond", "hsv_dist", "hsv_trigger", "triggered",
              "mllm_memory_len", "n_ctx_frames", "mem_appended", "bank_reason", "mask_px", "pred_iou",
              "object_score_logit", "predict_forward_ms", "frame_ms",
              "ctx_frames", "bank_frames", "sam_mem_frames", "concept_reason"]
    with open(os.path.join(exp_log_dir, "frames", f"{video_name}.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    n_obj = len({r["obj"] for r in rows})
    n_frames = len({r["frame"] for r in rows})
    n_trig = sum(int(r["triggered"]) for r in rows)
    videos_csv = os.path.join(exp_log_dir, "videos.csv")
    new = not os.path.exists(videos_csv)
    with open(videos_csv, "a") as f:
        if new:
            f.write("video,n_obj,n_frames,n_obj_frames,n_triggers,video_sec\n")
        f.write(f"{video_name},{n_obj},{n_frames},{len(rows)},{n_trig},{video_sec:.2f}\n")
    with open(os.path.join(exp_log_dir, "progress.txt"), "a") as f:
        f.write(f"{time.strftime('%F %T')} {n_done}/{n_total} {video_name} "
                f"obj={n_obj} frames={n_frames} triggers={n_trig} {video_sec:.1f}s\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="path to the SAM 2 model checkpoint (e.g. /path/to/sam2)",
    )
    parser.add_argument(
        "--base_video_dir",
        type=str,
        required=True,
        help="directory containing videos (as JPEG files) to run VOS prediction on",
    )
    parser.add_argument(
        "--input_mask_dir",
        type=str,
        required=True,
        help="directory containing input masks (as PNG files) of each video",
    )
    parser.add_argument(
        "--video_list_file",
        type=str,
        default=None,
        help="text file containing the list of video names to run VOS prediction on",
    )
    parser.add_argument(
        "--output_mask_dir",
        type=str,
        required=True,
        help="directory to save the output masks (as PNG files)",
    )
    parser.add_argument(
        "--score_thresh",
        type=float,
        default=0.0,
        help="threshold for the output mask logits (default: 0.0)",
    )
    parser.add_argument(
        "--use_all_masks",
        action="store_true",
        help="whether to use all available PNG files in input_mask_dir "
        "(default without this flag: just the first PNG file as input to the SAM 2 model; "
        "usually we don't need this flag, since semi-supervised VOS evaluation usually takes input from the first frame only)",
    )
    parser.add_argument(
        "--per_obj_png_file",
        action="store_true",
        help="whether use separate per-object PNG files for input and output masks "
        "(default without this flag: all object masks are packed into a single PNG file on each frame following DAVIS format; "
        "note that the SA-V dataset stores each object mask as an individual PNG file and requires this flag)",
    )
    parser.add_argument(
        "--apply_postprocessing",
        action="store_true",
        help="whether to apply postprocessing (e.g. hole-filling) to the output masks "
        "(we don't apply such post-processing in the SAM 2 model evaluation)",
    )
    parser.add_argument(
        "--track_object_appearing_later_in_video",
        action="store_true",
        help="whether to track objects that appear later in the video (i.e. not on the first frame; "
        "some VOS datasets like LVOS or YouTube-VOS don't have all objects appearing in the first frame)",
    )
    parser.add_argument(
        "--chunk_id",
        type=int,
        default=0,
        help="the index of chunk",
    )
    parser.add_argument(
        "--num_chunks",
        type=int,
        default=1,
        help="total number of chunks",
    )
    parser.add_argument(
        "--node_id",
        type=int,
        default=0,
        help="the index of gpu node",
    )
    parser.add_argument(
        "--num_nodes",
        type=int,
        default=1,
        help="the number of node (8 cards) used",
    )
    # --- experiments (see CHANGES.md); defaults reproduce the original behaviour ---
    parser.add_argument("--hsv_thresh", type=float, default=0.35,
                        help="HSV scene-change threshold that triggers the concept (MLLM) update")
    parser.add_argument("--concept_mode", default="default",
                        choices=["default", "off", "always", "swap"],
                        help="off: never trigger; always: trigger on every frame; "
                             "swap: on trigger use concepts listed in --swap_map")
    parser.add_argument("--swap_map", type=str, default=None,
                        help="json {'<video>_<obj>': [concept .pt files]} for --concept_mode swap")
    parser.add_argument("--exp_log_dir", type=str, default=None,
                        help="write per-frame CSV (frames/<video>.csv), videos.csv and progress.txt here")
    parser.add_argument("--save_concepts", action="store_true",
                        help="save language_embd at every trigger to <exp_log_dir>/concepts/")
    parser.add_argument("--bank_rule", default="default", choices=["default", "hsv"],
                        help="hsv: MLLM memory updated only on HSV-trigger frames (even with --concept_mode always)")
    parser.add_argument("--bank_extra_gap", type=int, default=0,
                        help=">0: also add non-trigger frames (mask not empty, object score > 1) "
                             "at least this many frames after the last added frame; 0 = off")
    parser.add_argument("--concept_every", type=int, default=0,
                        help=">0: also compute the concept on frames with frame_idx %% N == 0 "
                             "(HSV triggers still apply); 0 = off")
    parser.add_argument("--burst_after_trigger", type=int, default=0,
                        help=">0: also compute the concept on the K tracked frames after each HSV trigger "
                             "(a new trigger restarts the count); 0 = off")
    parser.add_argument("--concept_every_sparse", type=int, default=0,
                        help=">0: outside bursts, also compute the concept when the last concept call is "
                             ">= P frames back; 0 = off")
    parser.add_argument("--trigger_mode", default="hsv", choices=["hsv", "unc", "hsv_or_unc"],
                        help="concept-call trigger: hsv (original) | unc (uncertainty only) | hsv_or_unc")
    parser.add_argument("--unc_logit_drop", type=float, default=None,
                        help="uncertainty S1: object score logit drops > D below the mean of the previous w frames")
    parser.add_argument("--unc_area_drop", type=float, default=None,
                        help="uncertainty S2: mask area drops > r (fraction) below the mean of the previous w frames")
    parser.add_argument("--unc_empty_n", type=int, default=0,
                        help="uncertainty S3: mask empty on the last N frames; 0 = off")
    parser.add_argument("--unc_window", type=int, default=5, help="w for S1/S2")
    parser.add_argument("--unc_cooldown", type=int, default=0,
                        help="no uncertainty call if the last concept call is <= C frames back")
    parser.add_argument("--oracle_keyframe", type=int, default=None,
                        help="put this frame with its GT mask into the MLLM memory from the start")
    parser.add_argument("--force_trigger", type=str, default="",
                        help="frames where the concept update is forced, e.g. '10-30,45'")
    parser.add_argument("--bank_score_thresh", type=float, default=1.0,
                        help="object score a trigger frame needs to enter the MLLM memory (original: 1)")
    parser.add_argument("--skip_done", action="store_true",
                        help="skip videos whose output masks are already complete (resume)")
    args = parser.parse_args()
    if args.concept_mode == "swap":
        assert args.swap_map, "--concept_mode swap needs --swap_map"
    use_exp = (args.exp_log_dir is not None or args.concept_mode != "default" or args.hsv_thresh != 0.35
               or args.bank_rule != "default" or args.bank_extra_gap != 0 or args.concept_every != 0
               or args.burst_after_trigger != 0 or args.concept_every_sparse != 0
               or args.trigger_mode != "hsv"
               or args.oracle_keyframe is not None or args.force_trigger or args.bank_score_thresh != 1.0)
    force_trigger = set()
    for part in filter(None, args.force_trigger.split(",")):
        a, _, b = part.partition("-")
        force_trigger.update(range(int(a), int(b or a) + 1))
    if args.exp_log_dir is not None:
        os.makedirs(os.path.join(args.exp_log_dir, "frames"), exist_ok=True)
        if args.save_concepts:
            os.makedirs(os.path.join(args.exp_log_dir, "concepts"), exist_ok=True)
    swap_map = json.load(open(args.swap_map)) if args.swap_map else None

    # if we use per-object PNG files, they could possibly overlap in inputs and outputs
    hydra_overrides_extra = [
        "++model.non_overlap_masks=" + ("false" if args.per_obj_png_file else "true")
    ]
    from inference.configuration_sec import SeCConfig
    from inference.modeling_sec import SeCModel
    from transformers import AutoTokenizer
    
    config = SeCConfig.from_pretrained(args.model_path)
    config.apply_postprocessing = args.apply_postprocessing
    config.hydra_overrides_extra = hydra_overrides_extra
    
    model = SeCModel.from_pretrained(
        args.model_path,
        config=config, 
        torch_dtype=torch.bfloat16, 
        use_flash_attn=True
    ).eval().to('cuda')
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )


    if args.use_all_masks:
        print("using all available masks in input_mask_dir as input to the SAM 2 model")
    else:
        print(
            "using only the first frame's mask in input_mask_dir as input to the SAM 2 model"
        )
    # if a video list file is provided, read the video names from the file
    # (otherwise, we use all subdirectories in base_video_dir)
    if args.video_list_file is not None:
        with open(args.video_list_file, "r") as f:
            video_names = [v.strip() for v in f.readlines()]
    else:
        video_names = [
            p
            for p in os.listdir(args.base_video_dir)
            if os.path.isdir(os.path.join(args.base_video_dir, p))
        ]

    video_names = get_chunk(video_names, args.num_nodes*args.num_chunks, args.chunk_id+8*args.node_id)
    print(f"running VOS prediction on {len(video_names)} videos:\n{video_names}")

    for n_video, video_name in enumerate(video_names):
        print(f"\n{n_video + 1}/{len(video_names)} - running on {video_name}")
        if args.skip_done and video_output_complete(
            args.base_video_dir, args.input_mask_dir, args.output_mask_dir, video_name
        ):
            print(f"skip {video_name}: output already complete")
            continue
        exp_cfg = None
        if use_exp:
            exp_cfg = dict(
                hsv_thresh=args.hsv_thresh, concept_mode=args.concept_mode, swap_map=swap_map,
                bank_rule=args.bank_rule, bank_extra_gap=args.bank_extra_gap,
                concept_every=args.concept_every,
                burst_after_trigger=args.burst_after_trigger, concept_every_sparse=args.concept_every_sparse,
                trigger_mode=args.trigger_mode, unc_logit_drop=args.unc_logit_drop,
                unc_area_drop=args.unc_area_drop, unc_empty_n=args.unc_empty_n,
                unc_window=args.unc_window, unc_cooldown=args.unc_cooldown,
                oracle_keyframe=args.oracle_keyframe, force_trigger=force_trigger,
                bank_score_thresh=args.bank_score_thresh,
                rows=[] if args.exp_log_dir else None,
                concept_dir=os.path.join(args.exp_log_dir, "concepts")
                if (args.exp_log_dir and args.save_concepts) else None,
            )
        torch.cuda.synchronize()
        t_video = time.perf_counter()
        if not args.track_object_appearing_later_in_video:
            raise NotImplementedError
        else:
            vos_separate_inference_per_object(
                predictor=model,
                tokenizer=tokenizer,
                base_video_dir=args.base_video_dir,
                input_mask_dir=args.input_mask_dir,
                output_mask_dir=args.output_mask_dir,
                video_name=video_name,
                score_thresh=args.score_thresh,
                use_all_masks=args.use_all_masks,
                per_obj_png_file=args.per_obj_png_file,
                exp_cfg=exp_cfg,
            )
        torch.cuda.synchronize()
        video_sec = time.perf_counter() - t_video
        if args.exp_log_dir:
            write_exp_logs(args.exp_log_dir, video_name, exp_cfg["rows"], video_sec,
                           n_video + 1, len(video_names))

    print(
        f"completed VOS prediction on {len(video_names)} videos -- "
        f"output masks saved to {args.output_mask_dir}"
    )


if __name__ == "__main__":
    main()
