#!/usr/bin/env bash
# Run one configuration on a video list, then score it (J&F per object, SeCVOS scene-aware protocol).
# Usage: bash run_one.sh NAME VIDEO_LIST [extra vos_inference.py flags...]
# Env:   SEC_ROOT  = SeC repo with sec_modifications.patch applied (default: current dir)
#        DATA_ROOT = folder with SeCVOS/{JPEGImages,Annotations,scene_info.json} (default: $SEC_ROOT/data)
#        PYTHON    = python of the SeC env (default: python)
#        OUT_ROOT  = where results go (default: $SEC_ROOT/runs)
# Simplified from experiments/run_exp.sh (that script also waited for a free GPU and retried on CUDA OOM).
set -euo pipefail
NAME=$1; LIST=$(realpath "$2"); shift 2
SEC_ROOT=${SEC_ROOT:-$PWD}; DATA_ROOT=${DATA_ROOT:-$SEC_ROOT/data}; PYTHON=${PYTHON:-python}
OUT=${OUT_ROOT:-$SEC_ROOT/runs}/$NAME
mkdir -p "$OUT/gt"
cd "$SEC_ROOT"
export PYTHONPATH=$SEC_ROOT PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# GT folder restricted to the video list (the evaluator scores every video in --gt_root)
for v in $(cat "$LIST"); do ln -sfn "$DATA_ROOT/SeCVOS/Annotations/$v" "$OUT/gt/$v"; done
"$PYTHON" vos_evaluation/vos_inference.py \
    --model_path saved_models/SeC-4B \
    --base_video_dir "$DATA_ROOT/SeCVOS/JPEGImages" \
    --input_mask_dir "$DATA_ROOT/SeCVOS/Annotations" \
    --video_list_file "$LIST" \
    --output_mask_dir "$OUT/masks" \
    --per_obj_png_file --track_object_appearing_later_in_video \
    --exp_log_dir "$OUT" --skip_done "$@"
"$PYTHON" vos_evaluation/sav_evaluator.py --gt_root "$OUT/gt" --pred_root "$OUT/masks" \
    --scene_info "$DATA_ROOT/SeCVOS/scene_info.json" --strict > "$OUT/eval.txt"
# per-object J&F: $OUT/masks/results_overall.csv
