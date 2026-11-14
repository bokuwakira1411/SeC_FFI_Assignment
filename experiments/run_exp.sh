#!/usr/bin/env bash
# One inference experiment on a video subset: wait for GPU, run (resumable), evaluate.
# Usage (from SeC/): bash experiments/run_exp.sh NAME VIDEO_LIST GT_DIR [extra vos_inference args...]
#   e.g. bash experiments/run_exp.sh E0 experiments/subset40.txt experiments/gt_subset40 --save_concepts
# Outputs in experiments/NAME/: masks/, frames/*.csv, videos.csv, progress.txt, run.log,
#                               wait.log, gpu_time.txt, eval.txt
set -u
cd "$(dirname "$0")/.."

NAME=$1; LIST=$2; GT=$3; shift 3
OUT=experiments/$NAME
PY=/home/student/.conda/envs/sec/bin/python
MIN_FREE=13312      # MiB (13 GiB)
CHECK_EVERY=${CHECK_EVERY:-300}     # seconds (override via env)
MAX_RETRIES=5
mkdir -p "$OUT"

export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] [$NAME] $*" | tee -a "$OUT/run.log"; }

wait_for_gpu() {
    while true; do
        nvidia-smi >> "$OUT/wait.log"
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
        if [ "${free:-0}" -ge "$MIN_FREE" ]; then
            log "GPU free = ${free} MiB >= ${MIN_FREE} MiB, starting."
            return
        fi
        log "GPU free = ${free} MiB < ${MIN_FREE} MiB, waiting ${CHECK_EVERY}s..." | tee -a "$OUT/wait.log"
        echo "$(date '+%F %T') waiting for GPU (free ${free} MiB)" >> "$OUT/progress.txt"
        sleep "$CHECK_EVERY"
    done
}

gpu_sec=0
retries=0
while true; do
    wait_for_gpu
    start_line=$(($(wc -l < "$OUT/run.log") + 1))
    t0=$(date +%s)
    "$PY" vos_evaluation/vos_inference.py \
        --model_path saved_models/SeC-4B \
        --base_video_dir data/SeCVOS/JPEGImages \
        --input_mask_dir data/SeCVOS/Annotations \
        --video_list_file "$LIST" \
        --output_mask_dir "$OUT/masks" \
        --per_obj_png_file \
        --track_object_appearing_later_in_video \
        --exp_log_dir "$OUT" \
        --skip_done \
        "$@" 2>&1 | tee -a "$OUT/run.log"
    rc=${PIPESTATUS[0]}
    gpu_sec=$((gpu_sec + $(date +%s) - t0))
    if [ "$rc" -eq 0 ]; then
        break
    fi
    if tail -n +"$start_line" "$OUT/run.log" | grep -qE "CUDA out of memory|OutOfMemoryError"; then
        retries=$((retries + 1))
        if [ "$retries" -gt "$MAX_RETRIES" ]; then
            log "CUDA OOM; reached max retries ($MAX_RETRIES). Stopping."
            exit 1
        fi
        log "CUDA OOM. Retry $retries/$MAX_RETRIES (resumes from the first unfinished video)."
        continue
    fi
    log "Inference failed with exit code $rc (not OOM). Stopping."
    exit "$rc"
done

echo "$gpu_sec" > "$OUT/gpu_time.txt"
log "GPU time (inference process wall time, all attempts): ${gpu_sec}s = $(awk "BEGIN{printf \"%.2f\", $gpu_sec/3600}") h"

"$PY" vos_evaluation/sav_evaluator.py \
    --gt_root "$GT" \
    --pred_root "$OUT/masks" \
    --scene_info data/SeCVOS/scene_info.json \
    --strict > "$OUT/eval.txt" 2>&1
log "Evaluation exit $? -> $OUT/eval.txt"
grep -A1 "Summary" "$OUT/eval.txt" | tee -a "$OUT/run.log"
