#!/usr/bin/env bash
# Run SeC inference on all 160 SeCVOS videos when the shared GPU has room,
# retry on CUDA OOM (max 5 retries), then evaluate.
# Usage (from SeC/): bash run_full.sh
set -u
cd "$(dirname "$0")"

PY=/home/student/.conda/envs/sec/bin/python
LOG=logs_secvos.txt
REST=data/SeCVOS/video_id_rest.txt
MIN_FREE=12000      # MiB
CHECK_EVERY=300     # seconds
MAX_RETRIES=5

export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { echo "[$(date '+%F %T')] [run_full] $*" | tee -a "$LOG"; }

wait_for_gpu() {
    while true; do
        free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
        if [ "${free:-0}" -ge "$MIN_FREE" ]; then
            log "GPU free = ${free} MiB >= ${MIN_FREE} MiB, starting."
            return
        fi
        log "GPU free = ${free} MiB < ${MIN_FREE} MiB, waiting ${CHECK_EVERY}s..."
        sleep "$CHECK_EVERY"
    done
}

retries=0
last_clean=0
while true; do
    wait_for_gpu

    "$PY" make_rest_list.py 2>&1 | tee -a "$LOG"
    if [ ! -s "$REST" ]; then
        log "All videos done."
        break
    fi
    if [ "$last_clean" -eq 1 ]; then
        log "Inference exited OK but $(wc -l < "$REST") videos are still incomplete: $(tr '\n' ' ' < "$REST"). Stopping."
        exit 1
    fi
    log "Remaining videos: $(wc -l < "$REST")"

    start_line=$(($(wc -l < "$LOG") + 1))
    "$PY" vos_evaluation/vos_inference.py \
        --model_path saved_models/SeC-4B \
        --base_video_dir data/SeCVOS/JPEGImages \
        --input_mask_dir data/SeCVOS/Annotations \
        --video_list_file "$REST" \
        --output_mask_dir outputs/secvos_full \
        --per_obj_png_file \
        --track_object_appearing_later_in_video \
        2>&1 | tee -a "$LOG"
    rc=${PIPESTATUS[0]}

    if tail -n +"$start_line" "$LOG" | grep -qE "CUDA out of memory|OutOfMemoryError"; then
        retries=$((retries + 1))
        if [ "$retries" -gt "$MAX_RETRIES" ]; then
            log "CUDA OOM again; reached max retries ($MAX_RETRIES). Stopping."
            exit 1
        fi
        log "CUDA OOM (exit $rc). Retry $retries/$MAX_RETRIES after waiting for GPU."
        continue
    fi

    if [ "$rc" -ne 0 ]; then
        log "Inference failed with exit code $rc (not OOM). Stopping."
        exit "$rc"
    fi
    # rc == 0: loop back; make_rest_list.py confirms everything finished.
    last_clean=1
done

log "Running evaluation -> eval_secvos_full.txt"
"$PY" vos_evaluation/sav_evaluator.py \
    --gt_root data/SeCVOS/Annotations \
    --pred_root outputs/secvos_full \
    --scene_info data/SeCVOS/scene_info.json \
    --strict 2>&1 | tee eval_secvos_full.txt
log "Evaluation finished (exit ${PIPESTATUS[0]})."
