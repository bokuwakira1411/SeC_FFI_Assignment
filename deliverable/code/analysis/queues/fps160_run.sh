#!/usr/bin/env bash
# fps160: interleaved E0 / E10a / E14 on 160 videos (run_interleaved.py), OOM retries, then evaluation.
set -u
cd "$(dirname "$0")/../.."
OUT=experiments/fps160
PY=${PYTHON:-python}
export PYTHONPATH=$PWD
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
log() { echo "[$(date '+%F %T')] fps160: $*" | tee -a $OUT/queue.log; }

while true; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -n1 | tr -d ' ')
    [ "${free:-0}" -ge 13312 ] && break
    log "GPU free ${free} MiB < 13312, waiting 300s"; sleep 300
done
retries=0
while true; do
    start_line=$(($(wc -l < $OUT/run.log 2>/dev/null || echo 0) + 1))
    log "start inference (GPU apps: $(nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | tr '\n' ';'))"
    t0=$(date +%s)
    "$PY" $OUT/run_interleaved.py 2>&1 | tee -a $OUT/run.log
    rc=${PIPESTATUS[0]}
    log "inference exit $rc after $(( $(date +%s) - t0 ))s"
    [ "$rc" -eq 0 ] && break
    if tail -n +"$start_line" $OUT/run.log | grep -qE "CUDA out of memory|OutOfMemoryError" && [ $retries -lt 5 ]; then
        retries=$((retries + 1)); log "OOM, retry $retries/5"; continue
    fi
    log "FAILED"; exit 1
done
for r in E0 E10a E14; do
    "$PY" vos_evaluation/sav_evaluator.py --gt_root experiments/full160/gt_all160 --pred_root $OUT/$r/masks \
        --scene_info data/SeCVOS/scene_info.json --strict > $OUT/$r/eval.txt 2>&1
    log "$r eval: $(grep -m1 -A1 'overall' $OUT/$r/eval.txt | tail -1)"
done
log "all done"
