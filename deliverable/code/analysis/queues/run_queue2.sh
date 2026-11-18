#!/usr/bin/env bash
# Second batch (2026-11-14): E4b on subset40, E7 (concept every 5 frames), E8 (E6 + HSV 0.2).
set -u
cd "$(dirname "$0")/.."
R=experiments/run_exp.sh
S40="experiments/subset40.txt experiments/gt_subset40"

run() {
    echo "[$(date '+%F %T')] queue: start $1" | tee -a experiments/queue.log
    bash $R "$@" || { echo "[$(date '+%F %T')] queue: $1 FAILED, stopping" | tee -a experiments/queue.log; exit 1; }
    echo "[$(date '+%F %T')] queue: done $1 (GPU $(cat experiments/$1/gpu_time.txt)s)" | tee -a experiments/queue.log
}

run E4b40 $S40 --concept_mode always --bank_rule hsv
run E7    $S40 --concept_every 5 --bank_rule hsv
run E8    $S40 --hsv_thresh 0.2 --bank_extra_gap 10
echo "[$(date '+%F %T')] queue2: all done" | tee -a experiments/queue.log
