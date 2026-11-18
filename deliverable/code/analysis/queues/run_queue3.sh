#!/usr/bin/env bash
# Third batch (2026-11-14): E7 with the original bank rule (concept every 5 frames; every concept frame is a bank candidate).
set -u
cd "$(dirname "$0")/.."
echo "[$(date '+%F %T')] queue: start E7bd" | tee -a experiments/queue.log
bash experiments/run_exp.sh E7bd experiments/subset40.txt experiments/gt_subset40 --concept_every 5 \
    || { echo "[$(date '+%F %T')] queue: E7bd FAILED" | tee -a experiments/queue.log; exit 1; }
echo "[$(date '+%F %T')] queue3: all done (E7bd GPU $(cat experiments/E7bd/gpu_time.txt)s)" | tee -a experiments/queue.log
