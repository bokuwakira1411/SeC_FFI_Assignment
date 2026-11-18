#!/usr/bin/env bash
# Run the approved inference experiments in order; stop at the first failure.
# Each step waits for >= 13 GiB free GPU memory and resumes per video (see run_exp.sh).
set -u
cd "$(dirname "$0")/.."
R=experiments/run_exp.sh
S40="experiments/subset40.txt experiments/gt_subset40"
S10="experiments/subset_short10.txt experiments/gt_subset_short10"
SMO="experiments/subset_multiobj.txt experiments/gt_subset_multiobj"

run() {
    echo "[$(date '+%F %T')] queue: start $1" | tee -a experiments/queue.log
    bash $R "$@" || { echo "[$(date '+%F %T')] queue: $1 FAILED, stopping" | tee -a experiments/queue.log; exit 1; }
    echo "[$(date '+%F %T')] queue: done $1 (GPU $(cat experiments/$1/gpu_time.txt)s)" | tee -a experiments/queue.log
}

run E1   $S40 --concept_mode off
run E2   $S40 --concept_mode swap --swap_map experiments/swap_cross.json
run E3a  $S40 --hsv_thresh 0.2
run E3b  $S40 --hsv_thresh 0.5
run E4   $S10 --concept_mode always
run E2b  $SMO --concept_mode swap --swap_map experiments/swap_same.json
run E4b  $S10 --concept_mode always --bank_rule hsv
run E6   $S40 --bank_extra_gap 10
# E0 again with the extended per-frame log (bank_reason, mask_px); must reproduce E0's masks
run E0r  $S40
echo "[$(date '+%F %T')] queue: all done" | tee -a experiments/queue.log
