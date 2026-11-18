#!/usr/bin/env bash
# Step 5 interventions on the snack-can video aP2up9N6H-g (K = 10), after the D0 diagnosis.
set -u
cd "$(dirname "$0")/.."
R=experiments/run_exp.sh
S="experiments/subset_snack.txt experiments/gt_snack"
export CHECK_EVERY=30

run() {
    echo "[$(date '+%F %T')] snack queue: start $1" | tee -a experiments/queue_snack.log
    bash $R "$@" || { echo "[$(date '+%F %T')] snack queue: $1 FAILED, stopping" | tee -a experiments/queue_snack.log; exit 1; }
    echo "[$(date '+%F %T')] snack queue: done $1" | tee -a experiments/queue_snack.log
}

run I5a $S --oracle_keyframe 10        # (a) frame 10 + GT mask in the MLLM memory from the start
run I5b $S --force_trigger 10-30       # (b) forced trigger on frames 10..30
run I5c $S --bank_score_thresh 0       # (c) bank object-score threshold 1 -> 0
echo "[$(date '+%F %T')] snack queue: all done" | tee -a experiments/queue_snack.log
