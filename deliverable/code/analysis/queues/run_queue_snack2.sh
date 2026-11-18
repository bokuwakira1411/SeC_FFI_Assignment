#!/usr/bin/env bash
# Snack-can video aP2up9N6H-g: I5b re-run with the current code (I5b_r) and (b') I5bp = forced concept calls on
# frames 10..30 but bank updated only on HSV triggers. Concepts saved for cosines.
set -u
cd "$(dirname "$0")/.."
R=experiments/run_exp.sh
S="experiments/subset_snack.txt experiments/gt_snack"
export CHECK_EVERY=30
run() {
    echo "[$(date '+%F %T')] snack2: start $1" | tee -a experiments/queue_snack.log
    bash $R "$@" || { echo "[$(date '+%F %T')] snack2: $1 FAILED" | tee -a experiments/queue_snack.log; exit 1; }
}
run I5b_r $S --force_trigger 10-30 --save_concepts
run I5bp $S --force_trigger 10-30 --bank_rule hsv --save_concepts
echo "[$(date '+%F %T')] snack2: all done" | tee -a experiments/queue_snack.log
