#!/usr/bin/env bash
# E18 step 0 (GPU, 3 smoke videos, seed 0): --e17_commit full / output / pixel. Equivalence checks:
#  full   -> masks bit-identical to E17a
#  output -> per-frame logs (state) compared to E10a by experiments/e18/check_step0.py; masks of non-accepted frames
#  pixel  -> smoke run only (no reference)
set -u
cd "$(dirname "$0")/../.."
Q=experiments/e18/queue.log
q() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }
while pgrep -f "bin/python vos_evaluation/vos_inference" > /dev/null; do sleep 60; done
E17A="--burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv --seed 0 --concept_only_lost --concept_only_logit 0"
SMOKE=experiments/e17/subset_smoke.txt; GTS=experiments/e17/gt_smoke
for c in full output pixel; do
    q "chk_$c start; nvidia-smi:"; nvidia-smi >> "$Q"
    bash experiments/run_exp.sh e18/chk_$c $SMOKE $GTS $E17A --e17_commit $c || { q "chk_$c FAILED"; exit 1; }
done
n=0; for v in $(cat $SMOKE); do for f in $(cd experiments/e17/E17a/masks && find "$v" -name '*.png'); do
    cmp -s experiments/e17/E17a/masks/$f experiments/e18/chk_full/masks/$f || n=$((n+1)); done; done
q "chk_full: $n mask files differ from E17a"
q "step 0 runs done"
