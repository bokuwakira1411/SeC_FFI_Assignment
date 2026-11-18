#!/usr/bin/env bash
# E17 step 0 (GPU): waits for fps160 (run_interleaved.py) to exit, then
#  chk   = E10a flags, all E17 flags off, subset_e10chk (3 videos) -> masks vs experiments/E10a
#  norms = E10a + --log_feat_norms + --concept_only_lost, 3 worst E0 videos (smoke test + norms of a, b, (a+b)/2)
# Seed 0. Outputs in experiments/e17/. Usage (from SeC/): bash experiments/e17/run_queue_e17_step0.sh
set -u
cd "$(dirname "$0")/../.."
Q=experiments/e17/queue.log
q() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }
E10A="--burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv --seed 0"
q "waiting for fps160 to finish"
while pgrep -f experiments/fps160/run_interleaved.py > /dev/null; do sleep 60; done
q "fps160 finished; nvidia-smi:"; nvidia-smi >> "$Q"
bash experiments/run_exp.sh e17/chk experiments/subset_e10chk.txt experiments/gt_e10chk $E10A \
    || { q "chk FAILED"; exit 1; }
n=0; for v in $(cat experiments/subset_e10chk.txt); do
    for f in $(cd experiments/E10a/masks && find "$v" -name '*.png'); do
        cmp -s experiments/E10a/masks/$f experiments/e17/chk/masks/$f || n=$((n+1)); done; done
q "chk: $n mask files differ from experiments/E10a"
nvidia-smi >> "$Q"
bash experiments/run_exp.sh e17/norms experiments/e17/subset_smoke.txt experiments/e17/gt_smoke $E10A \
    --log_feat_norms --concept_only_lost || { q "norms FAILED"; exit 1; }
q "step 0 done"
