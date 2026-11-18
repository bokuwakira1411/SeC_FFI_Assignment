#!/usr/bin/env bash
# E17c step 0 + step 1 (GPU). Waits for the E17a queue (run_queue_e17a.sh) to exit; never runs in parallel.
#  chk05 = E17a flags + --fusion_w 0.5, subset_smoke (3 videos)  -> masks must equal E10a (fps160/E10a)
#  chk10 = E17a flags + --fusion_w 1.0, subset_smoke             -> masks vs E17a (e17/E17a)
#  sweep_smoke = E10a trajectory + --fusion_sweep + --log_feat_norms, subset_smoke -> masks must equal E10a; norms
#  sweep = same on subset40 (step 1)                             -> masks must equal E10a
# Seed 0. Usage (from SeC/): bash experiments/e17c/run_queue_step01.sh
set -u
cd "$(dirname "$0")/../.."
Q=experiments/e17c/queue.log
q() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }
E17A="--burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv --seed 0 --concept_only_lost --concept_only_logit 0"
SW="--fusion_sweep 0.5,0.6,0.7,0.8,0.9,1.0 --log_feat_norms"
SMOKE=experiments/e17/subset_smoke.txt; GTS=experiments/e17/gt_smoke
ndiff() {  # ndiff RUN_MASKS REF_MASKS LIST -> number of differing / missing png files (over REF files of LIST)
    local n=0; for v in $(cat "$3"); do for f in $(cd "$2" && find "$v" -name '*.png'); do
        cmp -s "$2/$f" "$1/$f" || n=$((n+1)); done; done; echo $n; }
q "waiting for E17a queue"
while pgrep -f run_queue_e17a.sh > /dev/null; do sleep 60; done
q "E17a queue finished; nvidia-smi:"; nvidia-smi >> "$Q"
bash experiments/run_exp.sh e17c/chk05 $SMOKE $GTS $E17A --fusion_w 0.5 || { q "chk05 FAILED"; exit 1; }
n=$(ndiff experiments/e17c/chk05/masks experiments/fps160/E10a/masks $SMOKE); q "chk05: $n mask files differ from E10a"
[ "$n" -eq 0 ] || { q "chk05 not equal to E10a: stop"; exit 1; }
nvidia-smi >> "$Q"
bash experiments/run_exp.sh e17c/chk10 $SMOKE $GTS $E17A --fusion_w 1.0 || { q "chk10 FAILED"; exit 1; }
q "chk10: $(ndiff experiments/e17c/chk10/masks experiments/e17/E17a/masks $SMOKE) mask files differ from E17a"
nvidia-smi >> "$Q"
bash experiments/run_exp.sh e17c/sweep_smoke $SMOKE $GTS $E17A $SW || { q "sweep_smoke FAILED"; exit 1; }
n=$(ndiff experiments/e17c/sweep_smoke/masks experiments/fps160/E10a/masks $SMOKE); q "sweep_smoke: $n mask files differ from E10a"
[ "$n" -eq 0 ] || { q "sweep_smoke not equal to E10a: stop"; exit 1; }
nvidia-smi >> "$Q"
bash experiments/run_exp.sh e17c/sweep experiments/subset40.txt experiments/gt_subset40 $E17A $SW \
    || { q "sweep FAILED"; exit 1; }
q "sweep: $(ndiff experiments/e17c/sweep/masks experiments/fps160/E10a/masks experiments/subset40.txt) mask files differ from E10a"
nvidia-smi >> "$Q"
q "step 0 + 1 done"
