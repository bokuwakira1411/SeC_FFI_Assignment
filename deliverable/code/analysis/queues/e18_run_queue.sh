#!/usr/bin/env bash
# E18 GPU queue (one run at a time, seed 0), approved at stop 1:
#  0. chk_e18b: E18b on the 3 smoke videos (crash test only)
#  1. E18a_output, E18a_pixel on all160 (step 2)
#  2. bank gate (bank_gate.py): stop if the bank part (E17a - pixel, rest120) < 0.5 and its CIs contain 0
#  3. E18b on all160: --e17_commit pixel --bank_confirm_k 3
# Usage (from SeC/): bash experiments/e18/run_queue.sh
set -u
cd "$(dirname "$0")/../.."
Q=experiments/e18/queue.log
q() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }
while pgrep -f "bin/python vos_evaluation/vos_inference" > /dev/null; do sleep 60; done
E17A="--burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv --seed 0 --concept_only_lost --concept_only_logit 0"
ALL=experiments/e17/all160.txt; GT=experiments/full160/gt_all160
E18B_OK=1
q "chk_e18b start; nvidia-smi:"; nvidia-smi >> "$Q"
bash experiments/run_exp.sh e18/chk_e18b experiments/e17/subset_smoke.txt experiments/e17/gt_smoke $E17A \
    --e17_commit pixel --bank_confirm_k 3 || { q "chk_e18b FAILED (E18b will not run)"; E18B_OK=0; }
for c in output pixel; do
    q "E18a_$c start; nvidia-smi:"; nvidia-smi >> "$Q"
    bash experiments/run_exp.sh e18/E18a_$c $ALL $GT $E17A --e17_commit $c || { q "E18a_$c FAILED"; exit 1; }
    q "E18a_$c done"
done
python3 experiments/e18/bank_gate.py 2>&1 | tee -a "$Q"
gate=${PIPESTATUS[0]}
if [ "$gate" -eq 3 ]; then q "bank part ~0: E18b not run (rule set at stop 1)"; exit 0; fi
[ "$gate" -eq 0 ] || { q "bank gate FAILED (exit $gate)"; exit 1; }
[ "$E18B_OK" -eq 1 ] || exit 1
q "E18b start; nvidia-smi:"; nvidia-smi >> "$Q"
bash experiments/run_exp.sh e18/E18b $ALL $GT $E17A --e17_commit pixel --bank_confirm_k 3 \
    || { q "E18b FAILED"; exit 1; }
q "E18b done"; nvidia-smi >> "$Q"
