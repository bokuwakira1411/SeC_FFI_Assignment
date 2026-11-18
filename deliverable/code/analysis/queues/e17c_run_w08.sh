#!/usr/bin/env bash
# E17c sensitivity run (deviation from the pre-set rule, asked by the user after stop 1): w = 0.8 on all160.
# E17a flags + --fusion_w 0.8. Seed 0. Not in parallel with other GPU runs.
set -u
cd "$(dirname "$0")/../.."
Q=experiments/e17c/queue.log
q() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }
while pgrep -f "bin/python vos_evaluation/vos_inference" > /dev/null; do sleep 60; done
q "E17c_w08 start; nvidia-smi:"; nvidia-smi >> "$Q"
bash experiments/run_exp.sh e17c/E17c_w08 experiments/e17/all160.txt experiments/full160/gt_all160 \
    --burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv --seed 0 \
    --concept_only_lost --concept_only_logit 0 --fusion_w 0.8 || { q "E17c_w08 FAILED"; exit 1; }
q "E17c_w08 done"; nvidia-smi >> "$Q"
