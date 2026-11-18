#!/usr/bin/env bash
# E17a on all 160 videos: E10a flags + --concept_only_lost, acceptance pred IoU >= 0.78 and object score > 0
# (deviation from the design's > 1, see step0.md). Seed 0. Not in parallel with fps160.
set -u
cd "$(dirname "$0")/../.."
Q=experiments/e17/queue.log
q() { echo "[$(date '+%F %T')] $*" | tee -a "$Q"; }
while pgrep -f experiments/fps160/run_interleaved.py > /dev/null; do sleep 60; done
q "E17a start; nvidia-smi:"; nvidia-smi >> "$Q"
bash experiments/run_exp.sh e17/E17a experiments/e17/all160.txt experiments/full160/gt_all160 \
    --burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv --seed 0 \
    --concept_only_lost --concept_only_logit 0 || { q "E17a FAILED"; exit 1; }
q "E17a done"; nvidia-smi >> "$Q"
