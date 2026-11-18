#!/usr/bin/env bash
# E13b (ORACLE, uses GT): E10a + a frame with its GT mask added to the bank every >= 10 frames. subset40 only.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); L=$H/../tools/lists
bash $H/run_one.sh E13b ${LIST:-$L/subset40.txt} --burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv --bank_gt_gap 10 --seed 0
