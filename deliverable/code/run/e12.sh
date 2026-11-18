#!/usr/bin/env bash
# E12: E10a + bank also updated every >= 10 frames with the predicted mask (E6 rule).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); L=$H/../tools/lists
bash $H/run_one.sh E12 ${LIST:-$L/all160.txt} --burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv --bank_extra_gap 10 --seed 0
