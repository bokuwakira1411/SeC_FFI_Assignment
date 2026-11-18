#!/usr/bin/env bash
# E17a: E10a + when lost, keep the concept-branch-only mask if pred IoU >= 0.78 and logit > 0 (output + pixel memory + bank).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); L=$H/../tools/lists
bash $H/run_one.sh E17a ${LIST:-$L/all160.txt} --burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv --seed 0 --concept_only_lost --concept_only_logit 0
