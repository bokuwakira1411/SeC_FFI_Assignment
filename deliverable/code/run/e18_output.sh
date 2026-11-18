#!/usr/bin/env bash
# E18 output: as E17a, but a kept concept mask only replaces the scored output; tracking state as E10a.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); L=$H/../tools/lists
bash $H/run_one.sh E18_output ${LIST:-$L/all160.txt} --burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv --seed 0 --concept_only_lost --concept_only_logit 0 --e17_commit output
