#!/usr/bin/env bash
# E18 pixel: as E17a, but a kept concept mask is not added to the MLLM bank directly (output + pixel memory only).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); L=$H/../tools/lists
bash $H/run_one.sh E18_pixel ${LIST:-$L/all160.txt} --burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv --seed 0 --concept_only_lost --concept_only_logit 0 --e17_commit pixel
