#!/usr/bin/env bash
# E10a: concept burst (3 frames) after each HSV trigger + a call when >= 15 frames without one; bank only on HSV triggers.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); L=$H/../tools/lists
bash $H/run_one.sh E10a ${LIST:-$L/all160.txt} --burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv --seed 0
