#!/usr/bin/env bash
# E0: original SeC (all flags at default). Masks on 160 videos: experiments/fps160/E0 (seed 0).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); L=$H/../tools/lists
bash $H/run_one.sh E0 ${LIST:-$L/all160.txt} --seed 0
