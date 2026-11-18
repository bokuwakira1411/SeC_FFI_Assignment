#!/usr/bin/env bash
# E17c: offline sweep of w in f_w = (1-w)*memory + w*concept on lost calls (trajectory and output stay E10a; numbers come from the frame logs).
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); L=$H/../tools/lists
bash $H/run_one.sh E17c_sweep ${LIST:-$L/subset40.txt} --burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv --seed 0 --concept_only_lost --concept_only_logit 0 --fusion_sweep 0.5,0.6,0.7,0.8,0.9,1.0 --log_feat_norms
