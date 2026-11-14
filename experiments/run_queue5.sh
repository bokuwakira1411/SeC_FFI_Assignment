#!/usr/bin/env bash
# Fifth batch (2026-11-14): E11a = HSV OR uncertainty trigger (D=2, r=0.5, N=1, C=5), bank on HSV triggers.
# Waits for queue4 (E10) to finish. E11chk: default trigger settings on 3 videos; masks must equal E0's, else stop.
set -u
cd "$(dirname "$0")/.."
q() { echo "[$(date '+%F %T')] queue5: $*" | tee -a experiments/queue.log; }
gpu_apps() { nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | tr '\n' ';'; }
while pgrep -f run_queue4.sh > /dev/null; do sleep 60; done
q "start E11chk (GPU apps: $(gpu_apps))"
bash experiments/run_exp.sh E11chk experiments/subset_e10chk.txt experiments/gt_e10chk \
    --trigger_mode hsv || { q "E11chk FAILED"; exit 1; }
for v in $(cat experiments/subset_e10chk.txt); do
    diff -rq "experiments/E11chk/masks/$v" "experiments/E0/masks/$v" > /dev/null || { q "E11chk: masks differ from E0 on $v"; exit 1; }
done
q "E11chk: masks identical to E0 on $(wc -l < experiments/subset_e10chk.txt) videos"
q "start E11a (GPU apps: $(gpu_apps))"
bash experiments/run_exp.sh E11a experiments/subset40.txt experiments/gt_subset40 \
    --trigger_mode hsv_or_unc --unc_logit_drop 2 --unc_area_drop 0.5 --unc_empty_n 1 --unc_cooldown 5 \
    --bank_rule hsv --save_concepts || { q "E11a FAILED"; exit 1; }
q "all done (E11a GPU $(cat experiments/E11a/gpu_time.txt)s, GPU apps: $(gpu_apps))"
