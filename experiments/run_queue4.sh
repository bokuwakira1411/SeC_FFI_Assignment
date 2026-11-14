#!/usr/bin/env bash
# Fourth batch (2026-11-14): E10 = concept burst after HSV triggers (+ sparse calls), bank on HSV triggers.
# E10chk: K=0, P=0 on 3 videos; masks must equal E0's, else stop.
set -u
cd "$(dirname "$0")/.."
q() { echo "[$(date '+%F %T')] queue4: $*" | tee -a experiments/queue.log; }
gpu_apps() { nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | tr '\n' ';'; }
q "start E10chk (GPU apps: $(gpu_apps))"
bash experiments/run_exp.sh E10chk experiments/subset_e10chk.txt experiments/gt_e10chk \
    --burst_after_trigger 0 --concept_every_sparse 0 || { q "E10chk FAILED"; exit 1; }
for v in $(cat experiments/subset_e10chk.txt); do
    diff -rq "experiments/E10chk/masks/$v" "experiments/E0/masks/$v" > /dev/null || { q "E10chk: masks differ from E0 on $v"; exit 1; }
done
q "E10chk: masks identical to E0 on $(wc -l < experiments/subset_e10chk.txt) videos"
q "start E10a (GPU apps: $(gpu_apps))"
bash experiments/run_exp.sh E10a experiments/subset40.txt experiments/gt_subset40 \
    --burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv || { q "E10a FAILED"; exit 1; }
q "start E10b (GPU apps: $(gpu_apps))"
bash experiments/run_exp.sh E10b experiments/subset40.txt experiments/gt_subset40 \
    --burst_after_trigger 3 --concept_every_sparse 0 --bank_rule hsv || { q "E10b FAILED"; exit 1; }
q "all done (E10a GPU $(cat experiments/E10a/gpu_time.txt)s, E10b GPU $(cat experiments/E10b/gpu_time.txt)s, GPU apps: $(gpu_apps))"
