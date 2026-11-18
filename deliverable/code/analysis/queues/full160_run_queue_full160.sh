#!/usr/bin/env bash
# 160-video run (2026-11-15): E10a and E12 on the 120 videos outside subset40; subset40 masks are reused
# from experiments/E10a and experiments/bank/E12 (pipeline is deterministic; chk re-runs E10a on 3 videos of
# subset40 with the current code and stops if masks differ). Seed 0. Outputs in experiments/full160/.
set -u
cd "$(dirname "$0")/../.."
E10A="--burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv"
R120="experiments/full160/rest120.txt experiments/full160/gt_rest120"
q() { echo "[$(date '+%F %T')] full160: $*" | tee -a experiments/full160/queue.log; }
gpu_apps() { nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | tr '\n' ';'; }
run() {
    local n=$1; shift
    q "start $n (GPU apps: $(gpu_apps))"
    nvidia-smi >> experiments/full160/queue.log
    bash experiments/run_exp.sh "full160/$n" "$@" --seed 0 || { q "$n FAILED"; exit 1; }
    q "$n done (GPU apps: $(gpu_apps))"
}
run chk_E10a experiments/subset_e10chk.txt experiments/gt_e10chk $E10A
for v in $(cat experiments/subset_e10chk.txt); do
    diff -rq "experiments/full160/chk_E10a/masks/$v" "experiments/E10a/masks/$v" > /dev/null \
        || { q "chk_E10a: masks differ from E10a on $v"; exit 1; }
done
q "chk_E10a: masks identical to E10a on 3 videos"
run E10a_rest120 $R120 $E10A
run E12_rest120 $R120 $E10A --bank_extra_gap 10
q "all runs done"
