#!/usr/bin/env bash
# Keyframe-bank batch (E12, E13, E14), subset40, outputs in experiments/bank/. Seed 0 everywhere.
# chk: new code, all new flags off, 3 videos -> masks must equal E0's, else stop.
# smoke13 / smoke14: 1 video, new flags on (crash test only, not reported).
# tau for E14: experiments/bank/sim_diverse.py on E10a's logs (no J&F), target = E6's 480 added frames.
# Usage (from SeC/): bash experiments/bank/run_queue_bank.sh [--with-e13b]
set -u
cd "$(dirname "$0")/../.."
PY=${PYTHON:-python}
S40="experiments/subset40.txt experiments/gt_subset40"
E10A="--burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv"
q() { echo "[$(date '+%F %T')] bank: $*" | tee -a experiments/bank/queue.log; }
gpu_apps() { nvidia-smi --query-compute-apps=pid,used_memory,process_name --format=csv,noheader | tr '\n' ';'; }
run() {  # NAME LIST GT args...
    local n=$1; shift
    q "start $n (GPU apps: $(gpu_apps))"
    nvidia-smi >> experiments/bank/queue.log
    bash experiments/run_exp.sh "bank/$n" "$@" --seed 0 || { q "$n FAILED"; exit 1; }
    q "$n done (GPU apps: $(gpu_apps))"
}

run chk experiments/subset_e10chk.txt experiments/gt_e10chk
for v in $(cat experiments/subset_e10chk.txt); do
    diff -rq "experiments/bank/chk/masks/$v" "experiments/E0/masks/$v" > /dev/null \
        || { q "chk: masks differ from E0 on $v"; exit 1; }
done
q "chk: masks identical to E0 on $(wc -l < experiments/subset_e10chk.txt) videos"

mkdir -p experiments/bank/gt_smoke
ln -sfn "$PWD/data/SeCVOS/Annotations/OFVRaOdgV1Y" experiments/bank/gt_smoke/OFVRaOdgV1Y
echo OFVRaOdgV1Y > experiments/bank/subset_smoke.txt
run smoke13 experiments/bank/subset_smoke.txt experiments/bank/gt_smoke --bank_gt_gap 10
run smoke14 experiments/bank/subset_smoke.txt experiments/bank/gt_smoke $E10A --bank_diverse_tau 0.9

q "descriptors GT E10a (GPU apps: $(gpu_apps))"
$PY experiments/bank/descriptors.py GT E10a >> experiments/bank/queue.log 2>&1 || { q "descriptors FAILED"; exit 1; }
$PY experiments/bank/sim_diverse.py 480 >> experiments/bank/queue.log 2>&1 || { q "sim_diverse FAILED"; exit 1; }
TAU=$($PY -c "import json; print(json.load(open('experiments/bank/tau.json'))['tau'])")
q "E14 tau = $TAU"

run E13a $S40 --bank_gt_gap 10
run E12 $S40 $E10A --bank_extra_gap 10
run E14 $S40 $E10A --bank_diverse_tau "$TAU"
if [ "${1:-}" = "--with-e13b" ]; then
    run E13b $S40 $E10A --bank_gt_gap 10
fi
q "all runs done"
