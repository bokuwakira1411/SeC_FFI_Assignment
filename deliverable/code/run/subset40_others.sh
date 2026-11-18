#!/usr/bin/env bash
# The other subset40 runs (design phase). One line per experiment; run the ones you need.
set -euo pipefail
H=$(cd "$(dirname "$0")" && pwd); L=$H/../tools/lists; R="bash $H/run_one.sh"
E10A="--burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv"
case "${1:-}" in
  E1)    $R E1    $L/subset40.txt --concept_mode off ;;                                   # concept off
  E2)    $R E2    $L/subset40.txt --concept_mode swap --swap_map $L/swap_cross.json ;;   # concept of an object in another video
  E2b)   $R E2b   $L/subset_multiobj.txt --concept_mode swap --swap_map $L/swap_same.json ;; # other object, same video
  E3a)   $R E3a   $L/subset40.txt --hsv_thresh 0.2 ;;                                    # lower HSV threshold
  E7)    $R E7    $L/subset40.txt --concept_every 5 --bank_rule hsv ;;                   # extra call every 5 frames
  E4b40) $R E4b40 $L/subset40.txt --concept_mode always --bank_rule hsv ;;               # concept on every frame
  E11a)  $R E11a  $L/subset40.txt --trigger_mode hsv_or_unc --unc_logit_drop 2 --unc_area_drop 0.5 --unc_empty_n 1 --unc_cooldown 5 ;;
  E6)    $R E6    $L/subset40.txt --bank_extra_gap 10 ;;                                 # E0 + periodic bank
  E13a)  $R E13a  $L/subset40.txt --bank_gt_gap 10 --seed 0 ;;                           # ORACLE: E0 + GT bank
  E14)   $R E14   $L/subset40.txt $E10A --bank_diverse_tau 0.97 --seed 0 ;;             # bank by diversity
  *) echo "usage: $0 {E1|E2|E2b|E3a|E7|E4b40|E11a|E6|E13a|E14}"; exit 1 ;;
esac
