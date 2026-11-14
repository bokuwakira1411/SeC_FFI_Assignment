# CHANGES

Local changes on top of upstream SeC (commit 0a797af). Inference only; no training.

## Earlier patches (before the inference experiments)
- `inference/modeling_sec.py`: fixed `==` → `=` in the `longsam2.1` config fallback; LLM attention set
  to `sdpa` (flash-attn not installed); when `labels is None`, `_llm_forward` skips the `lm_head` logits
  and returns only hidden states (saves ~2–3 GB VRAM).
- `inference/modeling_intern_vit.py`: SDPA attention for InternViT.
- `vos_evaluation/vos_inference.py`: `init_state(..., offload_video_to_cpu=True)`.

## 2026-11-13 — inference experiments (E0–E5)
All new options default to the original behaviour. Check: E0 with default settings reproduces the
earlier 160-video run on subset40 (same J&F/J/F on every segment; masks bit-identical on 38/40 videos).

### `inference/modeling_sec.py`
- `propagate_in_video(..., exp_ctx=None)`: new optional argument. `None` = original code path.
  The dict can contain:
  - `hsv_thresh` (default 0.35): scene-change threshold;
  - `concept_mode`: `default` | `off` (never trigger) | `always` (trigger on every tracked frame) |
    `swap` (on trigger, use `swap_concepts[k % n]` for the k-th trigger, without calling the MLLM;
    the MLLM memory update rule is unchanged);
  - `rows`: a list that receives one dict per (object, frame) with `hsv_dist, triggered,
    mllm_memory_len, n_ctx_frames, mem_appended, pred_iou` (max of `ious`), `object_score_logit`,
    `predict_forward_ms` and `frame_ms` (both timed with `torch.cuda.synchronize`);
  - `concept_dir`: saves `language_embd` (shape [1,1,256], bf16) at every trigger to
    `{video}_{obj}_{frame:05d}.pt`.
- `is_scene_change_hsv` was split: the histogram code moved unchanged into a new
  `hsv_distance(img1, img2)`; `is_scene_change_hsv(img1, img2, threshold)` returns
  `hsv_distance(...) > threshold` (same result). The trigger, keyframe/MLLM memory, fusion and SAM
  logic are unchanged.

### `vos_evaluation/vos_inference.py`
- New flags: `--hsv_thresh` (0.35), `--concept_mode` (default), `--swap_map` (json
  `{"<video>_<obj>": [concept .pt files]}`), `--exp_log_dir`, `--save_concepts`, `--skip_done`.
- `--skip_done`: skips a video whose output already has one PNG per frame for every GT object (resume).
- With `--exp_log_dir`: writes `frames/<video>.csv` (per-frame log), appends to `videos.csv`
  (`n_obj, n_frames, n_triggers, video_sec`) and `progress.txt`.

### New files
- `experiments/run_exp.sh NAME LIST GT_DIR [args]`: waits until GPU free ≥ 13 GiB (polls nvidia-smi
  every 5 min and logs the wait), runs `vos_inference.py` with `--skip_done`, retries on CUDA OOM
  (max 5), writes `gpu_time.txt`, then runs `sav_evaluator.py --scene_info --strict` → `eval.txt`.
- `experiments/subset40.txt`: 160 videos sorted by per-video J&F of the earlier 160-video run
  (`eval_secvos_full_per_video.csv`, tie-break by video id), every 4th starting at index 0.
- `experiments/gt_subset40/`: symlinks to `data/SeCVOS/Annotations/<video>` for the 40 videos.
- `experiments/prev160_subset40/masks/`: symlinks to the earlier run's masks (for comparison).

### Environment
- Installed `scenedetect==0.7.3` (+ `click`, `platformdirs`) into env `sec` for independent scene-cut
  detection. torch 2.7.1+cu128, transformers 4.42.3, numpy 1.26.4, opencv 4.11.0 unchanged.

## 2026-11-14 — bank options and extended log (E4b, E6)
### `inference/modeling_sec.py` (`exp_ctx`, default = original)
- `bank_rule`: `default` = the original rule (a frame where the concept was computed is added to the MLLM
  memory if its mask is not empty and object score > 1) | `hsv` = only frames with an HSV trigger are
  candidates, even if the concept is computed on every frame (E4b).
- `bank_extra_gap` (0 = off): non-trigger tracked frames that pass the same test are also added if they
  are at least `gap` frames after the last added frame (E6). Bank size (7), first frame kept + FIFO: unchanged.
- The bank-append block was rewritten so that it records the reason; with the defaults it evaluates the
  same condition in the same order (`mask not empty` then `object score > 1`).
- Extra log columns: `hsv_trigger`, `bank_reason` (`added` | `added_periodic` | `empty_mask` |
  `obj_score<=1`; empty when the frame was not a candidate), `mask_px`.
### `vos_evaluation/vos_inference.py`
- New flags `--bank_rule {default,hsv}` and `--bank_extra_gap N` (defaults = original).
### New files
- `experiments/make_swap_maps.py` → `swap_cross.json` (E2: every object → a random object of another
  video, `random.Random(0)`), `swap_same.json` (E2b: other object of the same video, videos with ≥2
  objects), `swap_sources.json` (which source was used); the k-th trigger uses the source's
  k mod n-th E0 concept (sorted by frame). Also `subset_short10.txt` (10 shortest videos of subset40,
  E4/E4b), `subset_multiobj.txt` (E2b) and their `gt_*` symlink dirs.
- `experiments/run_queue.sh`: runs E1, E2, E3a (0.2), E3b (0.5), E4, E2b, E4b, E6, E0r in order.
- `experiments/detect_cuts.py`: PySceneDetect ContentDetector (defaults) → `experiments/scenecuts.json`.
  Diagnostic variants with `min_scene_len` 1 and 5 → `scenecuts_msl{1,5}.json`.
- `experiments/analyze_frames.py EXP`: J per frame, incidents, cuts vs triggers, bank statistics
  → `experiments/analysis/EXP/`.

## 2026-11-14 — per-frame diagnosis of the lowest-J&F videos (D0)
Logging only; masks unchanged (check: D0 masks vs `outputs/secvos_full`).
- `inference/modeling_sec.py` (only when per-frame rows are logged): new row fields `ctx_frames`
  (frame_idx of the MLLM context used at a trigger), `bank_frames` (frame_idx in `mllm_memory` after this
  frame's update), `sam_mem_frames` (frames used by the SAM 2 pixel memory: cond frames + selected
  non-cond frames).
- `inference/sam2_video_predictor.py`: `_prepare_memory_conditioned_features` fills
  `self._exp_mem_frames` if the caller set it to a list (default `None` = nothing recorded).
- `vos_evaluation/vos_inference.py`: the three new columns in `frames/<video>.csv`.
- `experiments/run_exp.sh`: `CHECK_EVERY` can be set from the environment (default 300 s).
- `experiments/subset_diag5.txt`, `experiments/gt_diag5/`: the 5 lowest-J&F videos of the 160-video run.
- `experiments/diag_video.py` (CPU): per-frame J/F/J&F (Evaluator from `vos_evaluation/utils`),
  FN/FP flags, bank decision per frame, static timeline PNG, diagnostic mp4 (10 fps) and summary table.
- `experiments/final_report.py`: results table, `per_video_jf.csv`, `per_object_jf.csv`, E5 plot and J-per-frame
  plots → `experiments/results/`. E0 on the smaller subsets is scored through symlinked pred dirs in
  `experiments/_eval/` (E0's own CSVs are not overwritten).
- Check: E0r (default settings, new bank code + extended log) gives masks identical to E0 (0 files differ).

## 2026-11-14 — E4b on subset40, E7, E8; bootstrap CI; cost–accuracy plot
- `inference/modeling_sec.py`: `exp_ctx["concept_every"]` (0 = off, default): the concept is also computed on
  frames with `frame_idx % N == 0`; HSV triggers still apply. Bank/fusion/SAM logic unchanged.
- `vos_evaluation/vos_inference.py`: flag `--concept_every N` (default 0).
- `experiments/run_queue2.sh`: E4b40 (`--concept_mode always --bank_rule hsv`, subset40),
  E7 (`--concept_every 5 --bank_rule hsv`: bank updated only on HSV triggers, as in E0),
  E8 (`--hsv_thresh 0.2 --bank_extra_gap 10`).
- `experiments/analyze_frames.py EXP [CUTS]`: primary cut set is now `scenecuts_msl5.json`
  (ContentDetector `min_scene_len=5`) → `analysis/EXP/`; `default` (ContentDetector defaults) →
  `analysis/EXP_cutsdefault/` (sensitivity). Previous default-cut outputs kept in `analysis/old_defaultcuts/`
  (identical to the new `_cutsdefault` ones).
- `experiments/final_report.py`: ΔJ&F vs E0 on the same objects with a paired bootstrap 95% CI over objects
  (10,000 resamples, `default_rng(0)`), objects up/down; `cost_vs_accuracy.png` (J&F vs concept-call rate and
  vs fps, subset40); J-curve plot uses the msl5 cuts.

## 2026-11-14 — step-5 interventions on the snack-can video aP2up9N6H-g (I5a/b/c, K = 10)
Defaults unchanged (original behaviour).
- `inference/modeling_sec.py`: `exp_ctx` keys `oracle_keyframes` (list of (frame_idx, mask) put into the MLLM
  memory after the init frame before tracking starts), `force_trigger` (set of frame_idx where the concept
  update is forced), `bank_score_thresh` (default 1; the bank reason becomes `obj_score<=<thresh>`).
- `vos_evaluation/vos_inference.py`: `--oracle_keyframe K` (GT mask of frame K from `--input_mask_dir`),
  `--force_trigger '10-30,45'`, `--bank_score_thresh` (default 1.0).
- `experiments/diag_video.py`: `--oracle_frames`, `--bank_score_thresh`, `--after K` (J&F over frames K+1..n-2).
- `experiments/run_queue_snack.sh`: I5a `--oracle_keyframe 10`, I5b `--force_trigger 10-30`,
  I5c `--bank_score_thresh 0`; subset `experiments/subset_snack.txt`, GT `experiments/gt_snack/`.

## 2026-11-14 — E7bd; video-cluster bootstrap
- `experiments/run_queue3.sh`: E7bd = `--concept_every 5` with the original bank rule (every frame where the
  concept is computed is a bank candidate), subset40.
- `experiments/final_report.py`: video-cluster bootstrap CI for subset40 rows (resample the 40 videos, objects of a
  video kept together, object-weighted mean, 10,000 resamples, `default_rng(0)`); column "bank added (periodic)";
  E7bd vs E7 paired comparison line.
- Note: E7bd started 17:54 while another user's training job was on the GPU; its non-trigger frame time was
  143 ms (median) vs 76 ms in E0r/E7, so E7bd's fps / predict_forward / GPU-time figures are not comparable.
  Masks/J&F are unaffected by the timing.

## 2026-11-14 — E10: concept burst after HSV triggers, sparse calls elsewhere
Defaults unchanged (original behaviour). Check: E10chk (`--burst_after_trigger 0 --concept_every_sparse 0`,
3 videos of `subset_short10`: `experiments/subset_e10chk.txt`, `experiments/gt_e10chk/`) → masks identical to E0.
- `inference/modeling_sec.py`: `exp_ctx` keys `burst_after_trigger` (K, 0 = off): the concept is also computed on
  the K tracked frames after each HSV trigger (a trigger during a burst restarts the count);
  `concept_every_sparse` (P, 0 = off): outside a burst, the concept is also computed when the last concept call
  (or the start frame) is ≥ P frames back. The bank rule is not touched (E10 runs with `--bank_rule hsv`, so
  burst/sparse calls never add frames to the MLLM memory). Log column `concept_reason` on concept-call frames:
  `hsv` | `burst` | `sparse` | `other` (concept_every / force_trigger / concept_mode always/swap).
- `vos_evaluation/vos_inference.py`: flags `--burst_after_trigger K`, `--concept_every_sparse P` (default 0);
  `concept_reason` column in `frames/<video>.csv`.
- `experiments/sim_burst.py` (CPU): concept-call rate for K ∈ {3,5,10} × P ∈ {10,15,20,0} simulated from E0r's
  HSV triggers; E10a = closest to E7 (25.65 %) → K=3, P=15; E10b = closest to E3a/E8 (17.14 %) → K=3, P=0.
- `experiments/run_queue4.sh`: E10chk (stops if masks differ from E0), E10a, E10b (subset40, `--bank_rule hsv`).
- `experiments/final_report.py`: rows E10a/E10b; column "Δ vs E7 / E8" (E10a vs E7, E10b vs E8; object and
  video-cluster bootstrap, up/down); column "trigger %" renamed "% frame gọi concept"; incident table
  (E0, E7, E8, E10a, E10b); E10 points in `cost_vs_accuracy.png`.

## 2026-11-14 — E11 stage 1: offline uncertainty-trigger simulation (CPU only, no model code changed)
- `experiments/sim_uncertainty.py`: simulates S1 (logit drop > D vs mean of previous 5 frames), S2 (area drop > r),
  S3 (empty mask N frames), cooldown C, modes `unc` / `hsv_or_unc`, on E0r's logged `object_score_logit` / `mask_px`
  (decision for frame f uses frames < f). Grid D ∈ {-,2,4,6} × r ∈ {-,0.5,0.8} × N ∈ {-,1,3} × C ∈ {5,10,20}
  → `experiments/analysis/E11sim/{grid.csv,summary.md}`. No configuration met the GPU criterion; stage 2 not run.
- Criterion (c) (spurious % not above HSV) dropped by the user after stage 1: the cost of spurious calls is
  already covered by criterion (b) (concept-call rate ≤ 17 %), and "spurious" is only an approximation
  (incident window / ±2 cut matching). With (a)+(b) only, E11a = hsv_or_unc, D=2, r=0.5, N=1, C=5
  (sim: 14.34 % calls, 19/23 HSV-missed incidents caught). E11b (unc only) not run, by request.

## 2026-11-14 — E11a: uncertainty trigger (HSV OR uncertainty), bank on HSV triggers
Defaults unchanged (original behaviour). Check: E11chk (`--trigger_mode hsv`, `subset_e10chk`) → masks vs E0.
- `inference/modeling_sec.py` (`propagate_in_video`): `exp_ctx` keys `trigger_mode` ('hsv' default | 'unc' |
  'hsv_or_unc'), `unc_logit_drop` D, `unc_area_drop` r, `unc_empty_n` N, `unc_window` w (5), `unc_cooldown` C.
  The uncertainty decision for frame f uses the object score logit / mask area of frames < f (same rule as
  `experiments/sim_uncertainty.py`); an uncertainty call is skipped if the last concept call is ≤ C frames back.
  'unc' ignores HSV for the concept call; HSV still drives `--bank_rule hsv`. `concept_reason` is
  `unc_empty` | `unc_logit` | `unc_area` for these calls. History is only kept when trigger_mode != 'hsv'.
- `vos_evaluation/vos_inference.py`: flags `--trigger_mode`, `--unc_logit_drop`, `--unc_area_drop`, `--unc_empty_n`,
  `--unc_window`, `--unc_cooldown`.
- `experiments/run_queue5.sh`: waits for queue4 (E10), E11chk, E11a (`--save_concepts` for concept cosines).
- Check: E11chk (`--trigger_mode hsv`, 3 videos of `subset_e10chk`) → masks identical to E0 (0 files differ).
- `experiments/final_report.py`: row E11a; same-cost reference E11a → E3a (column renamed); E11 point (green) in
  `cost_vs_accuracy.png`; E11a in the incident table.
- `experiments/analyze_e11.py` (CPU): real vs simulated recall on E0's 102 incidents (v1), recovery within 10 frames
  on E0's incidents (E0 vs E11a J curves), cosine between consecutive concepts by call reason → `analysis/E11a/e11.md`.
- Pre-change copies for diffing: `experiments/E11_orig_modeling_sec.py.bak`, `experiments/E11_orig_vos_inference.py.bak`.
