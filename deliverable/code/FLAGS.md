# Flags added to `vos_evaluation/vos_inference.py`

All flags default to the original SeC behaviour. With every new flag at its default, the masks are
bit-identical to the original code (checked several times: E0r, E10chk, E11chk, bank/chk, full160/chk_E10a, e17/chk, e18/chk_full).
Inside the model the flags become keys of `exp_ctx` in `SeCModel.propagate_in_video` (`inference/modeling_sec.py`),
plus `_exp_e17` in `_track_step` (`inference/sam2_video_predictor.py`).

Terms: *concept* = the 256-d vector the MLLM produces from the keyframe bank. *Bank* = `mllm_memory`, the keyframes
(image + mask) given to the MLLM (max 7: frame 0 + FIFO). *HSV trigger* = colour-histogram scene-change test, the original
rule for calling the MLLM. *Lost* = mask empty or object score logit ≤ 0 on the last `lost_n` tracked frames.

## Concept-call schedule

| Flag | Default | Meaning | Used by |
|---|---|---|---|
| `--hsv_thresh` | 0.35 | HSV scene-change threshold | E3a (0.2), E3b (0.5), E8 |
| `--concept_mode` | `default` | `off`: never call; `always`: every frame; `swap`: use concepts from `--swap_map` instead of the MLLM | E1 (off), E2/E2b (swap), E4/E4b/E4b40 (always) |
| `--swap_map` | none | json `{"<video>_<obj>": [concept .pt files]}` (files come from an E0 run with `--save_concepts`) | E2, E2b |
| `--concept_every N` | 0 (off) | also call on frames with `frame_idx % N == 0` | E7, E7bd |
| `--burst_after_trigger K` | 0 (off) | also call on the K tracked frames after each HSV trigger | **E10a** (3), E10b, all later runs |
| `--concept_every_sparse P` | 0 (off) | outside bursts, also call when the last call is ≥ P frames back | **E10a** (15), all later runs |
| `--trigger_mode` | `hsv` | `unc` / `hsv_or_unc`: call on an uncertainty signal | E11a (`hsv_or_unc`) |
| `--unc_logit_drop D`, `--unc_area_drop r`, `--unc_empty_n N`, `--unc_window w`, `--unc_cooldown C` | off, off, 0, 5, 0 | uncertainty signals (logit drop, area drop, empty mask) and cooldown | E11a (D=2, r=0.5, N=1, C=5) |
| `--force_trigger` | "" | force a concept call on these frames, e.g. `10-30,45` | I5b (snack-can video) |

## Bank (MLLM memory) rules

| Flag | Default | Meaning | Used by |
|---|---|---|---|
| `--bank_rule` | `default` | `hsv`: only HSV-trigger frames can enter the bank, even if the concept is called more often | E4b, E7, **E10a** and all later runs |
| `--bank_extra_gap N` | 0 (off) | also add trusted non-trigger frames (mask not empty, logit > 1) every ≥ N frames | E6, E8, **E12** (10) |
| `--bank_score_thresh` | 1.0 | logit a frame needs to enter the bank | I5c (0) |
| `--bank_gt_gap N` | 0 (off) | **oracle**: add a frame **with its GT mask** every ≥ N frames | **E13a**, **E13b** (10) |
| `--oracle_keyframe K` | none | **oracle**: put frame K with its GT mask in the bank from the start | I5a |
| `--bank_diverse_tau τ`, `--bank_diverse_gap` | off, 5 | bank by diversity: add a candidate if its max cosine (Hiera mask descriptor) to the bank < τ | **E14** (τ = 0.97) |
| `--bank_confirm_k k` | 0 (off) | (with `--e17_commit pixel`) add an accepted concept frame to the bank only after k confirming frames | E18b — implemented, **not run** |

## Concept branch alone when lost (E17, E18)

| Flag | Default | Meaning | Used by |
|---|---|---|---|
| `--concept_only_lost` | off | on a concept call while lost, also run the SAM decoder on the concept branch alone; keep that mask if it passes the gate | **E17a**, E17c, **E18** |
| `--lost_n` | 2 | frames used by the "lost" test | E17a (2) |
| `--concept_only_iou` | 0.78 | gate: predicted IoU ≥ this | E17a |
| `--concept_only_logit` | 1.0 | gate: object score logit > this | **E17a uses 0** (changed from the planned 1 before the run, see `results/reports/e17__step0.md`) |
| `--lost_call_every K` | 0 (off) | E17b: while lost, call the concept every ≥ K frames | not run |
| `--fusion_w w` | 1.0 | decoder input (1 − w)·memory + w·concept instead of the concept branch alone (0.5 = original mix) | E17c (`E17c_w08`, **incomplete**: 19/160 videos) |
| `--fusion_sweep` | "" | offline: log the gated decoder output for every w in the list; output and trajectory stay unchanged | **E17c sweep** (0.5,…,1.0, subset40) |
| `--e17_commit` | `full` | where an accepted concept mask goes: `full` = output + pixel memory + bank (E17a); `pixel` = output + pixel memory; `output` = scored output only | **E18 pixel**, **E18 output** |

## Logging and running

| Flag | Default | Meaning |
|---|---|---|
| `--exp_log_dir` | none | write `frames/<video>.csv` (one row per object and frame: trigger, bank decision, logit, pred IoU, timings, E17 columns), `videos.csv`, `progress.txt` |
| `--save_concepts` | off | save each concept vector to `<exp_log_dir>/concepts/` |
| `--log_feat_norms` | off | log norms of memory / concept / mixed features on concept-call frames (E17 step 0, E17c) |
| `--seed` | none | seed torch / numpy / random (runs from the bank experiments on use `--seed 0`) |
| `--skip_done` | off | resume: skip videos whose masks are complete |

Other code changes (no flag): SDPA attention instead of flash-attn (`modeling_intern_vit.py`, LLM `attn_implementation`),
`lm_head` skipped when no labels (saves 2–3 GB VRAM), `offload_video_to_cpu=True`, a fix of `==` → `=` in the `longsam2.1`
config fallback, and a bank guard that never adds the same frame twice (only reachable with `--oracle_keyframe`).
