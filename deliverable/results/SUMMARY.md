# Results summary

Every J&F, Δ and CI below was **recomputed** from the per-object scores (`<run>/masks/results_overall.csv`) by
`code/analysis/make_tables.py`. They were not copied from the reports. Output files are in `tables/`.

Numbers about mechanisms (accepts, A1/A2, w sweep, I1) come from the logs, through the scripts named in each section.
For those, the source report is given and the number is quoted from it.

How to read the tables:
- J&F = mean over objects of the per-object J&F, from the SeCVOS evaluator (`sav_evaluator.py --scene_info --strict`).
- Δ = paired difference on the same objects.
- CI = 95 % bootstrap, 10,000 resamples, seed 0. Two versions: resampling **objects**, and resampling **videos** (objects of a video kept together).
- `*` = CI excludes 0. "up/down" = number of objects that improve / get worse.
- Splits:
  - **subset40** = 40 videos / 44 objects used to design and select configurations;
  - **rest120** = the other 120 videos / 138 objects, the main held-out test;
  - **all160** = all 160 videos / 182 objects.
  - The split is in `code/tools/lists/`. `subset40` = every 4th video after sorting by the original run's J&F.

## 1. Reproduction

| Run | Videos | J&F | J | F | Source |
|---|---|---|---|---|---|
| SeC-4B, paper | 160 | 70.0 | – | – | SeC README, SeCVOS column |
| SeC-4B, original code, our GPU | 160 | **69.9** | 69.7 | 70.2 | `outputs/secvos_full` (eval log `eval_secvos_full.txt`) |
| E0 = original settings, seed 0, our code with all flags off | 160 | 69.9 | 69.7 | 70.2 | `experiments/fps160/E0` → `tables/per_object_E0.csv` |

The two E0 runs differ on 1 object of 182, by 0.1 J&F. E0 is the reference everywhere below.

## 2. subset40: design experiments

Source: `tables/summary_subset40.csv` and `tables/per_object_subset40_all_runs.csv`.
"% calls" and fps come from `reports/results__summary.md`; they were not recomputed.

| ID | Change | J&F | Δ vs E0 [obj CI; video CI] | up/down | Direct control: Δ [obj CI] | % frames with a concept call | fps |
|---|---|---|---|---|---|---|---|
| E0 | original (HSV trigger 0.35) | 69.8 | – | – | – | 7.2 | 9.9 |
| E1 | concept off | 61.6 | −8.1 [−11.9, −4.6]*; [−11.8, −4.6]* | 6/33 | – | 0 | 11.3 |
| E2 | concept of an object from another video | 60.1 | −9.7 [−15.1, −4.6]*; [−14.8, −4.5]* | 7/33 | – | 7.2 | 11.2 |
| E2b | concept of another object, same video (7 objects) | 60.4 | −26.7 [−48.4, −10.0]*; [−30.7, −20.3]* | 0/6 | – | 9.6 | 6.1 |
| E3a | HSV threshold 0.2 | 71.5 | +1.7 [−1.8, +5.4]; [−1.8, +5.7] | 12/14 | – | 17.1 | 7.9 |
| E7 | + concept call every 5 frames, bank on HSV triggers | 72.2 | +2.4 [+0.4, +4.7]*; [+0.4, +4.8]* | 16/17 | – | 25.7 | 7.6 |
| **E10a** | + 3-frame burst after each trigger, + call after ≥ 15 frames without one; bank on HSV triggers | **73.9** | **+4.1 [+1.6, +6.8]*; [+1.5, +7.1]*** | 23/14 | vs E7: +1.7 [−0.0, +3.8] | 24.6 | 4.4 (GPU shared); 8.3 median in `fps160` |
| E11a | HSV OR uncertainty trigger | 70.6 | +0.8 [−1.0, +2.5]; [−1.0, +2.6] | 18/8 | vs E3a: −0.8 [−4.9, +3.0] | 15.3 | 8.8 |
| E4b40 | concept on every frame, bank on HSV triggers | 74.5 | +4.8 [−0.0, +9.8]; [+0.1, +10.0]* | 21/19 | – | 100 | 4.0 |
| E6 | E0 + bank also every ≥ 10 frames | 71.1 | +1.3 [−0.8, +3.5]; [−0.8, +3.5] | 10/8 | – | 7.2 | 9.6 |
| E13a | **oracle**: E0 + GT-mask frame in bank every ≥ 10 frames | 74.0 | +4.2 [+1.3, +7.6]*; [+1.4, +7.8]* | 17/6 | vs E6: **+2.9 [+0.3, +6.0]*** | 7.2 | 9.4 |
| E12 | E10a + bank also every ≥ 10 frames (as E6) | 75.1 | +5.3 [+1.8, +9.3]*; [+1.7, +9.5]* | 23/11 | vs E10a: +1.2 [−2.0, +4.7] | 24.6 | 6.8 |
| E14 | E10a + bank by diversity (τ = 0.97) | 75.0 | +5.2 [+1.7, +9.3]*; [+1.6, +9.5]* | 21/16 | vs E12: −0.1 [−2.0, +1.6] | 24.6 | 7.0 |
| E13b | **oracle**: E10a + GT-mask frame in bank every ≥ 10 frames | **81.1** | +11.3 [+6.6, +16.7]*; [+6.4, +17.0]* | 31/8 | vs E10a: **+7.2 [+2.5, +12.5]*** | 24.6 | 6.6 |

Notes:
- E2b uses the 3 multi-object videos (7 objects), compared with E0 on the same objects.
- fps is not comparable across all rows. E10a (subset40) and E7bd ran while another job shared the GPU.

What the table shows:
- **The concept carries object-specific information.** Removing it (E1) or swapping it (E2, E2b) costs 8–27 points.
- **Calling it more often helps.** E10a is the cheapest schedule with a clear gain.
- **Bank quality is the bottleneck.** Adding more predicted frames (E6, E12) or more diverse frames (E14) does not help.
  Adding GT-mask frames (E13b) gives +7.2.

## 3. 160 videos: held-out test

Source: `tables/summary_160.csv`, `tables/per_object_*.csv`, `tables/per_object_jf_160_wide.csv`.
Each E12 row combines `experiments/bank/E12` (subset40) with `experiments/full160/E12_rest120`.

| Split | Run | J&F | Δ vs E0 [obj CI; video CI] | Δ vs E10a [obj CI; video CI] | up/down vs E10a |
|---|---|---|---|---|---|
| **rest120** (138 obj) | E0 | 70.0 | – | −2.3 | – |
| | E10a | 72.2 | +2.3 [+1.0, +3.7]*; [+1.0, +3.7]* | – | – |
| | E12 | 70.8 | +0.9 [−0.7, +2.5]; [−0.7, +2.5] | **−1.4 [−2.7, −0.2]*; [−2.8, −0.2]*** | 36/57 |
| | E17a | **77.4** | +7.4 [+4.7, +10.3]*; [+4.7, +10.4]* | **+5.1 [+2.8, +7.6]*; [+2.8, +7.6]*** | 61/28 |
| | E18 output | 72.7 | +2.7 [+1.4, +4.2]*; [+1.4, +4.3]* | +0.5 [+0.2, +0.7]*; [+0.2, +0.7]* | 48/22 |
| | E18 pixel | 77.1 | +7.1 [+4.6, +9.7]*; [+4.6, +9.7]* | +4.8 [+2.8, +7.0]*; [+2.8, +6.9]* | 60/25 |
| **all160** (182 obj) | E0 | 69.9 | – | −2.7 | – |
| | E10a | 72.6 | +2.7 [+1.6, +4.0]*; [+1.5, +4.0]* | – | – |
| | E12 | 71.9 | +1.9 [+0.4, +3.5]*; [+0.4, +3.5]* | −0.8 [−2.1, +0.5]; [−2.1, +0.5] | 55/68 |
| | E17a | **78.6** | +8.7 [+6.1, +11.3]*; [+6.0, +11.4]* | +5.9 [+3.8, +8.3]*; [+3.7, +8.3]* | 83/36 |
| | E18 output | 73.2 | +3.3 [+2.1, +4.6]*; [+2.0, +4.7]* | +0.6 [+0.4, +0.9]*; [+0.3, +0.9]* | 62/28 |
| | E18 pixel | 77.8 | +7.9 [+5.6, +10.3]*; [+5.5, +10.3]* | +5.1 [+3.3, +7.1]*; [+3.3, +7.1]* | 79/33 |

**10 worst E0 videos** (all160, ranked by E0 per-video J&F; mean J&F over their objects; `tables/worst10_E0_videos.csv`):
E0 19.1, E10a 28.7, E17a 48.8, E18 pixel 48.2 (recomputed from `tables/per_object_jf_160_wide.csv`).

Cost (quoted from `reports/e17__report.md`, rest120; fps measured in different sessions, compare only roughly):
- % frames with a concept call: E0 7.5, E10a 26.0, E17a 26.0.
- Median fps: E0 10.5, E10a 8.0, E17a 6.7, pixel 6.9 (`reports/e18__report_memory.md`).

What the table shows:
- E10a's gain holds on the held-out 120 videos (+2.3). It is smaller than on subset40 (+4.1), as expected for a configuration selected on subset40.
- E12 does **not** hold: it is worse than E10a on rest120.
- E17a and E18 pixel add about +5 over E10a on rest120. The CI excludes 0 for both resampling schemes.

## 4. E17a: concept branch alone when lost

Source: `reports/e17__report.md` (numbers from `experiments/e17/report.py` → `data/e17_report_numbers.txt`, `data/e17_lost_calls.csv`),
and `reports/e18__analysis_report.md` (A1/A2, `experiments/e18/analysis.py` → `data/e18_accepts.csv`).

**Pre-set criteria (rest120):**

| Criterion | Result | |
|---|---|---|
| (a) J of the concept-only mask > J of the mixed mask, on concept calls while lost (GT not empty) | 0.300 > 0.207 | pass |
| (b) incident recovery rate higher than E10a | 64.3 % > 57.0 % | pass |
| (c) J&F not more than 0.5 below E10a | +5.1 [+2.8, +7.6] | pass |
| (d) wrong accepts < 20 % of accepts | **22.3 %** (59/265: 45 invented + 14 wrong object) | **fail** |

Following the pre-set rule, E17a is **not** declared a pass, despite the large J&F gain.
Deviation from the design: the acceptance logit was changed from > 1 to > 0 before the run (`reports/e17__step0.md`).
The new (0, 1] logit group has 36.6 % wrong accepts; the > 1 group has 17.0 %.

Check of the accept counts. `data/e18_accepts.csv` was tabulated again:

| Split | correct | invented (GT empty) | wrong object (J < 0.5) |
|---|---|---|---|
| rest120 | 206 | **45** | **14** |
| subset40 | 54 | 10 | 10 |
| all160 | 260 | 55 | 24 |

- This matches the 45 / 14 in `reports/e17__report.md`.
- In every row of `accepts.csv`, the type agrees with `gt_empty` and `J_con`.

**A1. How long does a wrong accept hurt?** (rest120; ΔJ = J(E17a) − J(E10a) summed over frames in the window after the accept)

| Type | n | mean ΔJ per accept | total ΔJ | harm length median; mean (frames) |
|---|---|---|---|---|
| correct | 206 | +7.36 | +1515 | 0; 1.6 |
| invented | 45 | −6.43 | −290 | 3; 7.9 |
| wrong object | 14 | −0.64 | −9 | 0; 0.1 |

- Total gain is about 5× the total harm.
- Invented objects cause almost all the harm. 5 of 45 last more than 20 frames.

**A2. Is there a better signal than the logit?** AUC, correct vs wrong accept:

| Signal | subset40 | rest120 |
|---|---|---|
| pred IoU | 0.77 | 0.70 |
| logit | 0.77 | 0.65 |
| max cosine to bank | 0.61 | 0.57 |
| mask area | 0.54 | 0.50 |

No signal separates well on rest120. The AUC measured on subset40 does not carry over.

## 5. E17c: sweep of w

Source: `reports/e17c__sweep_report.md`, `data/e17c_sweep_by_w.csv`, `data/e17c_sweep_calls.csv`, `figures/e17c_sweep_plot.png`.

Setup:
- Decoder input f_w = (1 − w)·memory + w·concept. w = 0.5 is the original mix; w = 1.0 is E17a.
- Offline: the trajectory is E10a's, so the numbers are per frame and do not include later effects.
- subset40, 875 concept calls while lost.

| w | effective J (all calls) | Δ vs 0.5 [CI] | accepts | invented | wrong object | % wrong |
|---|---|---|---|---|---|---|
| 0.5 | 0.652 | – | – | – | – | – |
| 0.6 | 0.658 | +0.006 [+0.002, +0.012] | 62 | 5 | 4 | 14.5 |
| 0.7 | 0.677 | +0.025 [+0.012, +0.040] | 81 | 5 | 5 | 12.3 |
| 0.8 | 0.706 | +0.053 [+0.026, +0.086] | 106 | 5 | 4 | 8.5 |
| 0.9 | 0.733 | +0.081 [+0.034, +0.133] | 144 | 11 | 6 | 11.8 |
| 1.0 | **0.743** | +0.091 [+0.041, +0.146] | 159 | 13 | 7 | 12.6 |

- The pre-set rule picks **w\* = 1.0**, which is E17a itself. No intermediate w beats the concept branch alone.
- The 160-video run `E17c_w08` (w = 0.8) was **stopped at 19/160 videos**. It is incomplete and not reported.

## 6. E18: where does E17a's gain come from?

Source: `tables/e18_decomposition.csv` (recomputed).
Setup:
- G = E17a − E10a, split in a fixed order:
  - output (`--e17_commit output` − E10a);
  - pixel memory (`pixel` − `output`);
  - bank (E17a − `pixel`).
- Bootstrap over objects and over videos.

| Split | Part | Δ | CI obj | CI video | share of G |
|---|---|---|---|---|---|
| **rest120** | output | +0.47 | [+0.2, +0.7]* | [+0.2, +0.7]* | 9 % |
| | **pixel memory** | **+4.35** | [+2.4, +6.4]* | [+2.5, +6.3]* | **85 %** |
| | bank | +0.31 | [−0.9, +1.5] | [−0.9, +1.6] | 6 % |
| | total G | +5.13 | [+2.8, +7.6]* | [+2.8, +7.6]* | |
| all160 | output / pixel / bank | +0.59 / +4.55 / +0.80 | bank [−0.3, +2.0] | | 10 / 77 / 13 % |
| subset40 | output / pixel / bank | +0.97 / +5.19 / +2.32 | bank [−0.1, +5.0] | | 11 / 61 / 27 % |

- **Most of the gain comes from writing the accepted mask into the SAM 2 pixel memory.** The bank part is small and its CI includes 0.
- `pixel` keeps almost all the gain: −0.3 vs E17a, CI [−1.5, +0.9].
  It puts far fewer wrong frames into the bank: 59 → 17 (`reports/e18__report_memory.md`).
- `reports/e18__report_e18a.md` writes G = +5.2 and pixel share 84 %. The recomputed values are +5.13 and 85 % (a rounding difference).
- E18b (bank after k confirming frames) was not run. The pre-set stop rule applied: bank part < 0.5 with a CI including 0.
- E18c was dropped: SeC already filters its pixel memory, and only frame f−1 can be bad.

## 7. I1: replace the LLM by a small student network (negative result)

Source: `reports/CHANGES.md` (I1 section), `data/i1_distill_*.json`.
Setup:
- Train / val split by video: 108 / 12 videos of rest120. Test = subset40, not used.
- 4140 teacher concept vectors.
- Pre-set pass criterion on val: cosine ≥ 0.8 **and** relative MSE (student MSE / MSE of the constant mean vector) ≤ 0.5.

| Model | val cosine | val relative MSE | Pass |
|---|---|---|---|
| constant mean vector | 0.867 | 1.00 | – |
| student (1.8 M params) | 0.869 | **0.97** | no |
| object-centric student (0.37 M) | 0.902 | **0.74** | no |
| oracle: knows the object (leave-one-out object mean) | – | 0.55 | – |
| oracle: object + visible in current frame | – | 0.42 | – |

- Cosine is a weak measure here: the constant mean vector already reaches 0.87.
- Even knowing which object it is gives only 0.55.
- Steps 4–5 (student inside inference) were not run.

## Corrections to the copied reports

The reports in `reports/` are copied unchanged. Known issues:

1. **`reports/e17__report.md`, line 49:** "`mGkTa8gcANY` object 000, frame 107: GT không rỗng (GT not empty), J concept 0, pred_iou 0.79, logit 4.6 → wrong object."
   - This is wrong. The GT masks of frames 105–109 are empty (0 pixels), and `data/e18_accepts.csv` labels this accept `invented`.
   - It is an **invented object with a high logit (4.6)**, not a wrong object. Harm: 1 frame-J, harm length 0.
   - All other labelled examples in the reports were checked against the GT and `accepts.csv` / `sweep_calls.csv`, and they are correct:
     - dIuH0AhcX5Q/001 f88; E_UbFKdWFl4/000 f131; cIX8usnGj7s f52; IppaafjwUUA f105 (invented);
     - lKsyz336zNw f49–51 and -klPo9pilc4/002 f95–96 (invented, w = 1.0 sweep);
     - CKmkgzeNKZw f41/58/118 (wrong object);
     - JJNUDzMqeFU f136 (GT empty).
2. `reports/e18__report_e18a.md`: G = +5.2 → recomputed +5.13 (see section 6).
3. `reports/e18__analysis_report.md` writes "+1515 vs −299" in the text and −290 in the table.
   −290 is the invented accepts only; −299 adds the 14 wrong-object accepts (−9).
4. `reports/CHANGES.md` stops at E17a. The E17c and E18 code changes (`--fusion_w`, `--fusion_sweep`, `--e17_commit`, `--bank_confirm_k`)
   are described in `reports/e17c__sweep_report.md`, `reports/e18__analysis_report.md` and `code/FLAGS.md`.

## Number check

`tables/number_check.csv` compares recomputed values with the values quoted in the slides. No difference is larger than 0.1 (largest: 0.05).
