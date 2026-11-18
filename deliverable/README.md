# SeC on SeCVOS: inference-time experiments (code and results)

This folder accompanies the slides. It holds the code we changed, the scripts we ran, the results, and a few illustration videos.

## What this is

**Task.** Semi-supervised video object segmentation: the object mask is given on the first frame, and the model segments it
in every later frame.

**Paper.** SeC (Segment Concept, [arXiv 2507.15852](https://arxiv.org/abs/2507.15852), [code](https://github.com/OpenIXCLab/SeC)).
SeC adds an MLLM to SAM 2:
- On a scene change, the MLLM looks at a few keyframes (the **bank**) and produces a **concept** vector.
- SAM 2 mixes this concept with its own **pixel memory** to find the object again.
- The paper reports 70.0 J&F on SeCVOS (160 multi-shot videos).

**What we did.** Inference only, no retraining, model SeC-4B:
1. Reproduced 69.9 J&F.
2. Tested whether the concept matters, and how often to call it (E1–E11).
3. Tested how to fill the bank (E6, E12–E14, with GT oracles).
4. Used the concept branch alone when the tracker is lost (E17), and found where that gain comes from (E18).
5. Tried to replace the LLM by a small network (I1, negative).

Main results on the 120 held-out videos (rest120):
- E10a (call the concept more often): 72.2 vs 70.0 for the original.
- E17a / E18 pixel (concept-only mask when lost): 77.4 / 77.1.
- E17a fails one pre-set criterion: wrong accepts are 22.3 %, above the 20 % limit.

## Folder map

| Path | Content |
|---|---|
| `code/sec_modifications.patch` | our changes to SeC (4 files), as a diff against upstream commit `0a797af` |
| `code/FLAGS.md` | every new flag: meaning, default, which experiment uses it |
| `code/run/` | one script per main experiment (`run_one.sh` = run + evaluate) |
| `code/analysis/` | analysis scripts as used (bootstrap CI, incidents, bank gate, A1/A2, w sweep, I1); `make_tables.py` builds `results/tables/` |
| `code/analysis/queues/` | the original queue scripts: the exact command lines used for each run |
| `code/tools/` | video lists (`subset40`, `rest120`, `all160`), swap maps, scene cuts, I1 split |
| `code/render/make_videos.py` | renders `videos/` and `keyframes/` from saved masks and logs |
| `results/SUMMARY.md` | **start here**: all main tables, each with its source file |
| `results/tables/` | per-object J&F per configuration, summary tables, number check |
| `results/reports/` | the original experiment reports, copied unchanged (**mostly in Vietnamese**; `SUMMARY.md` gives the numbers in English) |
| `results/figures/` | cost vs accuracy, J curves, w sweep |
| `results/data/` | small per-event tables (accepts, lost calls, sweep calls, I1 reports) |
| `videos/`, `keyframes/` | 10 illustration clips (6 cases incl. failures + 4 extra E17a gains), each a 3×2 grid: GT, E0 (baseline), E10a, **E17a (best)**, E18 pixel, legend; PNG frames for slides; see `videos/README.md` |

## Experiment map

Flag shorthand: `E10A` = `--burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv`;
`E17A` = `E10A --concept_only_lost --concept_only_logit 0 --seed 0`.
"Data" = folder in the original repo (`experiments/…`); per-object J&F is in `<data>/masks/results_overall.csv`.

| ID | Question | Change | Flags | Script | Main result (J&F) | Data / report |
|---|---|---|---|---|---|---|
| E0 | Reproduce SeC | none | – | `run/e0.sh` | 69.9 on 160 (paper 70.0) | `fps160/E0`, `outputs/secvos_full` |
| E1 | Does the concept matter? | concept off | `--concept_mode off` | `run/subset40_others.sh E1` | 61.6 vs 69.8 (subset40) | `E1` |
| E2 / E2b | Is it object-specific? | swap concepts across videos / objects | `--concept_mode swap --swap_map …` | `subset40_others.sh E2/E2b` | 60.1 / 60.4 | `E2`, `E2b` |
| E3a | Lower HSV threshold | 0.35 → 0.2 | `--hsv_thresh 0.2` | `subset40_others.sh E3a` | 71.5, CI incl. 0 | `E3a` |
| E7 | Periodic calls | + every 5 frames | `--concept_every 5 --bank_rule hsv` | `subset40_others.sh E7` | 72.2 * | `E7` |
| **E10a** | Calls after triggers | burst 3 + sparse 15 | `E10A` | `run/e10a.sh` | 73.9 (subset40); **72.2 rest120** * | `E10a`, `fps160/E10a`, `full160/E10a_rest120` |
| E11a | Uncertainty trigger | HSV or uncertainty | `--trigger_mode hsv_or_unc …` | `subset40_others.sh E11a` | 70.6, CI incl. 0 | `E11a`, `analysis/E11sim` |
| E4b40 | Upper bound on calls | concept every frame | `--concept_mode always --bank_rule hsv` | `subset40_others.sh E4b40` | 74.5 | `E4b40` |
| E12 | More bank frames | E10a + bank every ≥ 10 frames | `E10A --bank_extra_gap 10` | `run/e12.sh` | 75.1 subset40; **70.8 rest120, −1.4 vs E10a** * | `bank/E12`, `full160/E12_rest120` |
| E14 | More diverse bank | bank by descriptor cosine | `E10A --bank_diverse_tau 0.97` | `subset40_others.sh E14` | 75.0 (= E12) | `bank/E14` |
| E13a / **E13b** | Bank ceiling (oracle) | GT-mask frames in bank | `--bank_gt_gap 10` (+`E10A` for E13b) | `subset40_others.sh E13a`, `run/e13b.sh` | 74.0 / **81.1** (+7.2 vs E10a) * | `bank/E13a`, `bank/E13b` |
| I1 | Replace LLM by a student | distillation | – (offline) | `analysis/i1_distill/` | rel. MSE 0.97 / 0.74 > 0.5: **fail** | `i1_distill` |
| **E17a** | Concept branch alone when lost | keep concept-only mask if gate passes | `E17A` | `run/e17a.sh` | **77.4 rest120**, +5.1 vs E10a *; criterion (d) fails | `e17/E17a` |
| E17c | Best mix weight w | f_w = (1−w)·memory + w·concept | `--fusion_sweep 0.5,…,1.0` | `run/e17c_sweep.sh` | w\* = 1.0 (= E17a) | `e17c/sweep` |
| E18 | Where does E17a's gain come from? | accepted mask → output only / + pixel memory | `E17A --e17_commit output\|pixel` | `run/e18_output.sh`, `run/e18_pixel.sh` | output 72.7, **pixel 77.1** (rest120); 85 % of the gain = pixel memory | `e18/E18a_output`, `e18/E18a_pixel` |

`*` = 95 % CI of the difference excludes 0. Full tables: `results/SUMMARY.md`.

Not run or incomplete:
- **E17c_w08** (w = 0.8 on 160 videos): stopped at 19/160 videos. It is an unfinished run, and no number from it is reported.
- E17b, E18b and E18c: not run. Stop rules or decisions are given in `results/reports/e18__report_memory.md`.
- I1 steps 4–5: not run.

## Evaluation protocol

- **Metric.** J&F from the SeCVOS evaluator (`vos_evaluation/sav_evaluator.py --scene_info --strict`), averaged over objects.
- **Design set: subset40.** 40 videos / 44 objects: every 4th video after sorting the original run by J&F.
  All configurations were designed and selected on this set.
- **Main test: rest120.** The other 120 videos / 138 objects were not used to select anything.
  E10a, E12, E17a and E18 were run on them, and these are the main results.
- **Uncertainty.** Paired difference on the same objects, 95 % bootstrap CI (10,000 resamples, seed 0).
  We resample over objects and, separately, over videos (objects of a video kept together).
- **Pre-set criteria.** For E11, I1, I5, the bank experiments, E17a, E17c and E18, the pass/fail criteria and stop rules were written before the run.
  Results are reported against them and not reinterpreted afterwards.
- **Deviations from the pre-set plan:**
  - E17a acceptance logit: the plan said > 1, the run used **> 0**. This was decided before the 160-video run, after a 3-video check on subset40 videos (`results/reports/e17__step0.md`).
  - **E18 pixel was chosen after seeing the data.** It is reported because it is the closest run to the planned E18b, which was not run.
    Its check against the E18 criteria is a post-hoc check, not a pre-registered test.
  - E12 looked good on subset40 (+1.2) but is worse on rest120 (−1.4). We report both.
- **Equivalence checks.** With all new flags off, the masks are bit-identical to the original code. This was checked after each code change.
  The pipeline is deterministic (greedy decoding).

## Environment

- GPU: 1 × NVIDIA RTX 5070 Ti (16 GB), driver 580. Shared machine; some fps values were measured while another job used the GPU (marked in the tables).
- Python 3.10, torch 2.7.1+cu128, transformers 4.42.3, peft 0.11.1, numpy 1.26.4, opencv 4.11.0, scenedetect 0.7.3 (scene cuts only).
- No flash-attn: we use SDPA attention (included in the patch).
- Analysis scripts need pandas, numpy, Pillow, scikit-image (evaluator). Rendering needs ffmpeg.

### Install SeC and apply the patch

```bash
git clone https://github.com/OpenIXCLab/SeC.git && cd SeC
git checkout 0a797af                       # the upstream commit our patch is based on
git apply /path/to/deliverable/code/sec_modifications.patch
pip install -r requirements.txt            # then torch matching your CUDA
# model: https://huggingface.co/OpenIXCLab/SeC-4B  -> saved_models/SeC-4B
# data:  https://huggingface.co/datasets/OpenIXCLab/SeCVOS -> data/SeCVOS/{JPEGImages,Annotations,scene_info.json}
```

`git apply --check` was run on a clean copy of `0a797af`. The patch applies cleanly, and the 4 patched files are identical to the ones used for the experiments.

### Re-run one experiment (example: E18 pixel on one video)

```bash
cd SeC                                     # patched repo
echo 09trCkd99Os > one.txt
SEC_ROOT=$PWD DATA_ROOT=$PWD/data PYTHON=python LIST=one.txt bash /path/to/deliverable/code/run/e18_pixel.sh
# -> runs/E18_pixel/masks/09trCkd99Os/000/*.png, runs/E18_pixel/frames/09trCkd99Os.csv (per-frame log),
#    runs/E18_pixel/masks/results_overall.csv (per-object J&F; expected 90.8 for this object)
```

Without `LIST`, each script runs its default set (160 videos, or subset40 for the oracle and sweep runs). One 160-video run took about 1.3–1.4 h on our card (E17a 4656 s, E18 pixel 4982 s).
The analysis scripts in `code/analysis/` expect the original layout (`experiments/<run>/…` under the SeC root) and the raw masks and logs, which are not shipped (size).
`make_tables.py` needs only the `results_overall.csv` files.

## Reproducibility notes

- Videos: rendered from masks and logs saved by the original runs. **No inference was re-run** to make this folder, and the GPU was not used.
- Seeds: runs from the bank experiments onward use `--seed 0`. Earlier subset40 runs did not set a seed, but the pipeline is deterministic.
  E0 and E10a re-runs with seed 0 gave bit-identical masks.
- Number check: every headline number was recomputed from the per-object files (`results/tables/number_check.csv`). The largest difference from the slide values is 0.05.

## Limitations

- One benchmark (SeCVOS), 160 videos. rest120 has 138 objects, so CIs are about ±1–2.5 points. Effects below about 1 point cannot be resolved.
- All thresholds were chosen on subset40: E10a schedule, pred IoU 0.78, the logit gate and the "lost" definition (2 frames). E17a's logit change was decided after looking at 3 subset40 videos.
- E17a fails its pre-set wrong-accept criterion (22.3 % vs < 20 %). Invented objects (accepted while GT is empty) cause most of the harm. No tested signal detects them reliably on rest120 (AUC ≤ 0.70).
- fps was measured in different sessions on a shared GPU. Compare speeds only roughly.
- The oracle runs (E13a, E13b) use GT. They show a ceiling, not a usable method.
- The copied reports are in Vietnamese and contain one mislabelled example. See "Corrections" in `results/SUMMARY.md`.
- Video frames belong to the SeCVOS dataset (annotations CC BY-NC-SA 4.0). They are shown here only to illustrate this coursework.
