# Illustration videos

The videos come from the **SeCVOS** benchmark. They are used here only to illustrate this coursework.

## The five panels

Each video (and each PNG in `keyframes/`) is a 3 × 2 grid. The 6th cell holds the legend.

| | left | middle | right |
|---|---|---|---|
| **top** | **GT**: ground-truth mask | **E0**: original SeC, the **baseline** (rest120 J&F 70.0, all160 69.9) | **E10a**: SeC with denser concept calls (3 frames after each scene-change trigger, plus a call after ≥ 15 frames without one; rest120 72.2) |
| **bottom** | **E17a**: E10a + concept-only mask when lost. **Our best configuration** (rest120 77.4, all160 78.6) | **E18**: E17a, but the accepted mask is **not** put in the MLLM keyframe bank (`--e17_commit pixel`; rest120 77.1) | legend |

E0 masks and logs: `experiments/fps160/E0` (the same run as in `results/tables/per_object_E0.csv`).

What E17a adds:
- SAM 2 normally decodes the mask from a 50/50 mix of two features:
  - its **pixel memory** (masks of earlier frames);
  - the **concept** from the MLLM.
- When the tracker has been lost for 2 frames and the concept is called, E17a also decodes from the **concept alone**.
- If that mask passes a gate (not empty, predicted IoU ≥ 0.78, object-score logit > 0), it is **accepted**. It replaces the output and is written to:
  - E17a: the output, the SAM 2 pixel memory **and** the MLLM keyframe bank;
  - E18: the output and the pixel memory only. E18 shows that the pixel memory brings ~85 % of E17a's gain.

Markers:
- Orange = mask, the same colour in every panel. Label: configuration; frame number and J of this frame against GT.
- "object not visible" on the GT panel = the GT mask is empty on this frame.
- White dot (top right) = a concept (MLLM) call for this object on this frame.
- **Red border** = the run is lost (its own mask was empty or its logit ≤ 0 on the 2 previous frames).
  This is the model's own state. It is correct when the object really left the frame.
- **Green border** = the concept-only mask was accepted on this frame (E17a / E18 only).

Format: 6 fps, 1920 px wide, H.264. Rendered by `code/render/make_videos.py` from the masks and per-frame logs saved by the
original runs. **No inference was re-run.** J&F below = per-object score over the whole video (`results/tables/per_object_jf_160_wide.csv`).

## Selection rules (fixed before looking at the clips)

Videos 1–6. Prefer rest120 (the 120 videos not used for design):
1. the largest gain vs E10a, with at least one accepted concept mask while lost;
2. the bank self-lock case (snack can);
3. an invented object (accepted while GT is empty);
4. a wrong object with a high logit that the gate did not block;
5. a scene cut, choosing the typical outcome;
6. the object with the largest loss vs E10a.

Added later, to show E17a's gains (rule also fixed before rendering):

7–10. the next 4 objects with the largest J&F gain of **E17a over E10a** on rest120, after the rule-1 object, one object per video.

For rules 1 and 6, ranking by E17a or by E18 picks the same object.
Videos 1 and 7–10 are the **5 largest E17a gains on rest120**. Videos 2, 4 and 6 show where **E17a hurts**.
For scale: on rest120, E17a beats E10a on 61 objects and is worse on 28 (mean +5.1 J&F).
The top-5 gains are, by construction, the most striking cases, not typical ones.

## Videos

| # | File | Video / object (split) | Frames | J&F E0 → E10a → **E17a** / E18 |
|---|---|---|---|---|
| 1 | `1_recovery_09trCkd99Os.mp4` | 09trCkd99Os / 000 (rest120) | 0–127 | 17.6 → 17.6 → **90.9** / 90.8 |
| 2 | `2_bank_lock_aP2up9N6H-g.mp4` | aP2up9N6H-g / 000 (rest120) | 0–150 | 11.8 → 27.3 → **15.5** / 47.1 |
| 3 | `3_invented_E_UbFKdWFl4.mp4` | E_UbFKdWFl4 / 000 (rest120) | 71–160 | 56.2 → 57.5 → **81.2** / 81.0 |
| 4 | `4_wrong_object_CKmkgzeNKZw.mp4` | CKmkgzeNKZw / 000 (**subset40**) | 0–118 | 88.3 → 88.7 → **86.1** / 87.1 |
| 5 | `5_cut_both_fail_GMnti60Gxgc.mp4` | GMnti60Gxgc / 000 (rest120) | 0–108 | 65.3 → 67.8 → **67.8** / 67.8 |
| 6 | `6_harm_cIX8usnGj7s.mp4` | cIX8usnGj7s / 000 (rest120) | 0–162 | 91.6 → 91.6 → **49.2** / 49.3 |
| 7 | `7_gain_ccj3faULpKw.mp4` | ccj3faULpKw / 000 (rest120) | 0–141 | 29.6 → 30.4 → **86.9** / 84.1 |
| 8 | `8_gain_H7hkYP9CeSk.mp4` | H7hkYP9CeSk / 000 (rest120) | 0–178 | 44.0 → 44.0 → **99.6** / 66.9 |
| 9 | `9_gain_aYO2eRIFjQs.mp4` | aYO2eRIFjQs / 000 (rest120) | 0–167 | 52.1 → 46.8 → **96.0** / 82.7 |
| 10 | `10_gain_cw5P5iAJ3Xw.mp4` | cw5P5iAJ3Xw / 000 (rest120) | 0–140 | 39.1 → 52.9 → **92.9** / 92.6 |

The notes below compare E17a with E10a, because E10a is E17a's direct control (same flags, without the concept-only step).
E0 is shown as the baseline. In 9 of the 10 videos E0 scores at or below E10a. The exception is video 9 (E0 52.1, E10a 46.8).

Accept details below come from the logs `experiments/e17/E17a/frames/<video>.csv` and `experiments/e18/E18a_pixel/frames/<video>.csv`.

### 1. Recovery after a scene cut: 09trCkd99Os / 000 (E17a helps)
- This object has the largest gain vs E10a on rest120 (+73.3 for E17a).
- At the cut (frame 21), E10a loses the car and never finds it again.
- E17a accepts the concept-only mask at frame 27 (J 0.99, pred IoU 0.95, logit 0.49) and at frame 82 (J 0.98, logit 1.35).
  Both frames also go into its bank. From then on it tracks to the end. E18 does the same.
- Keyframes: `keyframes/1_recovery_09trCkd99Os_f018.png` (before the cut), `_f025` (all lost), `_f027` (accepted), `_f082` (second accept).

### 2. Bank self-lock: aP2up9N6H-g / 000, a snack can (E17a hurts, E18 helps)
- The can changes pose (frame 12: seen end-on in a hand).
- In E10a, 11 of the 13 scene-change triggers are refused by the bank because the mask is empty. Its bank stays at frames {0, 16}.
- **E17a gets worse (27.3 → 15.5).** Frames E17a adds to its bank:
  - frame 12, `added_concept`, J = **0.00**: the accepted concept-only mask is a *second* can at the right edge, a wrong object (logit 0.14);
  - frame 16, `added`, J = 0.42.
  - Its bank becomes {0, 12, 16} and keeps the wrong frame. E17a then misses the can at frames 72 and 145 (J 0.00).
- **E18 improves (47.1).** It accepts the same wrong mask at frame 12, but does not put it in the bank. Later accepts at 64 (J 0.98) and 139 (J 0.86).
  Its bank becomes {0, 16, 72} (frame 72, J 0.88). It finds the can at frames 72 (J 0.88) and 145 (J 0.93).
- This is the clearest case where writing accepted masks into the bank hurts.
- Keyframes: `_f005`, `_f012` (wrong accept), `_f030`, `_f072`, `_f145`.

### 3. Invented object: E_UbFKdWFl4 / 000, frame 131 (one harmful accept in an object that gains overall)
- At frame 131 the phone is not in GT. E10a correctly outputs nothing.
- E17a accepts a concept-only mask there (pred IoU 0.98, logit 0.50) and draws the phone: an invented object.
  The A1 analysis measures the harm as −10 frame-J over 10 frames. E18 does the same (logit 0.04).
- The object still gains overall (+23.7 for E17a), mostly from a correct accept at frame 148 (J 0.996).
- Keyframes: `_f128`, `_f131` (invented), `_f138`, `_f148` (correct accept).

### 4. Wrong object with a high logit: CKmkgzeNKZw / 000, frame 58 (the gate cannot block it)
- **This video is in subset40.** On rest120 no wrong-object accept causes clear harm.
  The closest is VGa1imApfdg frame 193 (logit 6.19, harm 1 frame).
- At frame 58 the concept-only mask covers the man's whole head and hood, instead of the GT part. J 0.27 vs 0.85 for E10a.
- E17a: logit **6.06**, pred IoU 0.93. A1 measures −4.8 frame-J over 16 frames. E18: logit 4.06.
- E17a also accepts wrong masks at frames 41 (J 0.15) and 118 (J 0.45), all with logit > 3. A high logit does not protect against this error.
- Keyframes: `_f050`, `_f058`, `_f066`, `_f074`.

### 5. Scene cut where every configuration fails: GMnti60Gxgc / 000, cut at frame 48 (typical case)
- On rest120, E10a is lost after a scene cut **60 times** (J before the cut > 0.7, mean J over the next 40 frames < 0.3).
  Here "scene cut" = PySceneDetect `min_scene_len=5`.
  - E18 finds the object again (mean J > 0.7) in **11** of them.
  - **Both fail (< 0.3) in 36**, so failing is the typical outcome.
- Here the concept is called 4 times while lost in frames 48–88 (frames 50, 51, 66, 81). The concept-only mask is empty every time (logit about −4).
  Nothing is accepted, and E10a, E17a and E18 give identical masks (J&F 67.8).
  When the concept itself cannot see the object, E17a has nothing to accept.
- Keyframes: `_f045` (before), `_f048` (cut), `_f060`, `_f090`.
- For the opposite outcome, a scene cut that E17a recovers, see video 7 (ccj3faULpKw, cut at frame 109).

### 6. Largest loss: cIX8usnGj7s / 000 (E17a hurts)
- This is the largest loss vs E10a on rest120 (−42.4 for E17a).
- The object is absent from GT for most of frames 10–120. E10a correctly outputs nothing there.
- E17a accepts at frame 34 while GT is empty (pred IoU 0.95, logit 0.34), and again at frame 52 (logit **2.69**).
  The invented mask enters the pixel memory (and the bank), and E17a keeps segmenting the wrong region until about frame 120.
- The frame-52 accept is the longest harm found by the A1 analysis on rest120: −55 frame-J over 42 frames.
- E18 fails the same way, starting at frame 34.
- Keyframes: `_f030`, `_f034` (first invented accept), `_f052` (second), `_f060`, `_f125`.

## E17a gains: videos 7–10

In all four, every E17a accept has GT not empty and J ≥ 0.95. There is no wrong accept.

### 7. Red car, two recoveries: ccj3faULpKw / 000 (+56.5)
- E10a loses the car around frame 45 and finds it again only briefly near frame 109.
- E17a accepts the concept-only mask at frame 56 (J 1.00, logit 1.49) and tracks the car from there (mean J ≈ 0.98).
- After the scene cut at frame 109, E17a accepts again (J 0.96, logit 1.69).
  Mean J over 40 frames after that cut: E10a 0.03, E17a 0.98, E18 0.92.
- Keyframes: `keyframes/7_gain_ccj3faULpKw_f045.png`, `_f056` (accept), `_f105`, `_f109` (cut, accept), `_f112`.

### 8. Race car where the bank matters: H7hkYP9CeSk / 000 (+55.6; E18 only +22.9)
- The car leaves and re-enters across several cuts. E10a is lost for frames 60–89 and from about frame 114 to the end.
- E17a accepts at frame 54 (J 0.95, logit 0.09) and **adds that frame to its MLLM bank**.
- At the cut at frame 114:
  - E17a's MLLM sees bank frames {0, 54, 88}, and the concept-only mask finds the car (J 0.99, logit 2.73).
  - E18 does not write accepts to the bank, so its bank is still only {0}. Its concept-only mask there is empty (logit −0.22), so it stays lost.
- This is the one case in this set where the bank write (E17a vs E18) clearly helps.
  On average the bank part is small: +0.31 on rest120, CI [−0.9, +1.5].
- Keyframes: `_f050`, `_f054` (accept), `_f070`, `_f114` (cut: E17a found, E18 lost), `_f150`.

### 9. Hand across many cuts: aYO2eRIFjQs / 000 (+49.2)
- A cooking video with 10 scene cuts. E10a loses the hand after most cuts.
- E17a accepts 5 times (frames 4, 105, 115, 132, 139; J 0.97–0.99) and stays on the hand: J ≈ 0.97–0.99 in almost every 15-frame block.
- E18 accepts 4 times (frames 4, 115, 133, 157) and reaches 82.7.
- Keyframes: `_f004` (first accept), `_f040` (E10a lost, E17a J 0.97), `_f105`, `_f139`.

### 10. Wooden object after a cut: cw5P5iAJ3Xw / 000 (+40.0)
- At the cut at frame 51, E10a loses the object. It stays lost until about frame 135, roughly 85 frames.
- E17a accepts once, at frame 55 (J 0.99, logit 0.28), and tracks it from then on (J ≈ 0.96–0.99).
  The logit 0.28 is below the originally planned gate (> 1), so this accept exists only because the gate was lowered to > 0.
- E18 does the same (92.6).
- Keyframes: `_f045`, `_f051` (cut), `_f055` (accept), `_f100`.
