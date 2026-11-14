## J&F / J / F

| Exp | J&F | ΔJ&F vs E0 [95% CI] | objects up / down | J | F | J&F first / second / remain | Δ video-cluster CI | Δ vs same-cost ref (E10a: E7, E10b: E8, E11a: E3a) [object CI; video CI; up / down] | % frame gọi concept | fps | predict_forward mean / p95 ms | bank added (periodic) | GPU h |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 – baseline (HSV 0.35) | 69.8 | – | – | 69.4 | 70.2 | 80.8 / 73.1 / 64.5 | – | – | 7.22 | 9.91 | 149 / 268 | – | 0.20 |
| E1 – concept off | 61.6 | -8.1 [-11.9, -4.6] * | 6 / 33 (n=44) | 61.2 | 62.1 | 77.6 / 64.0 / 52.6 | [-11.8, -4.6] * | – | 0.00 | 11.31 | nan / nan | 0 (0) | 0.18 |
| E2 – concept swapped (other video) | 60.1 | -9.7 [-15.1, -4.6] * | 7 / 33 (n=44) | 59.5 | 60.6 | 77.8 / 61.5 / 47.7 | [-14.8, -4.5] * | – | 7.22 | 11.21 | nan / nan | 97 (0) | 0.18 |
| E3a – HSV threshold 0.2 | 71.5 | +1.7 [-1.8, +5.4] | 12 / 14 (n=44) | 71.1 | 71.8 | 82.0 / 78.5 / 63.8 | [-1.8, +5.7] | – | 17.14 | 7.87 | 201 / 268 | 499 (0) | 0.26 |
| E3b – HSV threshold 0.5 | 68.3 | -1.5 [-3.7, +0.3] | 7 / 15 (n=44) | 67.9 | 68.7 | 79.4 / 71.7 / 62.7 | [-3.5, +0.4] | – | 4.34 | 10.54 | 122 / 266 | 104 (0) | 0.19 |
| E6 – bank also every >=10 frames | 71.1 | +1.3 [-0.8, +3.5] | 10 / 8 (n=44) | 70.7 | 71.5 | 81.5 / 73.8 / 65.1 | [-0.8, +3.5] | – | 7.22 | 9.55 | 194 / 268 | 480 (293) | 0.21 |
| E7 – concept every 5 frames + HSV 0.35, bank on HSV triggers | 72.2 | +2.4 [+0.4, +4.7] * | 16 / 17 (n=44) | 71.8 | 72.5 | 80.8 / 76.8 / 68.1 | [+0.4, +4.8] * | – | 25.65 | 7.60 | 149 / 268 | 200 (0) | 0.26 |
| E7bd – concept every 5 frames + HSV 0.35, original bank rule | 70.6 | +0.8 [-2.4, +4.2] | 17 / 18 (n=44) | 70.2 | 71.0 | 79.8 / 73.9 / 65.5 | [-2.4, +4.2] | – | 25.65 | 3.71 | 359 / 485 | 868 (0) | 0.54 |
| E8 – E6 + HSV 0.2 | 72.8 | +3.0 [-0.4, +6.8] | 17 / 14 (n=44) | 72.4 | 73.2 | 83.2 / 78.5 / 64.3 | [-0.5, +7.3] | – | 17.14 | 7.61 | 222 / 268 | 734 (241) | 0.26 |
| E10a – concept on 3 frames after each HSV trigger + every >=15 frames, bank on HSV triggers | 73.9 | +4.1 [+1.6, +6.8] * | 23 / 14 (n=44) | 73.5 | 74.3 | 82.3 / 77.8 / 69.3 | [+1.5, +7.1] * | vs E7: +1.7 [-0.0, +3.8]; video [-0.0, +3.8]; 22 / 12 | 24.59 | 4.41 | 242 / 399 | 204 (0) | 0.46 |
| E10b – concept on 3 frames after each HSV trigger, bank on HSV triggers | 72.4 | +2.6 [+0.5, +5.0] * | 22 / 9 (n=44) | 71.9 | 72.8 | 81.7 / 76.6 / 66.9 | [+0.4, +5.1] * | vs E8: -0.5 [-4.6, +3.4]; video [-4.7, +3.4]; 22 / 14 | 21.32 | 7.62 | 161 / 268 | 199 (0) | 0.26 |
| E11a – HSV OR uncertainty (D=2, r=0.5, N=1, C=5), bank on HSV triggers | 70.6 | +0.8 [-1.0, +2.5] | 18 / 8 (n=44) | 70.3 | 70.9 | 80.2 / 74.7 / 67.0 | [-1.0, +2.6] | vs E3a: -0.8 [-4.9, +3.0]; video [-5.1, +2.9]; 21 / 14 | 15.31 | 8.75 | 144 / 268 | 197 (0) | 0.23 |
| E4b40 – concept every frame, bank on HSV triggers | 74.5 | +4.8 [-0.0, +9.8] | 21 / 19 (n=44) | 74.3 | 74.8 | 81.8 / 76.0 / 71.6 | [+0.1, +10.0] * | – | 100.00 | 3.97 | 147 / 268 | 206 (0) | 0.50 |
| E0r – baseline re-run (extended log) | 69.8 | +0.0 [+0.0, +0.0] | 0 / 0 (n=44) | 69.4 | 70.2 | 80.8 / 73.1 / 64.5 | [+0.0, +0.0] | – | 7.22 | 9.90 | 149 / 268 | 199 (0) | 0.20 |
| E0@short10 – E0 on the 10 shortest | 70.2 | – | – | 70.1 | 70.3 | 81.9 / 78.0 / 62.0 | – | – | | | | | |
| E4 – concept on every frame | 74.0 | +3.8 [-11.5, +17.2] | 7 / 5 (n=12) | 73.9 | 74.1 | 82.8 / 84.2 / 66.4 | – | – | 100.00 | 2.47 | 258 / 272 | 699 (0) | 0.15 |
| E4b – concept every frame, bank on HSV triggers | 80.6 | +10.4 [+1.4, +21.1] * | 7 / 4 (n=12) | 80.7 | 80.6 | 83.9 / 84.9 / 79.4 | – | – | 100.00 | 3.61 | 150 / 268 | 65 (0) | 0.10 |
| E0@multiobj – E0 on the 3 multi-object videos | 87.1 | – | – | 87.4 | 86.8 | 93.8 / 81.2 / 86.8 | – | – | | | | | |
| E2b – concept swapped (same video) | 60.4 | -26.7 [-48.4, -10.0] * | 0 / 6 (n=7) | 60.4 | 60.3 | 81.5 / 62.6 / 54.5 | – | – | 9.57 | 6.05 | nan / nan | 9 (0) | 0.02 |

Δ: paired bootstrap over objects vs E0 on the same objects (10,000 resamples, seed 0); * = 95% CI excludes 0. Objects with equal J&F count as neither up nor down. Video-cluster CI: same, resampling the 40 videos (a video's objects stay together). Bank added: frames appended to the MLLM memory over all objects (non-cond frames; E0 has no bank log; see E0r).

E7bd vs E7 (original bank rule vs bank on HSV triggers): -1.6 object CI [-5.1, +1.7], video-cluster CI [-5.2, +1.7]; objects up / down 16 / 20 (n=44)

E10a ran while another user's job shared the GPU (median non-call frame 139 ms vs 76 ms in E0r/E7/E10b): its fps, predict_forward and GPU h are not comparable; masks/J&F are unaffected.

E6 on the 10 worst E0 videos (object mean J&F): E0 36.3 -> E6 40.3
E6 frames added to the bank: 480 (293 by the every-10-frames rule); E0r: 199

E5 (E0): 40 videos, time/video by #objects: 1 obj: 17.0s (n=37), 2 obj: 30.3s (n=2), 3 obj: 37.1s (n=1)

## Incidents (J drop > 0.3 vs mean of previous 5 frames; cuts = msl5; 'concept' = a concept call for this object within ±2 frames)

| Exp | incidents (objects) | at cut + concept | at cut, no concept | no cut + concept | no cut, no concept |
|---|---|---|---|---|
| E0 | 102 (39) | 53 (rec. 15/53) | 6 (rec. 3/6) | 20 (rec. 10/20) | 23 (rec. 14/23) |
| E7 | 122 (41) | 67 (rec. 29/67) | 0 (rec. 0/0) | 55 (rec. 36/55) | 0 (rec. 0/0) |
| E8 | 110 (39) | 57 (rec. 22/57) | 2 (rec. 2/2) | 42 (rec. 26/42) | 9 (rec. 6/9) |
| E10a | 120 (39) | 56 (rec. 20/56) | 6 (rec. 4/6) | 44 (rec. 31/44) | 14 (rec. 10/14) |
| E10b | 108 (38) | 54 (rec. 19/54) | 6 (rec. 4/6) | 30 (rec. 18/30) | 18 (rec. 12/18) |
| E11a | 111 (40) | 58 (rec. 18/58) | 4 (rec. 3/4) | 43 (rec. 27/43) | 6 (rec. 3/6) |

rec. = recovered within 10 frames (max J over f+1..f+10 >= previous mean - 0.1).
