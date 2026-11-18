# 160-video run: E0, E10a, E12


## all160 (160 videos, 182 objects)

| Run | J&F | J | F | J&F first / second / remain | Δ vs E0 [object CI] | Δ vs E0 video CI | up / down vs E0 | Δ vs E10a [object CI; video CI] | up / down vs E10a |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 69.9 | 69.7 | 70.2 | 84.0 / 69.8 / 67.5 | – | – | – | – | – |
| E10a | 72.6 | 72.4 | 72.9 | 85.7 / 72.9 / 70.2 | +2.7 [+1.6, +4.0] * | [+1.5, +4.0] * | 95 / 54 | – | – |
| E12 | 71.9 | 71.6 | 72.1 | 85.4 / 73.3 / 68.6 | +1.9 [+0.4, +3.5] * | [+0.4, +3.5] * | 83 / 65 | -0.8 [-2.1, +0.5]; [-2.1, +0.5] | 55 / 68 |

## rest120 (120 videos, 138 objects)

| Run | J&F | J | F | J&F first / second / remain | Δ vs E0 [object CI] | Δ vs E0 video CI | up / down vs E0 | Δ vs E10a [object CI; video CI] | up / down vs E10a |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 70.0 | 69.8 | 70.2 | 85.1 / 68.7 / 68.4 | – | – | – | – | – |
| E10a | 72.2 | 72.1 | 72.4 | 86.8 / 71.4 / 70.4 | +2.3 [+1.0, +3.7] * | [+1.0, +3.7] * | 72 / 40 | – | – |
| E12 | 70.8 | 70.6 | 71.0 | 86.3 / 70.7 / 68.3 | +0.9 [-0.7, +2.5] | [-0.7, +2.5] | 60 / 54 | -1.4 [-2.7, -0.2] *; [-2.8, -0.2] * | 36 / 57 |

## subset40 (40 videos, 44 objects)

| Run | J&F | J | F | J&F first / second / remain | Δ vs E0 [object CI] | Δ vs E0 video CI | up / down vs E0 | Δ vs E10a [object CI; video CI] | up / down vs E10a |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 69.8 | 69.4 | 70.2 | 80.8 / 73.1 / 64.5 | – | – | – | – | – |
| E10a | 73.9 | 73.5 | 74.3 | 82.3 / 77.8 / 69.3 | +4.1 [+1.6, +6.8] * | [+1.5, +7.1] * | 23 / 14 | – | – |
| E12 | 75.1 | 74.7 | 75.5 | 82.8 / 81.6 / 69.7 | +5.3 [+1.8, +9.3] * | [+1.7, +9.5] * | 23 / 11 | +1.2 [-2.0, +4.7]; [-2.0, +4.7] | 19 / 11 |

## Cost on rest120 (from the new runs' logs)

| Run | % frame gọi concept | fps | median non-call frame ms | bank added |
|---|---|---|---|---|
| E10a | 26.02 | 6.81 | 76 | 674 |
| E12 | 26.02 | 6.14 | 76 | 1686 |

E0 = outputs/secvos_full (earlier 160-video run, original code; no per-frame log). Δ: paired bootstrap over objects / video clusters, 10,000 resamples, seed 0; * = 95% CI excludes 0. subset40 rows use the existing subset40 masks (E10a, bank/E12); E0 there = earlier run (same J&F as experiments/E0).
