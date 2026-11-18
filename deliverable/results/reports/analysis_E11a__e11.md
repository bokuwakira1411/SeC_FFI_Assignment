## E11a: recall on E0's incidents (v1), real vs simulated
| | % frame gọi concept | recall (102) | recall HSV-missed |
|---|---|---|---|
| simulated (E0 trajectory) | 14.34 | 98/102 = 96.1% | 19/23 |
| real E11a | 15.31 | 98/102 = 96.1% | 19/23 |

E11a concept calls by reason: {'unc_logit': 103, 'hsv': 558, 'unc_empty': 502, 'unc_area': 20}

## Recovery within 10 frames on E0's incidents (max J over onset+1..+10 >= E0 pre_mean - 0.1)
| group | n | E0 recovered | E11a recovered | E11a J at onset < E0 pre_mean - 0.3 (incident still there) |
|---|---|---|---|---|
| all E0 incidents | 102 | 42/102 | 48/102 | 97 |
| caught by an unc call in E11a | 50 | 23/50 | 25/50 | 48 |
|   of which HSV-missed | 19 | 9/19 | 11/19 | 19 |

E11a's own incidents (v1): 111; with an unc call in [onset-2, onset+5]: 56, recovered 28/56

## Cosine between the previous and the new concept (same object), by reason of the new call
| reason | n | mean | median | p10 | p90 |
|---|---|---|---|---|---|
| hsv (all) | 533 | 0.930 | 0.959 | 0.824 | 0.993 |
| unc (all) | 606 | 0.974 | 0.992 | 0.930 | 0.999 |
| unc_empty | 501 | 0.979 | 0.993 | 0.952 | 0.999 |
| unc_logit | 85 | 0.957 | 0.980 | 0.903 | 0.997 |
| unc_area | 20 | 0.940 | 0.957 | 0.889 | 0.996 |
