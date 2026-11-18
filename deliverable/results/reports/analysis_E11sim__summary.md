HSV (E0r): rate 7.22% | recall 77.5% | incidents missed by HSV 23/102 | spurious 29.4% (558 calls)
configs passing (a) recall_miss>=30 (b) rate<=17 (c) spur<=29.4: 0

Passing (hsv_or_unc), sorted by recall HSV-missed then rate:
| D | r | N | C | mode | rate % | recall % | recall HSV-missed % | spurious % |
|---|---|---|---|---|---|---|---|---|

Best hsv_or_unc by recall HSV-missed at rate<=17 (any spurious):
| D | r | N | C | mode | rate % | recall % | recall HSV-missed % | spurious % |
|---|---|---|---|---|---|---|---|---|
| 2 | 0.5 | 1 | 5 | hsv_or_unc | 14.34 | 96.1 | 82.6 (19/23) | 53.3 |
| 2 | 0.5 | 3 | 5 | hsv_or_unc | 14.34 | 96.1 | 82.6 (19/23) | 53.3 |
| 6 | 0.5 | 1 | 5 | hsv_or_unc | 14.20 | 95.1 | 78.3 (18/23) | 53.2 |
| 6 | 0.5 | 3 | 5 | hsv_or_unc | 14.20 | 95.1 | 78.3 (18/23) | 53.2 |
| 4 | 0.5 | 1 | 5 | hsv_or_unc | 14.21 | 95.1 | 78.3 (18/23) | 53.3 |
| 4 | 0.5 | 3 | 5 | hsv_or_unc | 14.21 | 95.1 | 78.3 (18/23) | 53.3 |
| - | 0.5 | 1 | 5 | hsv_or_unc | 14.19 | 94.1 | 73.9 (17/23) | 53.3 |
| - | 0.5 | 3 | 5 | hsv_or_unc | 14.19 | 94.1 | 73.9 (17/23) | 53.3 |
| 2 | - | 1 | 5 | hsv_or_unc | 14.12 | 93.1 | 69.6 (16/23) | 53.1 |
| 2 | 0.8 | 1 | 5 | hsv_or_unc | 14.12 | 93.1 | 69.6 (16/23) | 53.1 |

Note (2026-11-14): criterion (c) dropped by the user — the cost of spurious calls is already in criterion (b)
(call rate ≤ 17 %) and the "spurious" definition is only approximate. With (a)+(b), chosen for E11a:
hsv_or_unc, D=2, r=0.5, N=1, C=5 (14.34 % calls, recall 96.1 %, HSV-missed 19/23).
