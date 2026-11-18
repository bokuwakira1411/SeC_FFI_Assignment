# Cổng chất lượng cho frame thêm vào bank (subset40, offline)

Ngày 2026-11-15. Không chạy GPU: chỉ dùng log `frames/*.csv` và mask đã lưu.
Script (chạy từ `SeC/`): `added_frames.py <out.csv>` → `added.csv`; `an.py added.csv` (phân phối J, AUC, kết hợp);
`gate.py added.csv` (tìm ngưỡng cổng).

Ghi chú tín hiệu: cả 5 tín hiệu đều có hoặc tính được, không phải chạy lại gì, nhưng
- độ ổn định IoU(t, t+1..t+3) dùng mask các frame sau → nếu làm cổng phải hoãn việc thêm frame vào bank 3 frame;
- khoảng cách tới cắt cảnh lấy từ PySceneDetect offline (`scenecuts_msl5.json`), không có trong log lúc chạy.

## 1. J của các frame được thêm vào bank

| Run | Số frame | J trung vị | J < 0.5 | J < 0.7 |
|---|---|---|---|---|
| E0r (luật gốc) | 199 | 0.968 | 6.0% | 9.0% |
| E10a (luật gốc) | 204 | 0.970 | 7.4% | 10.3% |
| E12 | 519 | 0.967 | 9.2% | 12.1% |
|  – trong đó: frame thêm theo HSV | 198 | 0.972 | 3.5% | 7.1% |
|  – trong đó: frame thêm định kỳ | 321 | 0.965 | 12.8% | 15.3% |
| E14 | 520 | 0.954 | 10.8% | 16.2% |

- E12 và E14 nạp vào bank nhiều frame xấu hơn luật gốc. Ở E12, phần xấu thêm ra nằm hết ở các frame thêm định kỳ;
  các frame thêm theo HSV còn tốt hơn luật gốc.
- Frame xấu dồn vào một số video: 147 frame xấu của E12+E14 nằm trong 21 video, riêng 5 video chiếm 48%.

## 2. AUC của từng tín hiệu

AUC là xác suất một frame tốt (J ≥ 0.7) chọn ngẫu nhiên có giá trị tín hiệu cao hơn một frame xấu (J < 0.7) chọn
ngẫu nhiên. 0.5 = không phân biệt được, 1.0 = tách hoàn hảo, dưới 0.5 = có tác dụng theo chiều ngược lại.

| Tín hiệu | E12+E14 | E12 | E14 | E0r+E10a |
|---|---|---|---|---|
| pred_iou | 0.73 | 0.70 | 0.75 | 0.79 |
| object_score_logit | 0.63 | 0.65 | 0.62 | 0.77 |
| diện tích / TB 5 frame trước | 0.60 | 0.59 | 0.61 | 0.62 |
| ổn định IoU(t, t+1..t+3) | 0.56–0.57 | ~0.59 | ~0.53 | ~0.60 |
| khoảng cách tới cắt cảnh | 0.45 (gần cắt cảnh lại tốt hơn) | 0.44 | 0.45 | 0.54 |

- Logit có AUC thấp một phần vì luật bank vốn đã chỉ nhận frame có logit > 1, nên giá trị logit bị cắt mất phần dưới.
- Tính riêng trong từng video, pred_iou và logit vẫn có AUC khoảng 0.71–0.73: tín hiệu không chỉ tách video dễ khỏi
  video khó.

Kết hợp: pred_iou + logit (trung bình thứ hạng) 0.72; thêm diện tích vẫn 0.72; hồi quy logistic kiểm định chéo theo
video 0.67. Không tổ hợp nào hơn pred_iou dùng một mình.

## 3. Luật cổng đề xuất

Chỉ thêm frame vào bank khi `pred_iou ≥ 0.78` (giữ điều kiện logit > 1 đang có). Ngưỡng thứ hai cho logit không cải
thiện gì nên bỏ.

Ước tính trên subset40 (cũng là dữ liệu dùng để chọn ngưỡng, nên còn lạc quan):

| | Frame tốt bị bỏ sót | Frame xấu bị chặn | Frame J < 0.5 bị chặn | Tỷ lệ J < 0.7 sau cổng |
|---|---|---|---|---|
| E12+E14 | 28/890 (3%) | 45/149 (30%) | 23/104 | 14.3% → 10.6% |
| E12 | 14/455 | 17/64 | 8/48 | 12.1% → 9.4% |
| E14 | 14/435 | 28/85 | 15/56 | 16.2% → 11.7% |
| E0r / E10a (đối chiếu) | 7 / 5 | 5 / 6 | — | 9.0% → 7.0% / 10.3% → 7.8% |

Cổng giúp có thật nhưng vừa phải: chặn khoảng 1/3 frame xấu, mất khoảng 3% frame tốt; sau cổng E12/E14 xuống ngang
mức luật gốc chưa lọc. Ngưỡng 0.81 chặn thêm 7 frame xấu nhưng bỏ sót thêm 18 frame tốt, không đáng.

## 4. Oracle (E13a/E13b): mask của run thường tại frame oracle thêm vào bank

Script `oracle_frames.py` → `oracle_frames.csv` (theo frame), `oracle_objects.csv` (theo object). Chỉ tính frame
`added_gt`; mask lấy từ run thường tương ứng (E13a → E0r, E13b → E10a). Mức tăng = J&F oracle − J&F run thường
(`results/per_object_jf.csv`), 44 object.

| | Số frame | Rỗng | Sai (J < 0.5) | Đúng (J ≥ 0.5) |
|---|---|---|---|---|
| E13a / E0r | 450 | 187 (42%) | 13 (3%) | 250 (56%) |
| E13b / E10a | 443 | 145 (33%) | 15 (3%) | 283 (64%) |
| 10 video tệ nhất E0: E13a / E0r | 144 | 111 | 6 | 27 |
| 10 video tệ nhất E0: E13b / E10a | 136 | 97 | 7 | 32 |

Mức tăng theo số frame "rỗng hoặc sai" của object:

| Số frame rỗng/sai | E13a: số object | tăng TB | trung vị | E13b: số object | tăng TB | trung vị |
|---|---|---|---|---|---|---|
| 0 | 11 | +1.9 | 0.0 | 14 | +0.2 | +0.2 |
| 1–2 | 9 | +0.8 | 0.0 | 12 | +1.0 | 0.0 |
| 3–5 | 10 | −1.6 | 0.0 | 7 | +3.1 | +0.1 |
| ≥ 6 | 14 | +12.3 | +5.4 | 11 | +25.4 | +28.4 |

Spearman (số frame rỗng/sai, mức tăng): E13a 0.31 (p = 0.04), E13b 0.43 (p = 0.004). Theo tỷ lệ: 0.27 (p = 0.08),
0.41 (p = 0.006).

10 video tệ nhất E0 (mỗi video 1 object): tăng TB +15.6 (E13a), +28.4 (E13b); 10/10 (E13a) và 9/10 (E13b) object có
≥ 6 frame rỗng/sai. Trong nhóm 10 video này Spearman ≈ 0.1 (không ý nghĩa): mọi object đều ở mức cao, không còn
khác biệt để xếp hạng.
