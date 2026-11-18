# E17 – bước 0 (2026-11-15)

- Tiền đề: lúc train decoder nhận output `token_attn` một mình (`training/sec/models/sec.py:262-279`,
  `training/sec/models/sam2_train.py:93-128`); token_attn và decoder đóng băng (`sec_4b.py`, `frozen_token_attn=True`).
- Fusion inference: `inference/sam2_video_predictor.py:305-319` (`pix_feat = (b + a) / 2`); pred_iou / logit:
  `sam2_video_predictor.py:376-394`, log ở `inference/modeling_sec.py` (`pred_iou`, `object_score_logit`).
- Tương đương (chk, E10a flags, E17 tắt, 3 video subset_e10chk): 0/373 file mask khác E10a; J&F 93.0 = TB 96.7/90.1/92.3.
  (Queue log báo nhầm 7464 do lỗi vòng lặp so sánh; đã sửa, xem queue.log.)
- Norm (norms run, 3 video tệ nhất E0, 126 lần gọi concept): norm/pixel |a| 18.05, |b| 17.94, |(a+b)/2| 16.91;
  |b|/|a| 0.994 (0.983–1.004); |(a+b)/2|/|a| 0.937 (0.874–0.975); cos(a,b) 0.77 (0.69 khi không mất dấu, 0.81 khi mất dấu).
- Smoke E17a (cùng run): 71 lần gọi lúc mất dấu, 54 có GT không rỗng: J TB mask trộn 0.033, mask concept 0.172;
  mask khác rỗng 2 vs 11; 9 mask concept khác rỗng qua pred_iou ≥ 0.78 (8 có J ≥ 0.5) nhưng chỉ 3 qua logit > 1 → 3 lần nhận.

## Sai lệch so với thiết kế (duyệt ngày 2026-11-15, trước khi chạy E17a)
- Điều kiện nhận mask concept: pred_iou ≥ 0.78 và **object_score_logit > 0** (thiết kế ban đầu: > 1).
- Lý do: > 0 là ngưỡng "có mặt" gốc của SAM2; lần thử 3 video cho thấy > 1 chặn các mask concept có J 0.96–0.99
  (JJNUDzMqeFU frame 60, 75, 139, 140, 185; logit 0.34–0.84). Rủi ro đã thấy: frame 136 cùng video, GT rỗng,
  pred_iou 0.86, logit 0.46 → sẽ được nhận với ngưỡng mới. Báo cáo tách riêng các lần nhận có logit trong (0, 1].
- Chạy: `--concept_only_lost --concept_only_logit 0` (cờ code giữ mặc định 1).
