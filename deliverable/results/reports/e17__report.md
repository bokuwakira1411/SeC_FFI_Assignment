# E17a – Dùng riêng nhánh concept khi đã mất dấu (160 video)

Ngày 2026-11-15. Seed 0. Chạy 15:02–16:20, một mình trên GPU (không song song với fps160 hay E17c).
Số liệu: `report.py` → `report_numbers.txt`; từng lần gọi lúc mất dấu: `lost_calls.csv`. Bước 0: `step0.md`.

**Thuật ngữ**
- *Mất dấu*: mask rỗng hoặc object_score_logit ≤ 0 ở 2 frame được track liền trước (không tính frame đầu).
- *Mask trộn*: decoder nhận (a+b)/2 như gốc. *Mask concept*: decoder chỉ nhận b (nhánh concept).
- *Nhận*: dùng mask concept làm output của frame (mask khác rỗng, pred_iou ≥ 0.78, logit > 0) và thêm frame vào bank.
- *Nhận nhầm*: nhận mà GT rỗng (bịa ra đối tượng) hoặc J < 0.5 (bắt nhầm vật khác).
- *Sự cố*: J giảm > 0.3 so với trung bình 5 frame trước. *Hồi phục*: trong 10 frame sau, J lên lại ≥ trung bình đó − 0.1.
- *CI*: khoảng tin cậy 95 % bootstrap (10 000 lần, theo object và theo video). `*` = CI không chứa 0.

## Sai lệch so với thiết kế
- Điều kiện nhận: **logit > 0** thay cho > 1 (duyệt trước khi chạy). Lý do: > 0 là ngưỡng "có mặt" gốc của SAM2;
  lần thử 3 video cho thấy > 1 chặn các mask concept có J 0.96–0.99. Bảng nhận nhầm tách riêng nhóm logit (0, 1].
- Tham chiếu E0, E10a: lượt fps160 (cùng code, seed 0; mask E10a trùng từng bit với các lần chạy trước).
- Cắt cảnh cho rest120 chưa có sẵn: đã chạy PySceneDetect (`min_scene_len=5`, như subset40) → `scenecuts_msl5_all160.json`.

## 1. Kiểm tra cơ chế (các lần gọi lúc mất dấu, GT không rỗng)

| Tập | Lần gọi lúc mất dấu (GT rỗng) | GT không rỗng | J TB trộn → concept | Mask khác rỗng trộn → concept | concept tốt hơn / kém hơn |
|---|---|---|---|---|---|
| **rest120** | 2366 (1555) | 811 | 0.207 → **0.300** | 24.3 % → 37.6 % | 123 / 57 |
| all160 | 3051 (2054) | 997 | 0.210 → 0.311 | 24.6 % → 40.2 % | 162 / 69 |
| subset40 | 685 (499) | 186 | 0.222 → 0.358 | 25.8 % → 51.6 % | 39 / 12 |

Ở các lần được nhận (rest120, GT không rỗng): J trộn 0.64 → J concept 0.90.

## 2. Nhận nhầm

| Tập | Logit lúc nhận | Số lần nhận | GT rỗng (bịa) | J < 0.5 (nhầm vật) | Tỷ lệ nhận nhầm | J concept trung vị |
|---|---|---|---|---|---|---|
| **rest120** | (0, 1] | 71 | 20 | 6 | **36.6 %** | 0.918 |
| | > 1 | 194 | 25 | 8 | 17.0 % | 0.969 |
| | tất cả | 265 | 45 | 14 | **22.3 %** | 0.964 |
| all160 | (0, 1] | 92 | 29 | 10 | 42.4 % | 0.891 |
| | > 1 | 247 | 26 | 14 | 16.2 % | 0.966 |
| | tất cả | 339 | 55 | 24 | 23.3 % | 0.960 |
| subset40 | (0, 1] | 21 | 9 | 4 | 61.9 % | 0.000 |
| | > 1 | 53 | 1 | 6 | 13.2 % | 0.954 |
| | tất cả | 74 | 10 | 10 | 27.0 % | 0.904 |

Nhóm mới được nhận do hạ ngưỡng (logit 0–1) **xấu**: tỷ lệ nhận nhầm gấp khoảng 2 lần nhóm logit > 1. Phần lớn là bịa ra đối tượng khi GT rỗng.

Ví dụ (rest120):
- `dIuH0AhcX5Q` object 001, frame 88: GT rỗng, mask concept pred_iou 0.94, logit 1.16 → bịa ra đối tượng.
- `E_UbFKdWFl4` object 000, frame 131: GT rỗng, pred_iou 0.98, logit 0.50 → bịa ra đối tượng (nhóm mới).
- `mGkTa8gcANY` object 000, frame 107: GT không rỗng, J concept 0, pred_iou 0.79, logit 4.6 → bắt nhầm vật khác.

## 3. Bảng chính

fps = số frame / thời gian xử lý video, đã trừ thời gian log của E17 (`e17_log_ms`).

| Tập | Run | J&F | Δ vs E0 [object CI; video CI] | Δ vs E10a [object CI; video CI] | tăng / giảm vs E10a | % frame gọi concept | fps (trung vị; TB) | frame được nhận | frame thêm vào bank |
|---|---|---|---|---|---|---|---|---|---|
| **rest120** | E0 | 70.0 | – | – | – | 7.5 | 10.48; 9.78 | – | 651 |
| | E10a | 72.2 | +2.3 [+1.0, +3.7] *; [+1.0, +3.7] * | – | – | 26.0 | 7.99; 7.63 | – | 674 |
| | **E17a** | **77.4** | +7.4 [+4.7, +10.3] *; [+4.7, +10.4] * | **+5.1 [+2.8, +7.6] *; [+2.8, +7.6] *** | 61 / 28 | 26.0 | 6.71; 6.57 | 265 | 866 (113 do nhận) |
| all160 | E0 | 69.9 | – | – | – | 7.4 | 10.50; 9.92 | – | 850 |
| | E10a | 72.6 | +2.7 [+1.6, +4.0] *; [+1.5, +4.0] * | – | – | 25.7 | 8.19; 7.75 | – | 878 |
| | E17a | 78.6 | +8.7 [+6.1, +11.3] *; [+6.0, +11.4] * | +5.9 [+3.8, +8.3] *; [+3.7, +8.3] * | 83 / 36 | 25.7 | 7.09; 6.84 | 339 | 1150 (152 do nhận) |
| subset40 (phụ) | E0 | 69.8 | – | – | – | 7.2 | 10.75; 10.32 | – | 199 |
| | E10a | 73.9 | +4.1 [+1.6, +6.8] *; [+1.5, +7.1] * | – | – | 24.6 | 8.33; 8.12 | – | 204 |
| | E17a | 82.4 | +12.6 [+6.8, +18.9] *; [+6.7, +19.6] * | +8.5 [+3.4, +14.1] *; [+3.4, +14.5] * | 22 / 8 | 24.6 | 7.85; 7.65 | 74 | 284 (39 do nhận) |

J&F theo đoạn (đầu / hai / còn lại), rest120: E10a 86.8 / 71.4 / 70.4; E17a 88.3 / 76.0 / 76.5.

**Tốc độ**: E17a chậm hơn E10a khoảng 16 % (rest120). Nguyên nhân chính không phải lần chạy decoder thêm. Bank lớn hơn nên
MLLM nhận nhiều frame ngữ cảnh hơn: trung bình 4.08 so với 3.61 frame, và mỗi lần gọi MLLM có trung vị 182 ms so với 146 ms.
Lưu ý: fps E10a đo trong lượt fps160 chạy xen kẽ, fps E17a đo trong lượt chạy riêng.

## 4. Sự cố, so với E10a

Cột "concept" nghĩa là có gọi concept cho object này trong khoảng ±2 frame quanh sự cố.

| Tập | Run | sự cố (object) | ở cắt cảnh + concept | ở cắt cảnh, không concept | không cắt cảnh + concept | không cắt cảnh, không concept | hồi phục (tất cả) |
|---|---|---|---|---|---|---|---|
| **rest120** | E10a | 442 (125) | 203 (rec. 80/201) | 14 (rec. 8/14) | 158 (rec. 113/158) | 67 (rec. 50/67) | 251/440 = 57.0 % |
| | E17a | 447 (124) | 203 (rec. 106/202) | 13 (rec. 7/13) | 159 (rec. 117/159) | 72 (rec. 57/72) | **287/446 = 64.3 %** |
| all160 | E10a | 562 (164) | 259 (rec. 100/257) | 20 (rec. 12/20) | 202 (rec. 144/202) | 81 (rec. 60/81) | 316/560 = 56.4 % |
| | E17a | 560 (163) | 254 (rec. 135/253) | 19 (rec. 11/19) | 202 (rec. 153/202) | 85 (rec. 66/85) | 365/559 = 65.3 % |
| subset40 | E10a | 120 (39) | 56 (rec. 20/56) | 6 (rec. 4/6) | 44 (rec. 31/44) | 14 (rec. 10/14) | 65/120 = 54.2 % |
| | E17a | 113 (39) | 51 (rec. 29/51) | 6 (rec. 4/6) | 43 (rec. 36/43) | 13 (rec. 9/13) | 78/113 = 69.0 % |

Bảng E10a trên subset40 khớp `results/summary.md` → cách tính sự cố tái tạo đúng.

**10 video tệ nhất của E0** (xếp theo J&F của E0 trên all160, lấy trong từng tập), J&F trung bình theo object:

| Tập | E0 | E10a | E17a |
|---|---|---|---|
| all160 | 19.1 | 28.7 | 48.8 |
| rest120 | 22.9 | 31.1 | 47.2 |
| subset40 | 36.3 | 41.4 | 77.3 |

## 5. Đối chiếu tiêu chí đặt trước (rest120)

| Tiêu chí | Kết quả rest120 | |
|---|---|---|
| (a) J mask concept > mask trộn ở các lần gọi lúc mất dấu | 0.300 > 0.207 | **Đạt** |
| (b) Tỷ lệ hồi phục sự cố cao hơn E10a | 64.3 % > 57.0 % | **Đạt** |
| (c) J&F không kém E10a quá 0.5 | +5.1 (CI [+2.8, +7.6]) | **Đạt** |
| (d) Tỷ lệ nhận nhầm < 20 % số lần nhận | 22.3 % (59/265) | **Trượt** |

all160 và subset40 cho cùng kết luận: (a), (b), (c) đạt, (d) trượt (23.3 %, 27.0 %).

**Giải thích (d)**: nhận nhầm tập trung ở nhóm logit (0, 1], là nhóm mới được nhận do hạ ngưỡng. Nhóm này nhận nhầm 36.6 %,
trong khi nhóm logit > 1 chỉ 17.0 %. Phần lớn nhận nhầm là bịa ra đối tượng khi GT rỗng (45/59 trên rest120): đối tượng thật
đã ra khỏi khung hình, nhưng nhánh concept vẫn khoanh một vật giống nó.

Vì (d) trượt, theo quy tắc đặt trước, tôi **không đề xuất** coi E17a là đạt. Điều này đúng dù J&F tăng lớn và có ý nghĩa.
Con số 17.0 % của nhóm logit > 1 chỉ là quan sát sau khi đã chạy: nếu thực sự dùng ngưỡng > 1, quỹ đạo track sẽ khác,
nên con số đó không thay được một lần chạy với ngưỡng > 1. E17b chưa chạy.

## 6. Hạn chế
- Ngưỡng pred_iou 0.78 được chọn trên subset40. Ngưỡng logit được đổi sau khi xem 3 video (đều thuộc subset40). Vì vậy
  rest120 là tập đánh giá chính.
- Định nghĩa "mất dấu" (2 frame, logit ≤ 0) là chọn tay, chưa thử giá trị khác.
- Nhận nhầm chỉ đo tại frame được nhận. Chưa đo thiệt hại về sau: frame nhận nhầm vẫn vào pixel memory và bank.
- Tỷ lệ % frame gọi concept giống E10a, nhưng E17a chậm hơn khoảng 16 % vì bank lớn hơn.
- subset40 chỉ có 44 object; CI ở đó rộng.

## Ghi chú vận hành
- E17a bắt đầu lúc 15:02, không phải ngay sau khi fps160 xong lúc 13:02. Vòng chờ `pgrep -f run_interleaved.py` vẫn thấy
  một process khớp tên cho tới 15:02. Tôi không xác định được đó là process nào.
- Lúc 15:36, một phiên khác sửa `inference/sam2_video_predictor.py`, `inference/modeling_sec.py` và `vos_inference.py`
  (thêm E17c: `--fusion_w`, `--fusion_sweep`). Process E17a đã nạp code từ 15:02 và không chạy lại lần nào, nên E17a dùng
  đúng code của bước 0. Hàng đợi E17c (`experiments/e17c/`) đợi E17a xong mới chạy, nên GPU không bị dùng chung.
