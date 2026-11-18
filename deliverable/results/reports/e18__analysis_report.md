# E18 – bước 0 và bước 1: đọc code, cài `--e17_commit`, phân tích log có sẵn (2026-11-15)

**Thuật ngữ**
- *Pixel memory*: memory của SAM2. Mỗi frame đã track được mã hoá (feature ảnh + mask) và lưu lại. Frame mới "nhìn" các frame này qua memory_attn (nhánh a).
- *Object pointer*: một vector tóm tắt đối tượng ở mỗi frame. Nó cũng nằm trong memory.
- *Keyframe bank*: các frame (ảnh + mask) đưa cho MLLM để tạo concept (nhánh b).
- *Nhận*: E17a dùng mask concept vì nó qua cổng (không rỗng, pred_iou ≥ 0.78, logit > 0).
- *Nhận nhầm* gồm hai loại: *bịa* (GT rỗng) và *bắt nhầm vật* (GT không rỗng, J < 0.5).
- *frame-J*: tổng J theo frame. 1 frame-J tương đương một frame từ J = 0 lên J = 1.
- *AUC*: xác suất một lần nhận đúng có tín hiệu cao hơn một lần nhận nhầm. 0.5 là ngẫu nhiên, 1 là tách hoàn toàn.

## Bước 0 – Đọc code

### 0.1 Vị trí trong code

| Việc | Nơi |
|---|---|
| Decoder cho ra mask, pred_iou, logit, object pointer | `inference/sam2_video_predictor.py:330` (`_forward_sam_heads`), gán vào `current_out` ở `:432-440` |
| **Mã hoá mask vào pixel memory** (memory encoder) | `inference/sam2_video_predictor.py:444` gọi `_encode_memory_in_output` → `inference/sam2/modeling/sam2_base.py:787` → `_encode_new_memory` (`sam2_base.py:676`; frame có logit ≤ 0 được cộng `no_obj_embed_spatial`, `:716-720`) |
| Gói gọn output của frame (maskmem_features, obj_ptr, ious, logit) | `inference/sam2_video_predictor.py:522` (`compact_current_out`) |
| **Lưu vào output dict** | `inference/modeling_sec.py:805` (`output_dict[storage_key][frame_idx] = current_out`) |
| **Chọn frame cho pixel memory** khi xử lý frame mới | `inference/sam2_video_predictor.py:105` (frame có prompt: `select_closest_cond_frames`); `:118-125` (frame trước: `valid_indices`); `:127-138` (gán slot); `:189` (object pointer, cùng `valid_indices`) |
| E17a: thay output bằng mask concept | `inference/sam2_video_predictor.py:378` |
| **E17a thêm frame vào keyframe bank** | `inference/modeling_sec.py:940-948` (`bank_reason = "added_concept"`). Luật bank thường của E10a nằm ngay trước, ở `:875` trở đi |
| Mask được trả ra để chấm điểm | `inference/modeling_sec.py:967` (`yield`) |

**Phát hiện quan trọng cho Q4 (chưa có trong đề bài):** SeC **đã lọc sẵn** pixel memory.
- `valid_indices` (`sam2_video_predictor.py:118-125`) chỉ lấy các frame trước có pred_iou > 0.3 **và** logit > 0. Nó lấy tối đa 21 frame tốt gần nhất (cấu hình SeC: `num_maskmem = 22`, `saved_models/SeC-4B/config.json`).
- Ngoại lệ duy nhất: frame f−1 luôn được thêm vào, dù tốt hay xấu (`:124-125`).
- Vậy khi mất dấu, pixel memory **không** đầy frame mask rỗng. Nó gồm frame có prompt, khoảng 17–21 frame tốt cuối cùng, và đúng 1 frame xấu là f−1. Log E10a xác nhận điều này (mục A3 bên dưới).

### 0.2 Cách cài `--e17_commit {full, pixel, output}`

Flag cần có `--concept_only_lost`. Mặc định là `full`, tức E17a.

- `full`: như E17a.
- `pixel`: như E17a, nhưng bỏ bước thêm thẳng vào bank (`modeling_sec.py:940`, điều kiện `e17_commit == "full"`).
  - Mask được nhận vẫn là output và vẫn được mã hoá vào pixel memory.
  - Bank vẫn theo luật thường của E10a (trigger HSV, mask không rỗng, score > 1). Luật này áp lên output của frame, mà output ở đây là mask concept.
- `output`: trong `_track_step` không thay `sam_outputs` (`sam2_video_predictor.py:378`).
  - Memory, object pointer, ious, logit, lịch sử mất dấu và luật bank đều lấy từ mask trộn, y như E10a.
  - Riêng mask trả ra để chấm (`_out_masks`, `modeling_sec.py:833, 855-861`) được thay bằng mask concept. Mask này đi qua cùng đường xử lý: lấp lỗ (fill holes) ở độ phân giải thấp, rồi resize về kích thước video.
  - Cột log `mask_px` ghi diện tích của mask được chấm.

### 0.3 Kiểm tra tương đương (3 video smoke, seed 0)

Script chạy: `experiments/e18/run_step0.sh`. Script kiểm tra: `experiments/e18/check_step0.py`. Video: `experiments/e17/subset_smoke.txt`.

| Run | Kiểm tra | Kết quả | J&F (3 video) |
|---|---|---|---|
| `full` | mask trùng từng bit với E17a | **0 file khác** | 67.3 (= E17a) |
| `output` | trạng thái tracking ở mọi frame trùng E10a (object_score_logit, pred_iou, triggered, frame trong pixel memory, frame trong bank, frame ngữ cảnh MLLM) | **0/580 frame khác** | 29.1 (E10a 25.7) |
| `output` | mask của frame không được nhận trùng E10a | **0 khác**. 24/24 frame được nhận có mask khác E10a | |
| `pixel` | chỉ chạy thử, không có tham chiếu. Không có `added_concept` trong bank | 6 lần nhận, 0 `added_concept` | 57.8 |

- Ba video này là các video tệ nhất của E0, lại thuộc subset40, nên J&F ở đây chỉ để kiểm tra code.
- Dù vậy, thứ tự output (29.1) < pixel (57.8) < full (67.3) đã gợi ý rằng phần lớn lợi ích không đến từ riêng output. Bước 2 sẽ kiểm chứng trên 160 video.

## Bước 1 – Phân tích log có sẵn (chỉ CPU)

Nguồn:
- E17a: `experiments/e17/E17a`.
- E10a: `experiments/fps160/E10a`.
- Số liệu: `experiments/e18/analysis.py` → `analysis_numbers.txt`; từng lần nhận ở `accepts.csv`.
- J tính theo số frame. Frame trước lần chú thích đầu tiên của object không được chấm và bị bỏ qua.

**Đặt trước:** định nghĩa cửa sổ, ba loại nhận, các tín hiệu, ngưỡng AUC 0.70 và luật "giữ ≥ 90% lần nhận đúng". Tất cả lấy từ đề bài.
**Chọn sau khi xem dữ liệu:** không có.

### A1. Nhận sai hại bao lâu

- *Cửa sổ*: từ frame nhận tới trước lần nhận kế tiếp của cùng object, hoặc tới hết video.
- ΔJ = J(E17a) − J(E10a), cộng trong cửa sổ.
- *Độ dài thiệt hại*: số frame tới khi J(E17a) ≥ J(E10a) trở lại.

**rest120** (đánh giá chính):

| Loại | Số lần | ΔJ TB / lần | ΔJ trung vị / lần | Tổng ΔJ | ΔJ ngay tại frame nhận (TB) | Độ dài cửa sổ TB | Độ dài thiệt hại: trung vị; TB |
|---|---|---|---|---|---|---|---|
| Nhận đúng | 206 | **+7.36** | +0.03 | **+1515** | +0.31 | 47 | 0; 1.6 |
| Bịa (GT rỗng) | 45 | **−6.43** | −3.00 | **−290** | −0.62 | 51 | **3; 7.9** |
| Bắt nhầm vật | 14 | −0.64 | 0.00 | −9 | −0.04 | 54 | 0; 0.1 |
| Tất cả | 265 | +4.59 | +0.01 | +1217 | +0.14 | 48 | 0; 2.6 |

all160:
- Đúng: 260 lần, +8.46 / lần, tổng +2199.
- Bịa: 55 lần, −6.05 / lần, tổng −333, thiệt hại trung vị 3 frame.
- Bắt nhầm: 24 lần, +1.04 / lần.

subset40:
- Đúng: 54 lần, +12.66 / lần.
- Bịa: 10 lần, −4.32 / lần.
- Bắt nhầm: 10 lần, +3.38 / lần.

Đọc bảng:
- **Bịa là loại gây hại.**
  - Trung bình mỗi lần mất khoảng 6 frame-J.
  - Phần lớn hồi lại nhanh: 62% lần bịa hết hại trong ≤ 5 frame.
  - Nhưng có một đuôi dài: 5/45 lần hại hơn 20 frame. Ví dụ `cIX8usnGj7s` frame 52 (−55 frame-J, 42 frame) và `IppaafjwUUA` frame 105 (−43 frame-J, 49 frame). Cả hai có logit > 2.5.
- **Bắt nhầm vật gần như vô hại.** Ở các frame đó E10a cũng đã mất đối tượng (J mask trộn TB 0.09). Mask sai chỉ thay một mask cũng sai.
- **Tổng lợi gấp khoảng 5 lần tổng hại:** +1515 so với −299 frame-J trên rest120.

**28 object bị giảm trên rest120** (J&F của E17a < E10a):
- Cả 28 object đều có ít nhất một lần nhận.
- 20/28 object có ít nhất một lần nhận nhầm.
- 97.7% phần J bị giảm (chỉ cộng các frame có ΔJ < 0) nằm trong cửa sổ sau các lần nhận nhầm.

**Hạn chế của A1:**
- Sau lần nhận đầu tiên, hai đường track đã khác nhau. Vì vậy việc gán ΔJ cho từng lần nhận chỉ là gần đúng.
- Cửa sổ kéo tới lần nhận kế tiếp (TB khoảng 50 frame). Nó có thể gom cả những khác biệt không do lần nhận đó gây ra. Con số 97.7% vì vậy là cận trên.

### A2. Tín hiệu phát hiện nhận nhầm (trên các lần E17a nhận)

| Tín hiệu | subset40: đúng vs nhầm | subset40: đúng vs bịa | rest120: đúng vs nhầm | rest120: đúng vs bịa |
|---|---|---|---|---|
| cos max tới bank (`con_maxcos`) | 0.614 | 0.644 | 0.573 | 0.548 |
| pred_iou | **0.767** | 0.826 | 0.700 | 0.673 |
| object_score_logit | **0.769** | 0.902 | 0.646 | 0.666 |
| diện tích mask / frame | 0.540 | 0.504 | 0.498 | 0.454 |

- Số lần nhận: subset40 có 54 đúng, 10 bịa, 10 bắt nhầm. rest120 có 206 đúng, 45 bịa, 14 bắt nhầm.
- **Cos trung bình tới bank không có.** E17a chỉ log cos max. Muốn có cos trung bình phải chạy lại GPU. Tôi chưa làm.
- AUC của logit và pred_iou chỉ đo **trong** nhóm đã qua cổng (logit > 0, pred_iou ≥ 0.78), nên giá trị bị kéo xuống.

**Ngưỡng chọn trên subset40, giữ ≥ 90% lần nhận đúng** (mô phỏng offline, không tính hiệu ứng lan):

| Tín hiệu | Ngưỡng | subset40: giữ đúng / chặn nhầm | rest120: giữ đúng | rest120: chặn nhầm (bịa; bắt nhầm) | rest120: tỷ lệ nhận nhầm còn lại |
|---|---|---|---|---|---|
| pred_iou | ≥ 0.840 | 48/54; 6/20 | 193/206 (94%) | 8/59 (7/45; 1/14) | 20.9% |
| logit | ≥ 0.717 | 48/54; 9/20 | 179/206 (87%) | 18/59 (14/45; 4/14) | 18.6% |

- AUC trên subset40 (0.77) không giữ được trên rest120 (0.70 và 0.65). Đây cùng hiện tượng với tỷ lệ nhận nhầm ở E17c.
- Ngưỡng logit ≈ 0.72 gần với ngưỡng > 1 ban đầu của E17a. Nó chặn khoảng 1/3 số lần bịa, đổi lại mất 13% lần nhận đúng.

### A3. Pixel memory của E10a lúc mất dấu (cho Q4 và E18c)

- Số frame mất dấu trên all160 là 13 389.
- Pixel memory TB có 18.1 frame, không tính frame có prompt.
- **Đúng 1.00 frame xấu** (mask rỗng hoặc logit ≤ 0), và frame xấu đó **luôn là f−1**.
- Điều này khớp với code ở mục 0.1: memory đã lọc sẵn, chỉ f−1 lọt qua.

## Kết luận ngắn cho Q2

1. **Hại chủ yếu đến từ bịa (GT rỗng), không phải bắt nhầm vật.**
   - Mỗi lần bịa mất TB khoảng 6 frame-J.
   - Trung vị thời gian hại là 3 frame. Nhưng có khoảng 10% lần hại kéo dài 20–50 frame.
   - Bắt nhầm vật gần như không thêm hại, vì E10a cũng đang sai ở đó.
2. **Lợi > hại khoảng 5 lần.** Gần như toàn bộ mức giảm của 28 object nằm sau các lần nhận nhầm (cận trên).
3. **Không có tín hiệu nào tốt hơn hẳn logit.**
   - pred_iou và logit đạt AUC ≥ 0.70 trên subset40, nhưng chỉ 0.65–0.70 trên rest120.
   - Cos max tới bank và diện tích mask gần như ngẫu nhiên (0.50–0.61).
   - Chặn offline bằng logit ≥ 0.72 giảm tỷ lệ nhận nhầm rest120 từ 22.3% xuống 18.6%, nhưng mất 27 lần nhận đúng.

## Hệ quả cho các bước sau (cần bạn xem)

- **E18b+**: theo luật đặt trước (AUC subset40 ≥ 0.70), cả pred_iou và logit đều đủ điều kiện. Nếu chạy, tôi đề xuất **logit ≥ 0.717**, vì nó chặn được nhiều lần bịa hơn trên subset40 (9/20 so với 6/20). Lưu ý AUC của nó giảm mạnh trên rest120.
- **E18c**: theo A3, "bỏ frame xấu khỏi pixel memory" thực chất **chỉ bỏ f−1**, vì code đã lọc phần còn lại. Thay đổi này nhỏ hơn nhiều so với giả định của Q4, nên E18c nhiều khả năng cho kết quả gần E10a. Bạn có muốn giữ E18c như thiết kế không?
