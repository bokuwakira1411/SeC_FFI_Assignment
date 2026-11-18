# Keyframe bank: E12, E13a, E13b, E14 (subset40, 40 video, 44 object)

Ngày chạy: 2026-11-15. Seed 0. Code: `CHANGES.md` (mục 2026-11-15). Cách tính số liệu giống `experiments/results/summary.md`.

**Thuật ngữ**
- **Bank** (`mllm_memory`): các frame tham chiếu, kèm mask, mà LLM nhận để tạo concept. Tối đa 7 frame: frame 0 và 6 frame khác.
- **Concept**: vector ngôn ngữ do LLM sinh ra để hướng dẫn SAM2.
- **Frame đáng tin**: frame có mask dự đoán không rỗng và object_score_logit > 1.
- **Oracle**: run dùng mask GT (đáp án) cho frame đưa vào bank. Oracle cho biết mức trần, không phải phương pháp dùng được thật.
- **CI**: khoảng tin cậy 95 %, tính bằng bootstrap ghép cặp (10.000 lần lấy mẫu lại, seed 0). Có hai cách lấy mẫu: theo object, và theo video (giữ nguyên các object của cùng một video). Dấu `*` nghĩa là CI không chứa 0.
- **τ** (tau): ngưỡng cosine của E14. Ứng viên chỉ được thêm vào bank khi cosine lớn nhất của nó với bank nhỏ hơn τ.

## Cấu hình

| Run | Lịch gọi concept | Bank |
|---|---|---|
| E13a | như E0 (trigger HSV 0.35) | luật gốc, cộng thêm mỗi ≥10 frame một frame dùng **mask GT** (bỏ qua nếu GT rỗng) |
| E12 | như E10a | luật E10a (trigger HSV), cộng thêm mỗi ≥10 frame một frame đáng tin (như E6) |
| E14 | như E10a | theo độ khác biệt. Ứng viên: frame đáng tin, mỗi ≥5 frame kể từ ứng viên trước, hoặc có trigger HSV. Thêm nếu cosine lớn nhất < τ = 0.97. Khi bank đầy, bỏ frame (không phải frame 0) giống ứng viên nhất |
| E13b | như E10a | luật E10a, cộng thêm mỗi ≥10 frame một frame dùng mask GT |

**Kiểm tra tương đương.** Chạy code mới với mọi flag mới tắt (`experiments/bank/chk`) trên 3 video. Mask trùng hệt E0, 0 file khác nhau, nên J&F cũng khớp.

**Chọn τ.** Script `sim_diverse.py` chạy lại luật E14 trên log và mask của E10a, không dùng J&F. Mục tiêu là số frame thêm gần 480 nhất (bằng E6).
- Kết quả: τ = 0.97, mô phỏng thêm 501 frame (τ = 0.96 cho 443, τ = 0.98 cho 569).
- **Số thực tế khi chạy E14 là 520 frame**: 236 frame thêm khi bank chưa đầy, 284 frame thay thế một frame cũ.
- Cosine giữa các descriptor rất cao: trung vị cosine lớn nhất của ứng viên là 0.97. Vì vậy τ phải nằm sát 1.

**GPU.** Trước và trong mọi run, không có job nào khác (`nvidia-smi` chỉ có gnome-remote-desktop, 225 MiB; xem `queue.log`).
- Median thời gian frame không gọi concept: E13a 76 ms, E12 76 ms, E14 76 ms, E13b 77 ms. E0r và E6 cũng 76 ms. Vậy fps của các run mới so được với E0r và E6.
- **E10a đo lúc GPU bị chia (139 ms).** fps của E10a không so được với E12, E14, E13b.

## 1. Bảng chính

| Run | J&F | Δ so với E0 [CI object; CI video] | Đối chứng trực tiếp: Δ [CI object; CI video], số object tăng/giảm | Object tăng/giảm so với E0 | % frame gọi concept | fps |
|---|---|---|---|---|---|---|
| E0 | 69.8 | – | – | – | 7.22 | 9.91 |
| E6 | 71.1 | +1.3 [-0.8, +3.5; -0.8, +3.5] | – | 10 / 8 | 7.22 | 9.55 |
| E10a | 73.9 | +4.1 [+1.6, +6.8 *; +1.5, +7.1 *] | – | 23 / 14 | 24.59 | 4.41 (GPU bị chia) |
| **E13a** | **74.0** | +4.2 [+1.3, +7.6 *; +1.4, +7.8 *] | so với E6: **+2.9** [+0.3, +6.0 *; +0.4, +6.1 *], 15 / 6 | 17 / 6 | 7.22 | 9.37 |
| **E12** | **75.1** | +5.3 [+1.8, +9.3 *; +1.7, +9.5 *] | so với E10a: **+1.2** [-2.0, +4.7; -2.0, +4.7], 19 / 11 | 23 / 11 | 24.59 | 6.81 |
| **E14** | **75.0** | +5.2 [+1.7, +9.3 *; +1.6, +9.5 *] | so với E12: **-0.1** [-2.0, +1.6; -2.0, +1.6], 12 / 13. So với E10a: **+1.1** [-1.7, +4.4; -1.8, +4.3], 15 / 17 | 21 / 16 | 24.59 | 6.97 |
| **E13b** | **81.1** | +11.3 [+6.6, +16.7 *; +6.4, +17.0 *] | so với E10a: **+7.2** [+2.5, +12.5 *; +2.6, +12.6 *], 24 / 8 | 31 / 8 | 24.59 | 6.63 |

Object có J&F bằng nhau không được tính là tăng hay giảm.

## 2. Số liệu về bank

Độ đa dạng bank được tính như sau:
1. Ở mỗi frame, lấy bank đúng như LLM nhận.
2. Tính trung bình cosine giữa mọi cặp descriptor trong bank.
3. Lấy trung bình theo frame của từng object, chỉ dùng các frame có bank ≥ 2 frame.
4. Lấy trung bình trên các object.

Số càng thấp thì bank càng đa dạng. Descriptor là feature Hiera lấy trung bình trong mask, vector 256 chiều, chuẩn hoá L2 (`descriptors.py`). E0 không có log bank. E0r cho mask trùng E0 nên dùng số của E0r.

| Run | Tổng số frame thêm | Số object có bank kẹt ở 1 frame cả video | Độ đa dạng (cosine TB, thấp = đa dạng hơn) | Số object dùng để tính độ đa dạng |
|---|---|---|---|---|
| E0 (= E0r) | 199 | 4 | 0.817 | 40 |
| E6 | 480 | 2 | 0.838 | 42 |
| E10a | 204 | 4 | 0.816 | 40 |
| E13a | 648 (450 bằng mask GT) | 0 | 0.796 | 44 |
| E12 | 519 (321 do luật mỗi 10 frame) | 2 | 0.828 | 42 |
| E14 | 520 (284 là thay thế) | 2 | **0.787** | 42 |
| E13b | 653 (443 bằng mask GT) | 0 | 0.794 | 44 |

Thêm định kỳ (E6, E12) làm bank **kém** đa dạng hơn luật gốc. E14 cho bank đa dạng nhất, kể cả so với hai run oracle. Nhưng E14 không hơn E12 về J&F (mục 1).

## 3. J&F trung bình của 10 video tệ nhất của E0

Tính trung bình J&F theo object.

| E0 | E6 | E10a | E13a | E12 | E14 | E13b |
|---|---|---|---|---|---|---|
| 36.3 | 40.3 | 41.4 | 51.9 | 50.4 | 47.3 | 69.8 |

## 4. Sự cố

Một **sự cố** là frame mà J tụt hơn 0.3 so với trung bình 5 frame trước. Cắt cảnh được phát hiện bằng PySceneDetect với min_scene_len = 5. "Có concept" nghĩa là có gọi concept cho object đó trong khoảng ±2 frame.

| Run | Số sự cố (số object) | Tại cắt cảnh, có concept | Tại cắt cảnh, không concept | Không cắt cảnh, có concept | Không cắt cảnh, không concept |
|---|---|---|---|---|---|
| E0 | 102 (39) | 53 (hồi 15/53) | 6 (hồi 3/6) | 20 (hồi 10/20) | 23 (hồi 14/23) |
| E6 | 97 (37) | 51 (hồi 15/51) | 6 (hồi 3/6) | 20 (hồi 9/20) | 20 (hồi 12/20) |
| E10a | 120 (39) | 56 (hồi 20/56) | 6 (hồi 4/6) | 44 (hồi 31/44) | 14 (hồi 10/14) |
| E13a | 99 (38) | 53 (hồi 18/53) | 6 (hồi 3/6) | 21 (hồi 10/21) | 19 (hồi 13/19) |
| E12 | 108 (37) | 50 (hồi 17/50) | 6 (hồi 5/6) | 39 (hồi 25/39) | 13 (hồi 8/13) |
| E14 | 104 (36) | 52 (hồi 21/52) | 6 (hồi 4/6) | 35 (hồi 21/35) | 11 (hồi 6/11) |
| E13b | 119 (38) | 56 (hồi 32/56) | 6 (hồi 5/6) | 45 (hồi 30/45) | 12 (hồi 8/12) |

"Hồi" (hồi phục) nghĩa là trong 10 frame tiếp theo, J lên lại ≥ (trung bình 5 frame trước − 0.1).

## 5. Đối chiếu với tiêu chí đặt trước

| Tiêu chí | Kết quả | Đạt? |
|---|---|---|
| E13a hơn E0 ≥ +4 thì hướng bank có dư địa lớn; < +2 thì dư địa nhỏ, nên dừng | +4.2 | **Đạt** (dư địa lớn) |
| E12 hơn E10a với CI không chứa 0 thì sửa bank cộng dồn với lịch concept | +1.2, CI [-2.0, +4.7] | **Không đạt** |
| E14 hơn E12 thì chọn frame theo độ khác biệt tốt hơn thêm định kỳ | -0.1 | **Không đạt** |

## 6. Hạn chế

- Chỉ có 44 object, nên CI rộng.
- E12 và E14 được thiết kế sau khi đã xem kết quả trên subset40. Cần xác nhận trên 120 video còn lại.
- Oracle dùng GT nên không phải phương pháp dùng được thật.
- E13a không có điều kiện đáng tin, nên thêm nhiều frame hơn E6 (648 so với 480). Vì vậy E13a so với E6 không hoàn toàn cùng ngân sách.
- τ được chọn bằng mô phỏng trên mask của E10a. Số frame thêm thực tế của E14 là 520, không phải 480. Nó vẫn xấp xỉ E12 (519).
- fps của E10a đo lúc GPU bị chia, nên không so được với E12, E14, E13b.

## 7. Đề xuất cho lượt chạy 160 video

Chạy **E12** cùng với E10a. Lý do: E12 và E14 cho J&F gần như bằng nhau (−0.1). E12 chỉ dùng các flag đã có, không cần descriptor hay τ, và giúp 10 video tệ nhất nhiều hơn (50.4 so với 47.3).
