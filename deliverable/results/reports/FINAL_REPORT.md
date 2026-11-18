# SeC trên SeCVOS: báo cáo cuối các thí nghiệm inference

Ngày đóng băng: 2026-11-15. Chỉ thay đổi inference, không huấn luyện lại. Model: SeC-4B. GPU: RTX 5070 Ti 16 GB.
Chi tiết code ở `CHANGES.md`. Số liệu đầy đủ ở `experiments/results/summary.md` (subset40), `experiments/bank/report.md`
(các thí nghiệm bank) và `experiments/full160/summary160.md` (160 video).

## Thuật ngữ

- **Concept**: vector ngôn ngữ 256 chiều. LLM sinh ra nó từ các frame tham chiếu có tô viền object. Concept được đưa vào SAM2 để giúp nhận lại object.
- **Trigger HSV**: phát hiện chuyển cảnh bằng khoảng cách histogram màu giữa hai frame liên tiếp (ngưỡng 0.35). Code gốc chỉ gọi LLM khi có trigger.
- **Bank** (`mllm_memory`): các frame tham chiếu đưa cho LLM, tối đa 7 frame (frame 0 và 6 frame gần nhất). Luật gốc chỉ thêm frame khi có trigger, mask không rỗng và object_score_logit > 1.
- **Frame đáng tin**: mask dự đoán không rỗng và object_score_logit > 1.
- **J&F**: trung bình của độ trùng vùng (J) và độ khớp đường biên (F), thang 0–100.
- **Δ, CI**: chênh lệch J&F theo cặp trên cùng object. CI 95 % tính bằng bootstrap (10.000 lần lấy mẫu lại, seed 0), lấy mẫu theo object và theo video. Dấu `*` nghĩa là CI không chứa 0.
- **Oracle**: run dùng GT (đáp án). Oracle cho biết mức trần, không phải phương pháp dùng được thật.

## Cách làm

- **Tập phát triển subset40**: 40 video, 44 object, lấy đều theo J&F của run gốc trên 160 video. Mọi thiết kế được làm trên tập này.
- **Tập kiểm chứng rest120**: 120 video, 138 object còn lại. Chỉ dùng một lần, ở cuối, cho E10a và E12.
- **Kiểm tra tương đương**: mỗi đợt sửa code đều được kiểm tra. Khi tắt flag, mask phải trùng hệt run gốc (E0r, E10chk, E11chk, `bank/chk`, `full160/chk_E10a`). Pipeline tất định: decode greedy, chạy lại cho mask giống từng bit.
- **Tiêu chí đặt trước**: E11, I1, I5b′ và các thí nghiệm bank đều có tiêu chí đặt trước khi chạy. Kết quả được đối chiếu đạt hay không đạt, không diễn giải lại.

## Kết quả chính

### 1. Concept có ích và mang thông tin riêng của từng object

| Run (subset40) | J&F | Δ so với E0 |
|---|---|---|
| E0: gốc | 69.8 | – |
| E1: tắt concept | 61.6 | −8.1 [−11.9, −4.6] * |
| E2: dùng concept của object ở video khác | 60.1 | −9.7 [−15.1, −4.6] * |
| E2b: dùng concept của object khác trong cùng video (7 object) | 60.4 | −26.7 * |

Tắt concept, hay thay bằng concept của object khác, đều làm J&F giảm mạnh. Vậy concept không chỉ là tín hiệu chung chung.

### 2. Gọi concept dày hơn giúp, và kết quả lặp lại được trên tập kiểm chứng

| Run (subset40) | % frame gọi concept | J&F | Δ so với E0 |
|---|---|---|---|
| E3a: ngưỡng HSV 0.2 | 17.1 | 71.5 | +1.7 [−1.8, +5.4] |
| E7: thêm lời gọi mỗi 5 frame | 25.7 | 72.2 | +2.4 [+0.4, +4.7] * |
| **E10a**: 3 frame sau mỗi trigger, cộng thêm khi đã ≥15 frame không gọi | 24.6 | **73.9** | **+4.1 [+1.6, +6.8] *** |
| E11a: HSV hoặc tín hiệu bất định | 15.3 | 70.6 | +0.8 [−1.0, +2.5] |
| E4b40: gọi ở mọi frame | 100 | 74.5 | +4.8 [−0.0, +9.8] |

Trên 160 video (`experiments/full160/summary160.md`):

| Tập | E0 | E10a | Δ (CI object; CI video) | Object tăng / giảm |
|---|---|---|---|---|
| 160 video (182 object) | 69.9 | **72.6** | +2.7 [+1.6, +4.0] *; [+1.5, +4.0] * | 95 / 54 |
| rest120 (138 object) | 70.0 | **72.2** | +2.3 [+1.0, +3.7] *; [+1.0, +3.7] * | 72 / 40 |

E10a vẫn hơn E0 trên 120 video chưa từng dùng để thiết kế. Mức tăng ở đó (+2.3) nhỏ hơn trên subset40 (+4.1). Điều này phù hợp với việc E10a được chọn dựa trên subset40.

### 3. Bank: dư địa lớn, nhưng các cách sửa không dùng GT chưa khai thác được

| Run | J&F | So với đối chứng |
|---|---|---|
| E6: E0 + thêm bank mỗi ≥10 frame | 71.1 | so với E0: +1.3 [−0.8, +3.5] |
| E13a (oracle): E0 + thêm bank mỗi ≥10 frame bằng mask GT | 74.0 | so với E6: +2.9 [+0.3, +6.0] * |
| E12: E10a + bank kiểu E6 | 75.1 | so với E10a: +1.2 [−2.0, +4.7] |
| E14: E10a + bank theo độ khác biệt | 75.0 | so với E12: −0.1 [−2.0, +1.6] |
| E13b (oracle): E10a + thêm bank bằng mask GT | 81.1 | so với E10a: **+7.2 [+2.5, +12.5] *** |

Tất cả số trong bảng trên là trên subset40. Trên **rest120**, E12 **thua** E10a: −1.4 [−2.7, −0.2] *, 36 object tăng / 57 object giảm. Phần tăng nhỏ của E12 trên subset40 không lặp lại được.

Kết luận về bank:
- **Thêm nhiều frame hơn bằng mask dự đoán không giúp.** E6 và E12 đều không đạt.
- **Bank đa dạng hơn cũng không giúp.** E14 có bank đa dạng nhất nhưng J&F không hơn E12.
- **Thêm bằng mask GT thì giúp rõ.** E13b hơn E10a +7.2. Như vậy nút thắt là **chất lượng mask** của frame đưa vào bank, không phải số lượng hay độ đa dạng.

### 4. Các hướng không đạt

- **E11 (trigger theo độ bất định):** trong mô phỏng không cấu hình nào đạt tiêu chí. E11a chạy thật chỉ +0.8, CI chứa 0.
- **I1 (thay LLM bằng mạng nhỏ học từ teacher):** không đạt tiêu chí đặt trước. Relative MSE trên tập val là 0.97, bản object-centric là 0.74, trong khi tiêu chí là ≤ 0.5. Ngay cả oracle biết object là gì cũng chỉ đạt 0.55.
- **I5 (can thiệp trên video lon snack):** gọi concept ép buộc có thêm hay không thêm frame vào bank chỉ chênh 0.2 điểm, dưới tiêu chí 10. Thí nghiệm `--bank_on_burst` vì vậy không chạy.

## Chi phí

- **E0:** 9.9 fps trên subset40, 7.2 % frame gọi concept.
- **E10a:** đo trên rest120 khi GPU không bị chia, đạt 6.8 fps và 26.0 % frame gọi concept. E0 không có log tốc độ trên rest120, nên tôi không có tỉ lệ fps trực tiếp trên cùng tập. Trên subset40, các run gọi concept dày tương tự (E7, E10b) chạy khoảng 7.6 fps.
- **E12:** 6.1 fps trên rest120. Thêm frame vào bank làm ngữ cảnh LLM dài hơn.

## Khuyến nghị

- **Dùng E10a** (`--burst_after_trigger 3 --concept_every_sparse 15 --bank_rule hsv`). Đây là cấu hình duy nhất hơn E0 có CI không chứa 0 trên cả subset40 và tập kiểm chứng 120 video. Cái giá là fps giảm khoảng 30 %.
- **Không dùng E12 hay E14.** Không lặp lại được (E12) hoặc không hơn (E14).
- **Hướng tiếp theo nếu mở lại:** chọn mask tốt cho bank. Đây là phần dư địa E13b cho thấy (+7.2). Hai ví dụ: lọc frame theo chất lượng mask dự đoán, hoặc chỉnh mask trước khi đưa vào bank. Kết quả mọi thử nghiệm như vậy phải được kiểm tra trên một tập chưa dùng.

## Hạn chế

- subset40 chỉ có 44 object, nên CI rộng. Mọi cấu hình được chọn trên subset40. Chỉ E10a và E12 được kiểm chứng trên rest120.
- E0 trên 160 video là run cũ chạy bằng code gốc (`outputs/secvos_full`), không phải chạy lại. Trên subset40, run này cho đúng J&F của E0.
- Các run oracle dùng GT nên chỉ cho biết mức trần.
- E10a (subset40) và E7bd chạy lúc GPU bị chia. fps của hai run này trong `summary.md` không so được. J&F không bị ảnh hưởng.
- E15 (concept là vector hằng số, bằng trung bình các vector teacher, lịch gọi như E0) đã được đề xuất nhưng **chưa chạy**.
