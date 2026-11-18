# E18 – báo cáo cuối: nguồn lợi của E17a và tác động vào memory (2026-11-15)

Chi tiết từng bước:
- Bước 0–1: `analysis_report.md`.
- Bước 2: `report_e18a.md`, số liệu ở `e18a_numbers.txt`.

Thuật ngữ như trong hai báo cáo trên. Nhắc lại ngắn:
- *Pixel memory*: memory của SAM2 (nhánh a).
- *Bank*: keyframe bank cho MLLM (nhánh b).
- *Nhận nhầm*: bịa (GT rỗng) hoặc bắt nhầm vật (J < 0.5).
- Dấu * nghĩa là CI 95% không chứa 0.

## Những gì đã chạy và không chạy

| Biến thể | Trạng thái | Lý do |
|---|---|---|
| E18a `output`, `pixel` | đã chạy, 160 video | bước 2 |
| E18b (pixel + thêm vào bank sau 3 frame xác nhận) | **không chạy** | Luật dừng đặt ở điểm dừng 1: phần do bank trên rest120 là +0.31 (< 0.5), CI object [−0.93, +1.54] và CI video [−0.92, +1.58] đều chứa 0. Code đã cài và chạy thử trên 3 video smoke. |
| E18b+ (thêm cổng logit ≥ 0.717) | **bỏ** (bạn quyết ở điểm dừng 1) | Luật cho phép nhưng không bắt buộc. Ước lượng offline từ A1 cho thấy lỗ ròng: mất khoảng 27 × 7.4 ≈ 200 frame-J từ các lần nhận đúng bị chặn, trong khi chặn được tối đa khoảng 18 × 6.4 ≈ 115 frame-J từ nhận nhầm. AUC của logit cũng giảm từ 0.77 (subset40) xuống 0.65 (rest120). |
| E18c (làm sạch pixel memory khi mất dấu) | **bỏ** (bạn quyết ở điểm dừng 1) | SeC đã lọc sẵn pixel memory. Khi mất dấu, chỉ có đúng 1 frame xấu, luôn là f−1 (xem Q4). |
| E17c w = 0.8 | **không dùng** | Bạn dừng lượt này ở 19/160 video. Không đủ để báo. |

## Bảng chính

`pixel` không có trong danh sách đề bài cho bảng này. Tôi thêm nó vì nó là nền của E18b, và là biến thể gần E18b nhất có kết quả.

| Tập | Run | J&F | Δ vs E10a [CI obj; CI video] | Δ vs E17a [CI obj; CI video] | tăng/giảm vs E10a | nhận | nhận nhầm % | nhận nhầm vào bank | hồi phục sự cố | fps (trung vị) |
|---|---|---|---|---|---|---|---|---|---|---|
| rest120 | E10a | 72.2 | – | – | – | – | – | – | 57.0% | 7.99 |
| rest120 | E17a | **77.4** | +5.1 [+2.8, +7.6]*; [+2.8, +7.6]* | – | 61/28 | 265 | 22.3% | 59 | 64.3% | 6.71 |
| rest120 | pixel | 77.1 | +4.8 [+2.8, +7.0]*; [+2.8, +6.9]* | −0.3 [−1.5, +0.9]; [−1.6, +0.9] | 60/25 | 267 | 20.2% | 17 | 64.2% | 6.89 |
| rest120 | E18b | không chạy | | | | | | | | |
| rest120 | E18c | bỏ | | | | | | | | |
| all160 | E10a | 72.6 | – | – | – | – | – | – | 56.4% | 8.19 |
| all160 | E17a | **78.6** | +5.9 [+3.8, +8.3]*; [+3.7, +8.3]* | – | 83/36 | 339 | 23.3% | 79 | 65.3% | 7.09 |
| all160 | pixel | 77.8 | +5.1 [+3.3, +7.1]*; [+3.3, +7.1]* | −0.8 [−2.0, +0.3]; [−2.0, +0.3] | 79/33 | 345 | 21.2% | 21 | 65.4% | 6.77 |

## Kiểm tra tiêu chí đặt trước (rest120)

Tiêu chí (a)–(c) được đặt cho E18b. E18b không chạy, nên **không có biến thể nào được kiểm tra chính thức**.

Để tham khảo, tôi áp cùng tiêu chí lên `pixel`. Đây là **kiểm tra sau khi xem dữ liệu**, không phải kiểm định đặt trước.

| Tiêu chí | Ngưỡng | pixel | Đạt? |
|---|---|---|---|
| (a) J&F không kém E17a quá 0.5 | Δ ≥ −0.5 | −0.3 (CI [−1.5, +0.9]) | đạt theo ước lượng điểm. CI vẫn cho phép kém tới 1.5 |
| (b) tỷ lệ nhận nhầm, hoặc số frame nhận nhầm vào bank, thấp hơn E17a | < 22.3%, hoặc < 59 | 20.2%; 17 frame | đạt (rõ nhất ở số frame vào bank) |
| (c) số object giảm so với E10a ít hơn E17a | < 28 | 25 | đạt, nhưng chênh lệch nhỏ (3 object) |

## Trả lời Q1–Q4

**Q1. Lợi ích của E17a đến từ output, pixel memory hay bank?**
- Chủ yếu từ **pixel memory**: +4.35 trên rest120, chiếm 84% của G = +5.2, CI [+2.4, +6.4].
- Output: +0.47 (9%). Có ý nghĩa nhưng nhỏ.
- Bank: +0.31 (6%). CI chứa 0.
- all160 và subset40 cho cùng thứ tự. Phần do bank lớn hơn ở đó (13% và 27%), nhưng CI vẫn chứa 0.

**Q2. Nhận sai hại bao lâu? Có tín hiệu nào tốt hơn logit không?** (`analysis_report.md`, A1–A2)
- Hại chủ yếu từ bịa: 45 lần trên rest120, TB −6.4 frame-J mỗi lần.
- Trung vị thời gian hại là 3 frame. Khoảng 10% lần hại kéo dài 20–50 frame.
- Bắt nhầm vật gần như vô hại (−0.6 frame-J mỗi lần), vì E10a cũng đang sai ở đó.
- Lợi gấp khoảng 5 lần hại: +1515 frame-J từ nhận đúng, so với −290 từ bịa.
- **Không có tín hiệu nào tốt hơn hẳn logit.**
  - pred_iou và logit có AUC ~0.77 trên subset40, nhưng chỉ 0.65–0.70 trên rest120.
  - Cos max tới bank và diện tích mask gần như ngẫu nhiên.

**Q3. Tách "đoán" khỏi "ghi nhớ" có giữ được lợi ích mà giảm hại không?**
- E18b không chạy, nên chưa có câu trả lời trực tiếp.
- `pixel` là dạng tách đơn giản nhất. Mask được nhận vẫn vào pixel memory, nhưng không thêm thẳng vào bank. Kết quả của nó:
  - **Giữ được gần hết lợi ích**: −0.3 so với E17a, CI chứa 0.
  - **Bank sạch hơn rõ**: 59 → 17 frame nhận nhầm vào bank.
  - **Hại lên J giảm ít**: object giảm 28 → 25, tỷ lệ nhận nhầm 22.3% → 20.2%.
- Lý do hại không giảm nhiều: hại chủ yếu đi qua output và pixel memory, không qua bank (Q1). Một mask bịa vẫn được ghi vào pixel memory và kéo các frame sau.
- Muốn giảm hại thật sự thì phải chặn ở **cổng nhận**, không phải ở bank. Nhưng Q2 cho thấy chưa có tín hiệu đủ tốt để làm việc này.

**Q4. Khi mất dấu, pixel memory toàn frame mask rỗng có phải là thứ kéo nhánh a xuống không?**
- **Phần lớn là không.** Giả thuyết dựa trên một giả định sai về code.
- SeC đã lọc sẵn pixel memory (`inference/sam2_video_predictor.py:118-125`). Nó chỉ lấy các frame trước có pred_iou > 0.3 và logit > 0. Riêng f−1 luôn được thêm vào.
- Log E10a trên 13 389 frame mất dấu (all160): pixel memory TB có 18.1 frame, trong đó **đúng 1.00 frame xấu, luôn là f−1**.
- Vậy E18c thực chất chỉ bỏ f−1. Thay đổi quá nhỏ so với giả định, nên bạn đã bỏ E18c.
- Điều kéo nhánh a xuống không phải memory đầy frame rỗng. Có thể là các frame tốt trong memory đã cũ, hoặc đối tượng đã đổi dạng. Phần này chưa được kiểm tra.
- Kết quả Q1 khớp với điều này: thêm **một** frame tốt mới (mask concept) vào pixel memory đã đủ đem lại phần lớn lợi ích.

## Hạn chế

- Chỉ có 160 video (138 object trong rest120). CI của phần do bank rộng khoảng ±1.2 điểm, không đủ để thấy một hiệu ứng < 1 điểm.
- Ngưỡng cổng (pred_iou ≥ 0.78, logit > 0) và ngưỡng logit ≥ 0.717 ở A2 đều chọn trên subset40. AUC trên subset40 không giữ được trên rest120. Tỷ lệ nhận nhầm cũng vậy: 12.6% open-loop ở E17c so với 22.3% closed-loop ở E17a.
- Định nghĩa mất dấu (2 frame trước mask rỗng hoặc logit ≤ 0) được chọn bằng tay, không tối ưu.
- Cách tách G theo thứ tự output → pixel → bank chỉ là một thứ tự. Các phần có thể tương tác: tác dụng của bank có thể khác khi không có pixel memory. Thứ tự này đặt trước trong đề bài.
- Việc gán ΔJ cho từng lần nhận ở A1 chỉ gần đúng, vì sau lần nhận đầu, hai đường track đã khác nhau.
- fps đo ở các lần chạy khác nhau. E10a và E17a chạy ở phiên khác, nên tải máy có thể khác. Chỉ nên so thô.
- Áp tiêu chí (a)–(c) lên `pixel` là việc làm sau khi xem dữ liệu.

## Trạng thái

- GPU đang trống, không còn lượt nào trong hàng đợi.
- Không sửa kết quả cũ. E10a, E17a, E17c chỉ được đọc (eval của E17a được symlink vào `experiments/e18/eval`).
- Tôi dừng ở đây chờ bạn.
