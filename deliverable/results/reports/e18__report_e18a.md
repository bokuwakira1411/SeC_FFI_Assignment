# E18a – bước 2: lợi ích của E17a đến từ đâu (2026-11-15)

**Thuật ngữ**
- *output*: mask của frame được trả ra để chấm điểm.
- *Pixel memory*: memory của SAM2 (nhánh a = memory_attn). Mask của mỗi frame được mã hoá và lưu vào đây.
- *Keyframe bank* (gọi tắt *bank*): các frame (ảnh + mask) đưa cho MLLM để tạo concept (nhánh b = token_attn).
- *Nhận*: dùng mask concept vì nó qua cổng (không rỗng, pred_iou ≥ 0.78, logit > 0).
- *Nhận nhầm*: *bịa* (GT rỗng) hoặc *bắt nhầm vật* (GT không rỗng, J < 0.5).
- Ba chế độ `--e17_commit` (mô tả ở `analysis_report.md`, mục 0.2):
  - `output`: mask được nhận chỉ thay output. Trạng thái tracking y như E10a.
  - `pixel`: mask được nhận là output và vào pixel memory. Không thêm thẳng vào bank. Bank vẫn theo luật thường của E10a.
  - `full`: = E17a (cả output, pixel memory, và thêm thẳng vào bank).
- *Sự cố*: J giảm > 0.3 so với TB 5 frame trước. *Hồi phục*: J trở lại trong 10 frame (định nghĩa như `experiments/e17/report.md`).
- CI 95%: bootstrap theo object và theo video (cụm). Dấu * nghĩa là CI không chứa 0.

**Đặt trước** (từ đề bài):
- Cách đọc G = J&F(full) − J&F(E10a) và cách tách 3 phần.
- Luật dừng E18b: phần do bank < 0.5 và CI chứa 0. Luật này bạn đặt ở điểm dừng 1, trước khi có kết quả bước 2.

**Chọn sau khi xem dữ liệu:** chỉ có đề xuất cấu hình nền ở cuối báo cáo.

**Cách chạy**
- Lệnh: `experiments/e18/run_queue.sh`. Cờ E17a giống `experiments/e17`, thêm `--e17_commit output|pixel`. Seed 0. Chạy tuần tự, một lượt GPU một lúc, có `nvidia-smi` trước mỗi run (`queue.log`).
- Kết quả: `experiments/e18/E18a_output`, `experiments/e18/E18a_pixel`.
- Số liệu: `python3 experiments/e18/report.py output pixel` → `e18a_numbers.txt`.
- E10a, E17a lấy từ kết quả có sẵn, không chạy lại.

## Bảng chính – rest120 (120 video, 138 object)

| Run | J&F | Δ vs E10a [CI obj; CI video] | tăng/giảm vs E10a | Δ vs E17a [CI obj; CI video] | số lần nhận | nhận nhầm: bịa / bắt nhầm / % | nhận nhầm vào bank | frame thêm vào bank | hồi phục sự cố | fps (trung vị) |
|---|---|---|---|---|---|---|---|---|---|---|
| E10a | 72.2 | – | – | – | – | – | – | 674 | 251/440 = 57.0% | 7.99 |
| E17a (full) | **77.4** | +5.1 [+2.8, +7.6]*; [+2.8, +7.6]* | 61/28 | – | 265 | 45 / 14 / 22.3% | 59 | 866 | 287/446 = 64.3% | 6.71 |
| output | 72.7 | +0.5 [+0.2, +0.7]*; [+0.2, +0.7]* | 48/22 | −4.7 [−6.9, −2.5]*; [−7.0, −2.5]* | 433 | 80 / 16 / 22.2% | – | 674 | 325/510 = 63.7% | 6.63 |
| pixel | **77.1** | +4.8 [+2.8, +7.0]*; [+2.8, +6.9]* | 60/25 | −0.3 [−1.5, +0.9]; [−1.6, +0.9] | 267 | 41 / 13 / 20.2% | 17 | 723 | 298/464 = 64.2% | 6.89 |

**Tách G** (rest120, G = +5.2):

| Phần | Δ (TB theo object) | CI object | CI video | % của G |
|---|---|---|---|---|
| output (output − E10a) | +0.47 | [+0.2, +0.7]* | [+0.2, +0.7]* | 9% |
| **pixel memory** (pixel − output) | **+4.35** | [+2.4, +6.4]* | [+2.5, +6.3]* | **84%** |
| bank (full − pixel) | +0.31 | [−0.9, +1.5] | [−0.9, +1.6] | 6% |

(Tổng 99% vì làm tròn. G tính trên J&F tổng; các phần là TB chênh lệch theo object.)

## Tham khảo – all160 và subset40

| Tập | Run | J&F | Δ vs E10a [CI obj] | Δ vs E17a [CI obj] | tăng/giảm vs E10a | nhận | nhận nhầm % | nhận nhầm vào bank | thêm vào bank | hồi phục |
|---|---|---|---|---|---|---|---|---|---|---|
| all160 | E10a | 72.6 | – | – | – | – | – | – | 878 | 56.4% |
| all160 | E17a | 78.6 | +5.9 [+3.8, +8.3]* | – | 83/36 | 339 | 23.3% | 79 | 1150 | 65.3% |
| all160 | output | 73.2 | +0.6 [+0.4, +0.9]* | −5.4 [−7.5, −3.4]* | 62/28 | 592 | 19.6% | – | 878 | 63.5% |
| all160 | pixel | 77.8 | +5.1 [+3.3, +7.1]* | −0.8 [−2.0, +0.3] | 79/33 | 345 | 21.2% | 21 | 952 | 65.4% |
| subset40 | E10a | 73.9 | – | – | – | – | – | – | 204 | 54.2% |
| subset40 | E17a | 82.4 | +8.5 [+3.4, +14.1]* | – | 22/8 | 74 | 27.0% | 20 | 284 | 69.0% |
| subset40 | output | 74.9 | +1.0 [+0.4, +1.6]* | −7.5 [−12.6, −2.9]* | 14/6 | 159 | 12.6% | – | 204 | 62.7% |
| subset40 | pixel | 80.0 | +6.2 [+2.3, +10.7]* | −2.3 [−5.0, +0.1] | 19/8 | 78 | 24.4% | 4 | 229 | 69.9% |

Tách G trên các tập tham khảo (CI theo video gần như trùng CI theo object):

| Tập | G | output | pixel memory | bank |
|---|---|---|---|---|
| all160 | +6.0 | +0.59 [+0.4, +0.9]* (10%) | +4.55 [+2.9, +6.4]* (76%) | +0.80 [−0.3, +2.0] (13%) |
| subset40 | +8.5 | +0.97 [+0.4, +1.6]* (11%) | +5.19 [+1.9, +9.1]* (61%) | +2.32 [−0.1, +5.0] (27%) |

## Đọc kết quả

1. **Pixel memory mang phần lớn lợi ích.** Trên rest120 nó chiếm 84% của G, và CI xa 0. Trên all160 là 76%, trên subset40 là 61%.
2. **Riêng output gần như không giúp.** +0.5, có ý nghĩa nhưng nhỏ.
   - Ở chế độ `output`, trạng thái vẫn mất dấu như E10a. Vì vậy nó gọi concept và nhận nhiều hơn (433 lần so với 265).
   - Mask đúng chỉ sống 1 frame, frame sau lại về mask trộn. Điều này tạo thêm sự cố (512 so với 442 ở E10a).
3. **Phần do bank nhỏ và chưa phân biệt được với 0.**
   - rest120: +0.31, CI [−0.9, +1.5].
   - Ước lượng điểm luôn dương (+0.3, +0.8, +2.3), nhưng cả ba CI đều chứa 0. Tập subset40 nhỏ (44 object) nên CI rộng.
   - Gợi ý ban đầu "E17a vượt oracle bank trên subset40" đúng hướng: lợi ích chủ yếu không đến từ bank.
4. **Bỏ thêm thẳng vào bank làm bank sạch hơn nhiều.**
   - Frame nhận nhầm đi vào bank: 59 (E17a) → 17 (pixel) trên rest120. 17 frame này vào bank qua luật bank thường của E10a, áp lên mask được nhận.
   - Tỷ lệ nhận nhầm gần như không đổi (22.3% → 20.2%). Lý do: cổng nhận giống nhau, chỉ đường ghi vào bank khác.
5. **fps**: output và pixel chậm hơn E10a khoảng 15%, giống E17a. Phần chậm là do chạy thêm decoder cho nhánh concept khi mất dấu. Các run chạy tuần tự trên cùng máy, nhưng tải máy có thể khác nhau.

## Trả lời Q1

**Lợi ích của E17a đến chủ yếu từ pixel memory.**
- Pixel memory: ~84% (rest120), CI không chứa 0.
- Output: ~9%.
- Bank: ~6%, CI chứa 0.

Nói cách khác, mask concept tốt cần được **ghi vào pixel memory** để các frame sau bám theo. Chỉ thay output thì gần như không còn lợi ích. Thêm thẳng vào bank chỉ góp thêm một phần nhỏ, chưa chắc chắn.

## Luật dừng E18b

- Phần do bank trên rest120 là +0.31 (< 0.5). CI theo object [−0.93, +1.54] và CI theo video [−0.92, +1.58] đều chứa 0.
- Theo luật bạn đặt ở điểm dừng 1, **E18b không chạy** (`queue.log`, 20:10:59). GPU đang trống.
- Code E18b (`--bank_confirm_k`) đã cài và chạy thử trên 3 video smoke (`chk_e18b`). Không có kết quả 160 video.

## Đề xuất cấu hình nền (chọn sau khi xem dữ liệu)

**Đề xuất: `pixel`** (`--e17_commit pixel`) làm nền cho các thí nghiệm memory tiếp theo.
- J&F không kém E17a có ý nghĩa: −0.3 trên rest120, CI chứa 0.
- Bank sạch hơn: nhận nhầm vào bank 59 → 17.
- Ít object giảm so với E10a hơn: 25 so với 28.
- Đơn giản hơn: bỏ một đường ghi mà không có bằng chứng là cần.

**Lưu ý ngược chiều:**
- Trên cả ba tập, ước lượng điểm của `full` cao hơn `pixel` (0.3 / 0.8 / 2.3).
- Nếu mục tiêu chỉ là J&F cao nhất, `full` (E17a) vẫn là cấu hình tốt nhất hiện có.
- Với 160 video, chưa đủ để kết luận bank có giúp hay không.
