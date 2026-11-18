# E17c – bước 0 và bước 1: quét w offline trên subset40 (2026-11-15)

**Thuật ngữ:**
- **a**: feature nhánh memory (pixel memory của SAM2).
- **b**: feature nhánh concept (token_attn).
- **f_w = (1 − w)·a + w·b**: tổ hợp lồi. w = 0.5 là gốc (E10a), w = 1.0 là E17a.
- **Lần gọi lúc mất dấu**: frame có gọi concept, trong khi 2 frame liền trước có mask rỗng hoặc object_score_logit ≤ 0.
- **Cổng**: mask không rỗng, pred_iou ≥ 0.78 và logit > 0.
- **Nhận nhầm**: mask qua cổng nhưng GT rỗng (bịa đối tượng) hoặc J < 0.5 (bắt nhầm vật khác).

## Bước 0 – Cài đặt và kiểm tra

- Flag mới: `--fusion_w` (mặc định 1.0, chỉ có tác dụng khi có `--concept_only_lost`) và `--fusion_sweep`.
  - w = 0.5 không bao giờ nhận mask, nên output bằng gốc.
  - w = 1.0 chạy đúng nhánh code của E17a.
  - Code: `inference/sam2_video_predictor.py` (đoạn E17), `inference/modeling_sec.py`, `vos_evaluation/vos_inference.py`.
- Kiểm tra tương đương trên 3 video (`experiments/e17/subset_smoke.txt`, 3 video tệ nhất của E0, có nhiều frame mất dấu), seed 0:

| Run | So với | Số file mask khác | J&F |
|---|---|---|---|
| chk05 (w = 0.5) | E10a | 0 | 25.7 = E10a |
| chk10 (w = 1.0) | E17a | 0 | 67.3 = E17a |
| sweep_smoke (quét w, output giữ nguyên) | E10a | 0 | 25.7 |
| sweep subset40 (bước 1) | E10a | 0 | 73.9 = E10a |

- Norm theo pixel (L2 theo kênh, lấy trung bình theo pixel). Tính trên 965 lần gọi lúc mất dấu (smoke + subset40):

| Feature | Norm TB | / norm a [min, max] |
|---|---|---|
| a (memory) | 18.06 | 1.000 |
| b (concept) | 17.95 | 0.994 [0.982, 1.011] |
| f_0.5 | 17.10 | 0.946 [0.882, 0.975] |
| f_0.6 | 17.12 | 0.948 [0.888, 0.976] |
| f_0.7 | 17.22 | 0.953 [0.905, 0.979] |
| f_0.8 | 17.40 | 0.963 [0.929, 0.985] |
| f_0.9 | 17.64 | 0.977 [0.959, 0.993] |
| f_1.0 | 17.95 | 0.994 [0.982, 1.011] |

  - cos(a, b) trung bình là 0.80.
  - Norm của f_w tăng đều theo w. Ở w = 0.5, norm thấp hơn a khoảng 5% (tối đa 12%).
  - Vì vậy f_0.5 hơi "co" so với feature decoder đã thấy lúc train, còn f_1.0 thì không.

## Bước 1 – Quét w (quỹ đạo cố định theo E10a)

- subset40 có 875 lần gọi lúc mất dấu có GT (40 object, 37 video): 527 lần GT rỗng, 348 lần GT không rỗng.
- **Lưu ý:** output và bank giữ y như E10a. Các số dưới đây chỉ đo **tại từng frame**. Chúng chưa tính hiệu ứng lan sang frame sau: mask được nhận sẽ vào memory và bank, và ảnh hưởng các frame tiếp theo.

### 1. Theo w, các lần gọi có GT không rỗng (n = 348)

| w | J TB | % mask khác rỗng | % qua cổng | số qua cổng | J TB của mask qua cổng |
|---|---|---|---|---|---|
| 0.5 | 0.148 | 17.0 | 12.9 | 45 | 0.880 |
| 0.6 | 0.187 | 21.6 | 16.4 | 57 | 0.874 |
| 0.7 | 0.230 | 27.0 | 21.8 | 76 | 0.882 |
| 0.8 | 0.339 | 40.2 | 29.0 | 101 | 0.906 |
| 0.9 | 0.407 | 48.9 | 38.2 | 133 | 0.911 |
| 1.0 | 0.445 | 53.4 | 42.0 | 146 | 0.912 |

Tăng w thì có nhiều mask khác rỗng hơn. Chất lượng mask qua cổng không giảm (J TB khoảng 0.88–0.91).

### 2–3. J hiệu dụng và nhận nhầm

**J hiệu dụng** = J của mask w nếu nó qua cổng, nếu không thì J của mask trộn (w = 0.5).

Tập chính (chốt trước khi có dữ liệu) là **mọi lần gọi lúc mất dấu** (n = 875). Ở frame GT rỗng, mask rỗng được J = 1, mask có vật được J = 0, giống bộ đánh giá. Như vậy việc bịa đối tượng cũng bị trừ điểm. Cột "GT không rỗng" chỉ để tham khảo.

CI là khoảng tin cậy 95%, bootstrap theo object (lấy mẫu lại object), 10 000 lần, seed 0.

| w | J hiệu dụng | Δ so với 0.5 [CI] | J hiệu dụng (GT không rỗng) | số nhận | bịa (GT rỗng) | bắt nhầm (J < 0.5) | % nhận nhầm |
|---|---|---|---|---|---|---|---|
| 0.5 | 0.652 | – | 0.148 | – | – | – | – |
| 0.6 | 0.658 | +0.006 [+0.002, +0.012] | 0.164 | 62 | 5 | 4 | 14.5 |
| 0.7 | 0.677 | +0.025 [+0.012, +0.040] | 0.213 | 81 | 5 | 5 | 12.3 |
| 0.8 | 0.706 | +0.053 [+0.026, +0.086] | 0.286 | 106 | 5 | 4 | **8.5** |
| 0.9 | 0.733 | +0.081 [+0.034, +0.133] | 0.368 | 144 | 11 | 6 | 11.8 |
| 1.0 | **0.743** | +0.091 [+0.041, +0.146] | **0.401** | 159 | 13 | 7 | 12.6 |

- Tỷ lệ nhận nhầm của mọi w đều dưới 20%.
- Thấp nhất ở w = 0.8 (8.5%). Từ 0.9 trở lên, số lần bịa đối tượng tăng gấp đôi (5 → 11 → 13).
- Mask bịa thường có logit thấp, trong khoảng (0, 1]. Ví dụ ở w = 1.0: lKsyz336zNw f49–51, -klPo9pilc4/002 f95–96.
- Có một trường hợp bắt nhầm với logit rất cao (3–4): CKmkgzeNKZw f41/58/118. Cổng không chặn được trường hợp này.
- Ví dụ đầy đủ: chạy `python3 experiments/e17c/sweep_analysis.py`. Dữ liệu từng lần gọi ở `experiments/e17c/sweep_calls.csv`.

### 4. Hình

![J hiệu dụng và tỷ lệ nhận nhầm theo w](sweep_plot.png)

`experiments/e17c/sweep_plot.png`: bên trái là J hiệu dụng, bên phải là tỷ lệ nhận nhầm. Đường chấm đỏ là w\*.

### 5. Chọn w\* theo luật đặt trước

1. Lọc theo tỷ lệ nhận nhầm < 20%: tất cả w đều đạt.
2. J hiệu dụng cao nhất: w = 1.0 (0.7433).
3. Xét w nhỏ hơn trong khoảng 0.01: w = 0.9 đạt 0.7328, kém 0.0105. Mức này **vượt ngưỡng 0.01 chỉ 0.0005**, nên không chọn 0.9.
4. So với w = 0.5: hơn +0.091 (≥ 0.02), nên **không dừng**.

**w\* = 1.0.**

### 6. So sánh w\* với w = 1.0 (E17a)

- **w\* trùng với E17a.** Không có trọng số ở giữa nào tốt hơn b một mình về J hiệu dụng. Giả thuyết "0.5 < w < 1 tốt hơn cả hai đầu" **không được xác nhận** (theo đo tại từng frame).
- Phần "ít nhận nhầm hơn" thì đúng một phần:
  - w = 0.8 có tỷ lệ nhận nhầm thấp nhất (8.5% so với 12.6%) và bịa ít hơn (5 so với 13).
  - Nhưng J hiệu dụng của w = 0.8 thấp hơn 0.037. CI của chênh lệch 1.0 − 0.8 là [+0.008, +0.069], không chứa 0.
- w = 1.0 và w = 0.9 sát nhau. CI của chênh lệch 1.0 − 0.9 là [−0.001, +0.026], chứa 0. w = 1.0 thắng nhờ có thêm 16 lần nhận (w = 0.9 có 1 lần nhận mà 1.0 không có).

## Hệ quả cho bước 2 – cần bạn quyết định

- E17c với w\* = 1.0 **chính là E17a**. chk10 cho thấy mask trùng từng bit.
- E17a đã chạy xong trên 160 video (J&F all160 78.6, so với E10a 72.6; bảng đầy đủ sẽ có ở bước 2).
- Vì vậy chạy lại "E17c(w\*)" trên 160 video sẽ cho kết quả y hệt E17a, và chỉ tốn GPU.
- Các lựa chọn:
  - **(A)** Không chạy lại. Viết `report.md` cho bước 2 từ kết quả E17a. Tiêu chí (a)–(d) được đối chiếu với E10a. Tiêu chí (d) "không cao hơn E17a" tự động đạt vì hai bên bằng nhau.
  - **(B)** Chạy thêm w = 0.9 trên 160 video, coi như kiểm tra độ nhạy. Lý do: 0.9 chỉ thua 0.0005 so với ngưỡng hòa, và bước 1 không tính hiệu ứng lan sang frame sau. Đánh giá chính vẫn trên rest120. Lần này sẽ ghi rõ là lệch khỏi luật đặt trước.
  - **(C)** Dừng E17c ở đây.
