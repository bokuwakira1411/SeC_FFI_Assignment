# fps160: E0, E10a, E14 trên 160 video, cùng code, chạy xen kẽ

Mỗi video chạy E0 → E10a → E14 liền nhau, trong một process (model load một lần). Một video khởi động (`_H00-2zCavQ`) chạy trước và không tính. Seed 0.

## 1. Kiểm tra tính đúng

- E0 trên all160: J&F 69.9, lần trước 69.9. Khớp.
- E10a trên all160: J&F 72.6, lần trước 72.6. Khớp.
- E14 trên subset40: J&F 75.0, lần trước 75.0. Khớp.
- Mask E0 trùng từng bit với lần chạy trước trên 40/40 video.
- Mask E10a trùng từng bit với lần chạy trước trên 160/160 video.
- Mask E14 trùng từng bit với lần chạy trước trên 40/40 video.
- E0 trên rest120 chỉ so được qua J&F: lần chạy E0 160 video trước dùng code gốc.

## 2. J&F


### all160 (160 video, 182 object)

| Run | J&F | J | F | J&F first / second / remain | Δ vs E0 [object CI] | Δ vs E0 video CI | up / down vs E0 | Δ vs E10a [object CI; video CI] | up / down vs E10a |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 69.9 | 69.7 | 70.2 | 84.0 / 69.8 / 67.5 | – | – | – | – | – |
| E10a | 72.6 | 72.4 | 72.9 | 85.7 / 72.9 / 70.2 | +2.7 [+1.6, +4.0] * | [+1.5, +4.0] * | 95 / 54 | – | – |
| E14 | 72.6 | 72.4 | 72.8 | 85.5 / 73.7 / 70.4 | +2.7 [+1.0, +4.4] * | [+1.0, +4.5] * | 88 / 71 | -0.0 [-1.3, +1.2]; [-1.3, +1.2] | 60 / 63 |

### rest120 (120 video, 138 object)

| Run | J&F | J | F | J&F first / second / remain | Δ vs E0 [object CI] | Δ vs E0 video CI | up / down vs E0 | Δ vs E10a [object CI; video CI] | up / down vs E10a |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 70.0 | 69.8 | 70.2 | 85.1 / 68.7 / 68.4 | – | – | – | – | – |
| E10a | 72.2 | 72.1 | 72.4 | 86.8 / 71.4 / 70.4 | +2.3 [+1.0, +3.7] * | [+1.0, +3.7] * | 72 / 40 | – | – |
| E14 | 71.8 | 71.7 | 72.0 | 86.2 / 71.4 / 70.9 | +1.9 [+0.1, +3.8] * | [+0.0, +3.8] * | 67 / 55 | -0.4 [-1.7, +0.9]; [-1.8, +0.9] | 45 / 46 |

### subset40 (40 video, 44 object)

| Run | J&F | J | F | J&F first / second / remain | Δ vs E0 [object CI] | Δ vs E0 video CI | up / down vs E0 | Δ vs E10a [object CI; video CI] | up / down vs E10a |
|---|---|---|---|---|---|---|---|---|---|
| E0 | 69.8 | 69.4 | 70.2 | 80.8 / 73.1 / 64.5 | – | – | – | – | – |
| E10a | 73.9 | 73.5 | 74.3 | 82.3 / 77.8 / 69.3 | +4.1 [+1.6, +6.8] * | [+1.5, +7.1] * | 23 / 14 | – | – |
| E14 | 75.0 | 74.6 | 75.4 | 83.2 / 81.1 / 69.1 | +5.2 [+1.7, +9.3] * | [+1.6, +9.5] * | 21 / 16 | +1.1 [-1.7, +4.4]; [-1.8, +4.3] | 15 / 17 |

## 3. Tốc độ

- **fps theo video** = số frame / thời gian xử lý video cho mọi object. Không tính load model. Có tính giải mã JPEG, frame đầu, đọc/ghi mask.
- **Thông lượng** = tổng (frame × object) / tổng thời gian.
- **ms/frame**: thời gian của một (object, frame), bỏ frame đầu.
- Bấm giờ bằng `perf_counter`, có `torch.cuda.synchronize()` ở hai đầu.


### cả 160 video

| Run | fps theo video (trung vị; trung bình) | thông lượng | ms/frame không gọi concept (trung vị) | ms/frame có gọi concept (trung vị) | % frame gọi concept | frame thêm vào bank | overhead ngoài frame |
|---|---|---|---|---|---|---|---|
| E0 | 10.50; 9.92 | 10.49 | 76 | 226 | 7.4 | 850 | 10.8 % |
| E10a | 8.19; 7.75 | 7.93 | 76 | 226 | 25.7 | 878 | 8.1 % |
| E14 | 7.25; 7.13 | 7.32 | 76 | 322 | 25.7 | 2115 (1178 thay thế) | 7.5 % |

### 157 video không bị chia GPU

| Run | fps theo video (trung vị; trung bình) | thông lượng | ms/frame không gọi concept (trung vị) | ms/frame có gọi concept (trung vị) | % frame gọi concept | frame thêm vào bank | overhead ngoài frame |
|---|---|---|---|---|---|---|---|
| E0 | 10.51; 9.91 | 10.49 | 76 | 226 | 7.4 | 837 | 10.8 % |
| E10a | 8.26; 7.75 | 7.93 | 76 | 226 | 25.6 | 865 | 8.1 % |
| E14 | 7.21; 7.13 | 7.32 | 76 | 322 | 25.6 | 2089 (1168 thay thế) | 7.5 % |

## 4. Tỷ lệ chậm đi (thời gian run / thời gian E0, theo từng video, trung vị, CI 95 % bootstrap theo video)

| So sánh | cả 160 video | video không bị chia GPU |
|---|---|---|
| E10a / E0 | ×1.27 [1.25, 1.30] | ×1.27 [1.25, 1.30] |
| E14 / E0 | ×1.41 [1.37, 1.44] | ×1.41 [1.37, 1.44] |
| E14 / E10a | ×1.08 [1.06, 1.10] | ×1.08 [1.06, 1.10] |

## 5. Video bị chia GPU

Bỏ qua `gnome-remote-desktop-daemon` (225 MiB, chạy thường trực), như đã thống nhất.

3 video:

- 8a6_6LWj3FY: 3895776:10MiB:rustdesk
- DGcdKA7159E: 3917512:224MiB:rustdesk
- aYO2eRIFjQs: 3937054:54MiB:rustdesk

## 6. Phần chậm thêm đến từ đâu

Tách theo tổng thời gian (không dùng trung vị), trên cả 160 video:

| Run | tổng thời gian (s) | số lần gọi concept | ms TB / frame có gọi | ms TB / frame không gọi | số frame ngữ cảnh TB / lần gọi | predict_forward TB (ms) |
|---|---|---|---|---|---|---|
| E0 | 3039 | 2347 | 237 | 73 | 3.4 | 160 |
| E10a | 4020 | 8138 | 241 | 74 | 3.5 | 163 |
| E14 | 4355 | 8138 | 282 | 74 | 4.9 | 203 |

- **E10a so với E0**: thêm 5791 lần gọi concept. Mỗi lần biến một frame thường thành frame có gọi, tốn thêm khoảng 168 ms. 5791 × 168 ms ≈ 971 s. Thời gian thực tăng 980 s.
- **E14 so với E10a**: cùng số lần gọi (8138). Mỗi lần gọi đắt hơn 41 ms, vì bank lớn hơn nên LLM nhận nhiều frame ngữ cảnh hơn. 8138 × 41 ms ≈ 331 s. Thời gian thực tăng 335 s.
- Phần ngoài frame (giải mã, đọc/ghi mask) gần như bằng nhau ở ba run.

Kiểm tra tính đúng: đạt.
