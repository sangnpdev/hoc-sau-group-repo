---
title: "Xấp xỉ hàm Gaussian 2 chiều bằng MLP (PyTorch)"
subtitle: "BTL01 – Học sâu – Lớp DCT123C3"
author:
  - "Nguyễn Phước Sang – MSSV: 3123411255"
  - "Nguyễn Gia Thiệu – MSSV: 3123411279"
  - "Trần Quốc An – MSSV: 3123411009"
  - "Nguyễn Nguyên Chương – MSSV: 3123411042"
date: "11/2026"
---

## Mục tiêu và bài toán

- Xấp xỉ hàm mật độ Gaussian 2 chiều bằng MLP: 2 đầu vào (x₁, x₂) → 1 đầu ra ŷ
- Miền x₁ ∈ [−2, 2], x₂ ∈ [−4, 4]; 0 ≤ y ≤ f_max ≈ 0.269
- Viết hoàn toàn bằng PyTorch (nn.Module, DataLoader, vòng lặp train tự viết)
- Khảo sát: **A** số mẫu train, **B** số nơ-ron (1 tầng), **C** số tầng ẩn
- Tự động hoá toàn bộ: `python run_all.py` (228 lần huấn luyện, 3 seed/cấu hình)

## Hàm Gaussian hai chiều

{{FIG:formula.png|det C = 0.35; hiệp phương sai âm → elip nghiêng, hàm không tách được theo x₁, x₂}}

## Dữ liệu

::: columns
:::: column
- Lấy mẫu đều, seed 2024, không nhiễu
- Train 2000 · Val 200 · Test 100
- Nhóm A: A mẫu đầu của pool
- Chuẩn hoá theo miền → không rò rỉ
- y lệch phải: ~11% mẫu có y > 0.1
::::
:::: column
{{FIG:data_scatter3d.png|Train / validation / test}}
::::
:::

## Mô hình MLP và chương trình PyTorch

::: columns
:::: column
- `MLP(nn.Module)`, số tầng/nơ-ron tuỳ chọn
- DataLoader, MSELoss, Adam
- forward → loss → zero_grad → backward → step
- eval() + no_grad() cho validation
- Early stopping, checkpoint theo val loss
::::
:::: column
{{FIG:mlp_deep.png|MLP C tầng ẩn}}
::::
:::

## Tự động tinh chỉnh siêu tham số

::: columns
:::: column
- Lưới: lr {0.001, 0.003, 0.01, 0.03} × {tanh, ReLU, sigmoid} × batch {32, 128}, 3 seed
- Chọn theo **val MSE** → **{{J:baseline:activation}}, lr {{J:baseline:lr}}, batch {{J:baseline:batch_size}}**
- ReLU kém nhất với 10 nơ-ron (ít mảnh tuyến tính)
- Baseline: 1 tầng × 10, A = 1000, ≤ 3000 epoch
::::
:::: column
{{FIG:tune_val_rmse.png|Val RMSE của 24 tổ hợp}}
::::
:::

## Nhóm A – số mẫu huấn luyện

::: columns
:::: column
- A = 200 → 800: test RMSE {{S:A:200:test_rmse}} → {{S:A:800:test_rmse}}
- A ≥ 800: bão hoà ≈ 4–5e-3, train ≈ test
- → giới hạn là **dung lượng mô hình** (10 nơ-ron), không phải dữ liệu
- A = 200: chỉ 26 mẫu gần đỉnh, phương sai giữa seed lớn
::::
:::: column
{{FIG:A_error.png|Sai số theo A}}
::::
:::

## Nhóm B – số nơ-ron tầng ẩn

::: columns
:::: column
- B = 1: R² = {{S:B:1:test_r2}} – thiếu khớp nặng
- B = 3: bắt đầu tạo được đỉnh; B ≥ 6: R² > 0.96
- B = 50: test RMSE {{S:B:50:test_rmse}}, lợi ích giảm dần
- A = 1000: không quá khớp; A = 200: khoảng cách val/train tăng tới ~1.4×
::::
:::: column
{{FIG:B_error.png|Sai số theo B (A = 1000 và A = 200)}}
::::
:::

## Nhóm B – bề mặt xấp xỉ

{{FIG:B_surfaces.png|B nhỏ chỉ tạo được dải, chưa tạo được đỉnh}}

## Nhóm C – số tầng ẩn

::: columns
:::: column
- C = 1 → 2: test RMSE {{S:C:1:test_rmse}} → {{S:C:2:test_rmse}} (~9×)
- 2 × 10 (151 tham số) tốt hơn 1 × 50 (201 tham số) ~3×
- C = 4–5: ≈ 2e-4, lợi ích giảm dần
- Không thấy tiêu biến gradient đến 5 tầng (Adam)
::::
:::: column
{{FIG:C_error.png|Sai số theo C}}
::::
:::

## Nhóm C – thời gian huấn luyện

{{FIG:C_time.png|Thời gian / epoch tăng tuyến tính; mạng sâu cần nhiều epoch hơn}}

## Tinh chỉnh cuối và mô hình được chọn

::: columns
:::: column
- A = 2000; {10, 20, 40} nơ-ron × {2, 3, 4} tầng; ≤ 6000 epoch; chọn theo val
- Chọn: tầng ẩn **{{J:final_selection:hidden_sizes}}**, {{J:final_selection:n_params}} tham số
- Test: MSE {{J:final_selection:test_mse}}, RMSE {{J:final_selection:test_rmse}}, MAE {{J:final_selection:test_mae}}, R² = {{J:final_selection:test_r2}}
::::
:::: column
{{FIG:final_grid.png|Val RMSE của lưới kiến trúc}}
::::
:::

## Minh hoạ mô hình xấp xỉ

{{FIG:final_y_vs_yhat.png|y thật – ŷ dự đoán trên train / validation / test}}

## Bề mặt xấp xỉ và sai số

{{FIG:final_surface.png|Bề mặt thật, bề mặt MLP và sai số ŷ − y}}

## Kết luận, hạn chế, hướng cải tiến

- Sai số chủ yếu do **thiếu khớp** → tăng dung lượng (nhất là **độ sâu**) hiệu quả hơn thêm dữ liệu khi A ≥ 800
- Cần B ≥ 3 để tạo đỉnh; quá khớp chỉ thấy rõ khi A = 200
- Hạn chế: dữ liệu không nhiễu, 3 seed, lr tinh chỉnh trên mạng 1 tầng
- Cải tiến: lr theo từng kiến trúc, lấy mẫu dày quanh đỉnh, thêm nhiễu
