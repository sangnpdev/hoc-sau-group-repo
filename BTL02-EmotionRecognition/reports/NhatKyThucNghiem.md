# Nhật ký thực nghiệm — BTL02: Nhận diện cảm xúc khuôn mặt (FER2013)

**Nhóm:** Nguyễn Phước Sang (3123411255, nhóm trưởng) · Trần Quốc An (3123411009) · Nguyễn Nguyên Chương (3123411042) · Nguyễn Gia Thiệu (3123411279)

Số liệu trong nhật ký lấy trực tiếp từ log lần chạy trên Colab: [`colab_run_2026-11-25/results.json`](colab_run_2026-11-25/results.json) và [`colab_run_2026-11-25/training_history.json`](colab_run_2026-11-25/training_history.json).

## 1. Môi trường và thiết lập chung

| Mục | Giá trị |
|---|---|
| Framework | PyTorch (script [`src/train_emotion.py`](../src/train_emotion.py)) |
| Phần cứng | Google Colab, GPU Tesla T4 (CUDA) |
| Dữ liệu | FER2013, ảnh xám 48×48, 7 lớp: angry, disgust, fear, happy, sad, surprise, neutral |
| Chia tập | Train 28 709 ảnh · Validation (PublicTest) 3 589 ảnh · Test (PrivateTest) 3 589 ảnh |
| Lệnh chạy | `python train_emotion.py --data-dir /content/fer2013 --output-dir /content/fer2013_results --epochs 12 --batch-size 128 --workers 2 --patience 4` |
| Optimizer | AdamW, ReduceLROnPlateau (factor 0.5, patience 2), early stopping (patience 4) |
| Chọn checkpoint | Checkpoint tốt nhất của từng thí nghiệm theo validation loss. Mô hình đưa đi đánh giá trên PrivateTest là mô hình có validation accuracy cao nhất. |

## 2. Diễn biến công việc

| Bước | Người thực hiện | Nội dung | Kết quả / ghi chú |
|---|---|---|---|
| 1 | Trần Quốc An | Chuẩn bị dữ liệu: sắp thư mục theo nhãn, tách PublicTest/PrivateTest, chuẩn hoá ảnh, thiết kế augmentation (lật ngang, xoay, tịnh tiến, scale) | Bộ dữ liệu dạng thư mục `data/train`, `data/test`, đóng gói `fer2013_data.zip` để tải lên Colab |
| 2 | Nguyễn Phước Sang | Viết `train_emotion.py`: kiến trúc CNN (2 conv 3×3/khối + BN + ReLU + MaxPool + Dropout, AdaptiveAvgPool, FC 256), vòng lặp huấn luyện, early stopping, lưu checkpoint | Script chạy được trên CPU/GPU, xuất JSON và hình |
| 3 | Nguyễn Nguyên Chương | Thiết kế 3 cấu hình thí nghiệm, chạy trên Colab (25/11/2026) | 3 checkpoint `*_best.pt`, log ở mục 3 |
| 4 | Nguyễn Nguyên Chương | So sánh đường cong học của 3 cấu hình | Chọn **small** (val acc cao nhất 60.04%) |
| 5 | Nguyễn Gia Thiệu | Đánh giá `small` trên PrivateTest: confusion matrix, recall theo lớp, ca dự đoán sai, trực quan hoá bộ lọc conv cuối | PrivateTest accuracy **61.33%** (mục 4) |
| 6 | Nguyễn Gia Thiệu | Làm slide thuyết trình | `presentation/BTL02_FER2013_BaoCao_Final.pptx` |
| 7 | Nguyễn Phước Sang | Tổng hợp kết quả, hoàn thiện báo cáo | `BaoCao_BTL02.pdf` |

## 3. Log huấn luyện theo từng thí nghiệm

Người chạy: Nguyễn Nguyên Chương. Mỗi thí nghiệm chạy tối đa 12 epoch. Cả 3 thí nghiệm đều chạy đủ 12 epoch, early stopping không kích hoạt.

### small

- Cấu hình: channels=[32, 64, 128], dropout=0.25, lr=0.001, weight_decay=0.0, augmentation=không
- Epoch tốt nhất (theo val loss): 12 — val loss 1.0389, val acc 60.04%

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.7593 | 27.59% | 1.6417 | 35.30% |
| 2 | 1.4987 | 41.46% | 1.3322 | 48.62% |
| 3 | 1.3521 | 47.45% | 1.2552 | 50.49% |
| 4 | 1.2815 | 50.48% | 1.2046 | 52.58% |
| 5 | 1.2265 | 52.35% | 1.2085 | 52.88% |
| 6 | 1.1859 | 54.45% | 1.1282 | 55.34% |
| 7 | 1.1458 | 56.11% | 1.1229 | 56.92% |
| 8 | 1.1178 | 57.58% | 1.1084 | 57.12% |
| 9 | 1.0891 | 58.83% | 1.0700 | 58.09% |
| 10 | 1.0664 | 59.46% | 1.0675 | 58.60% |
| 11 | 1.0393 | 60.56% | 1.0455 | 59.57% |
| 12 | 1.0203 | 61.29% | 1.0389 | 60.04% |

### medium_aug

- Cấu hình: channels=[32, 64, 128, 256], dropout=0.35, lr=0.001, weight_decay=0.0, augmentation=có
- Epoch tốt nhất (theo val loss): 12 — val loss 1.2215, val acc 52.47%

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.8092 | 24.29% | 1.7737 | 25.63% |
| 2 | 1.7841 | 26.07% | 1.7372 | 28.95% |
| 3 | 1.7412 | 28.80% | 1.5790 | 37.28% |
| 4 | 1.5963 | 37.76% | 1.4437 | 44.16% |
| 5 | 1.5183 | 41.23% | 1.3897 | 45.70% |
| 6 | 1.4690 | 43.18% | 1.3225 | 47.84% |
| 7 | 1.4288 | 44.62% | 1.3198 | 48.87% |
| 8 | 1.3998 | 46.16% | 1.2882 | 50.74% |
| 9 | 1.3672 | 47.04% | 1.2547 | 50.88% |
| 10 | 1.3444 | 48.53% | 1.2293 | 51.96% |
| 11 | 1.3254 | 49.07% | 1.2225 | 51.46% |
| 12 | 1.3091 | 49.55% | 1.2215 | 52.47% |

### medium_adamw

- Cấu hình: channels=[32, 64, 128, 256], dropout=0.4, lr=0.0005, weight_decay=0.0001, augmentation=có
- Epoch tốt nhất (theo val loss): 11 — val loss 1.2020, val acc 53.05%

| Epoch | Train loss | Train acc | Val loss | Val acc |
|---:|---:|---:|---:|---:|
| 1 | 1.8177 | 23.80% | 1.7870 | 25.86% |
| 2 | 1.7924 | 25.66% | 1.7677 | 27.25% |
| 3 | 1.7822 | 26.08% | 1.7366 | 28.56% |
| 4 | 1.7430 | 29.21% | 1.6463 | 36.11% |
| 5 | 1.6108 | 36.95% | 1.4197 | 45.25% |
| 6 | 1.4989 | 41.40% | 1.3784 | 45.81% |
| 7 | 1.4435 | 43.70% | 1.3010 | 48.87% |
| 8 | 1.4029 | 45.25% | 1.2741 | 50.46% |
| 9 | 1.3704 | 46.65% | 1.2564 | 50.24% |
| 10 | 1.3515 | 48.01% | 1.2229 | 52.08% |
| 11 | 1.3316 | 48.74% | 1.2020 | 53.05% |
| 12 | 1.3122 | 49.59% | 1.2068 | 52.80% |

## 4. Đánh giá mô hình được chọn (small) trên PrivateTest

Người thực hiện: Nguyễn Gia Thiệu.

- Accuracy tổng thể: **61.33%**

| Lớp | Số ảnh | Recall |
|---|---:|---:|
| Angry | 491 | 49.7% |
| Disgust | 55 | 14.5% |
| Fear | 528 | 23.3% |
| Happy | 879 | 86.2% |
| Sad | 594 | 64.3% |
| Surprise | 416 | 78.8% |
| Neutral | 626 | 57.2% |

## 5. Nhận xét và hướng tiếp theo

- **small** học đều đến epoch 12: train acc 61.29% và val acc 60.04%, chưa có dấu hiệu overfit rõ. Có thể tăng số epoch để cải thiện tiếp.
- **medium_aug** và **medium_adamw** sâu hơn và có augmentation, nhưng đến epoch 12 train acc vẫn dưới 50%: mô hình chưa hội tụ. Kết quả này chưa đủ để kết luận mô hình sâu hơn kém hơn; cần chạy lâu hơn hoặc chỉnh learning rate và regularization.
- Happy và surprise được nhận diện tốt nhất. Disgust và fear kém nhất: disgust hay bị nhầm thành angry/sad, fear hay bị nhầm thành sad. Nguyên nhân chính có thể là mất cân bằng lớp (disgust chỉ có 55 ảnh test).
- Hướng tiếp theo: tăng số epoch, dùng class weights hoặc oversampling cho disgust/fear, thử lr schedule khác cho các mô hình medium.
