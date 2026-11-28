# BTL02 — Nhận diện cảm xúc khuôn mặt (FER2013)

Học phần: Học sâu · Trường Đại học Sài Gòn, Khoa Công nghệ Thông tin

## Thành viên và phân công

| MSSV | Họ và tên | Nhiệm vụ | Sản phẩm chính |
|---|---|---|---|
| 3123411255 | Nguyễn Phước Sang (nhóm trưởng) | Viết script `train_emotion.py` (kiến trúc CNN, vòng lặp huấn luyện, early stopping, checkpoint); tổng hợp kết quả và hoàn thiện báo cáo | `src/train_emotion.py`, `reports/BaoCao_BTL02.pdf` |
| 3123411009 | Trần Quốc An | Xử lý dữ liệu FER2013: sắp xếp ảnh theo nhãn, tách PublicTest/PrivateTest, tiền xử lý và tăng cường dữ liệu | `data/`, hàm `make_transform`, `make_loaders` |
| 3123411042 | Nguyễn Nguyên Chương | Thiết kế và chạy 3 thí nghiệm (small, medium_aug, medium_adamw) trên Google Colab; so sánh và phân tích đường cong học; ghi nhật ký thực nghiệm | `notebooks/BTL02_FER2013_Colab.ipynb`, `reports/colab_run_2026-11-25/`, `reports/NhatKyThucNghiem.md` |
| 3123411279 | Nguyễn Gia Thiệu | Đánh giá mô hình trên PrivateTest (confusion matrix, precision/recall/F1, ca dự đoán sai, trực quan hoá bộ lọc); làm slide thuyết trình | `reports/presentation/BTL02_FER2013_BaoCao_Final.pptx` |

## Danh mục nộp

| Hạng mục | File |
|---|---|
| Báo cáo | [`reports/BaoCao_BTL02.pdf`](reports/BaoCao_BTL02.pdf) |
| Slide | [`reports/presentation/BTL02_FER2013_BaoCao_Final.pptx`](reports/presentation/BTL02_FER2013_BaoCao_Final.pptx) |
| Nhật ký thực nghiệm | [`reports/NhatKyThucNghiem.md`](reports/NhatKyThucNghiem.md) và log gốc: [`results.json`](reports/colab_run_2026-11-25/results.json), [`training_history.json`](reports/colab_run_2026-11-25/training_history.json), [`experiment_report.md`](reports/colab_run_2026-11-25/experiment_report.md) |
| Code | [`src/train_emotion.py`](src/train_emotion.py), [`notebooks/BTL02_FER2013_Colab.ipynb`](notebooks/BTL02_FER2013_Colab.ipynb) |
| Checkpoint | `reports/colab_run_2026-11-25/*_best.pt` |

## Chạy lại

```bash
python src/train_emotion.py --data-dir data --output-dir results --epochs 12 --batch-size 128 --workers 2 --patience 4
```

Trên Colab: mở notebook, tải lên `fer2013_data.zip` và `train_emotion.py`, rồi chạy tuần tự các cell.
