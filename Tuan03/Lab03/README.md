# Lab 03 — House Prices (Ames)

Dự án so sánh hồi quy scikit-learn và MLP PyTorch để dự đoán giá nhà `SalePrice`.

## 1. Chuẩn bị và chạy

Mở PowerShell tại thư mục gốc repository, sau đó:

```powershell
cd 'Tuan03\Lab03'
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Nếu không dùng virtual environment `venv` sẵn có, tạo môi trường riêng bằng `python -m venv .venv` rồi kích hoạt môi trường đó trước khi cài dependencies.

Kiểm tra dữ liệu:

```powershell
jupyter execute test.ipynb
```

Chạy nhanh toàn bộ luồng (ít cây/epoch hơn, điểm chỉ để tham khảo):

```powershell
jupyter execute run_experiments.ipynb
```

Chạy cấu hình đầy đủ:

```powershell
jupyter execute run_experiments.ipynb
```

Đường dẫn mặc định `data/` và `outputs/` tính từ thư mục `Tuan03(nhom)`. Có thể đổi bằng `--data-dir` và `--output-dir`; dùng `--skip-mlp` để bỏ MLP. Cấu hình đầy đủ tốn thời gian hơn vì Random Forest/Extra Trees dùng 700 cây mỗi lần fit và Gradient Boosting dùng 1.200 cây.

Chạy bộ kiểm tra mã (không huấn luyện cuộc thi):

```powershell
jupyter execute test_project.ipynb
```

## 1.1. Quy trình chạy bằng notebook

Các notebook trong `src/` gọi lại pipeline thật và tự nối log vào `outputs/train_log.txt`; không có log hay kết quả được dựng sẵn.

Chạy lần lượt trong VS Code hoặc JupyterLab:

1. `src/01_eda.ipynb`
2. `src/02_preprocess.ipynb`
3. `src/05_feature_engineering.ipynb`
4. `src/03_train_ml.ipynb`
5. `src/04_train_mlp.ipynb` (có thể bỏ qua nếu không dùng MLP)
6. `src/06_submission.ipynb`

Mở JupyterLab từ thư mục `Tuan03/Lab03`:

```powershell
python -m pip install -r requirements.txt
jupyter lab
```

Notebook huấn luyện dùng cấu hình `quick=True` để chạy thử; các kết quả CV, feature engineering, model và `submission.csv` chỉ được sinh sau khi chạy các cell tương ứng.

## 2. File trong project

- `data/train.csv`: 1.460 căn nhà, `Id`, 79 biến đầu vào và nhãn `SalePrice` (81 cột tổng).
- `data/test.csv`: 1.459 căn nhà, `Id` và cùng 79 biến đầu vào (80 cột tổng), không có nhãn.
- `data/data_description.txt`: giải nghĩa biến và mã phân loại.
- `data/sample_submission.csv`: ví dụ định dạng nộp Kaggle.
- `test.ipynb`: in kích thước, mẫu, kiểu dữ liệu, giá trị thiếu và thống kê giá.
- `house_price_pipeline.ipynb`: nạp dữ liệu, tạo đặc trưng, tiền xử lý, định nghĩa model, cross-validation và lưu model.
- `pytorch_mlp.ipynb`: MLP PyTorch tương thích với pipeline scikit-learn; có validation nội bộ và early stopping.
- `run_experiments.ipynb`: điều phối thí nghiệm, xếp hạng, fit model tốt nhất và tạo submission.
- `test_project.ipynb`: kiểm tra feature engineering, xử lý thiếu/nhãn mới, metric và smoke fit.
- `requirements.txt`: thư viện Python cần cài.

Chi tiết luồng, dữ liệu, cách đọc metric và khung trình bày nằm ở [HUONG_DAN_VA_PHAN_TICH.md](HUONG_DAN_VA_PHAN_TICH.md).

## 3. Các thí nghiệm

| Thí nghiệm | Đặc trưng | Model |
|---|---|---|
| `paper_core_ridge` | 6 biến cốt lõi | Ridge |
| `raw_ridge` | Biến gốc | Ridge |
| `engineered_ridge` | Biến gốc + biến dẫn xuất | Ridge |
| `engineered_random_forest` | Engineered | Random Forest |
| `engineered_extra_trees` | Engineered | Extra Trees |
| `engineered_gradient_boosting` | Engineered | Gradient Boosting |
| `engineered_gbr_remove_gt4000` | Engineered, bỏ nhà `GrLivArea > 4000` khỏi train | Gradient Boosting |
| `engineered_mlp` | Engineered | MLP PyTorch |

## 4. Kết quả được tạo

- `outputs/cv_results.csv`: điểm từng cấu hình; `log_rmse_mean` thấp thường tốt hơn, `log_rmse_std` nhỏ hơn nghĩa là ổn định hơn giữa các fold.
- `outputs/feature_importance.csv`: importance của cột đã biến đổi, từ Extra Trees fit trên toàn train. Đây không phải quan hệ nhân quả.
- `outputs/best_model.joblib` hoặc `outputs/best_model_mlp.joblib`: pipeline tốt nhất fit lại trên toàn train.
- `outputs/submission.csv`: `Id`, `SalePrice`, 1.459 dự đoán ở thang giá gốc.
- `outputs/run_metadata.json`: model tốt nhất, điểm CV và kích thước dữ liệu.

`--quick` giảm số cây và epoch, nên điểm chạy nhanh chỉ để kiểm tra luồng, không dùng thay kết quả cấu hình đầy đủ.
