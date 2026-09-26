# Lab 03 — House Prices (Ames)

Dự án so sánh các mô hình scikit-learn và MLP PyTorch để dự đoán giá nhà `SalePrice`.

## Chạy E2E (khuyến nghị)

`run_experiments.ipynb` là notebook điều phối toàn bộ luồng: đọc dữ liệu, đánh giá mô hình bằng cross-validation, chọn mô hình tốt nhất, huấn luyện lại trên toàn bộ tập train và tạo submission. Không cần chạy lần lượt các notebook trong `src/`.

Mở PowerShell tại thư mục gốc repository:

```powershell
cd Tuan03\Lab03
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
$env:PYTHONUTF8 = "1"
jupyter execute run_experiments.ipynb
```

`PYTHONUTF8` giúp Python trên Windows đọc notebook UTF-8 ổn định. Có thể mở `run_experiments.ipynb` trong VS Code/JupyterLab và chọn kernel từ `Tuan03\Lab03\.venv` thay cho lệnh `jupyter execute`.

Để mở JupyterLab từ thư mục dự án:

```powershell
python -m jupyter lab
```

### Chế độ chạy

Notebook mặc định đặt `NOTEBOOK_ARGS = ["--quick"]` trong cell điều phối. Chế độ này giảm số cây và số epoch để kiểm tra nhanh luồng; điểm số chỉ mang tính tham khảo.

Để chạy cấu hình đầy đủ, đổi thành `NOTEBOOK_ARGS = []`, rồi chạy lại toàn bộ notebook. Cấu hình đầy đủ tốn thời gian hơn vì cross-validation huấn luyện nhiều mô hình; Random Forest/Extra Trees dùng 700 cây mỗi lần fit, Gradient Boosting dùng 1.200 cây, và MLP chạy nhiều epoch hơn.

Kết quả E2E được lưu trong `outputs/`: `cv_results.csv`, `feature_importance.csv`, model tốt nhất (`best_model.joblib` hoặc `best_model_mlp.joblib`), `submission.csv`, `run_metadata.json` và các biểu đồ PNG.

Đường dẫn `data/` và `outputs/` mặc định được tính từ thư mục `Tuan03\Lab03`. Có thể sửa `NOTEBOOK_ARGS` trong notebook để truyền các tùy chọn như `--data-dir`, `--output-dir`, `--folds` hoặc `--skip-mlp`.

## Kiểm tra dữ liệu và smoke test

Từ `Tuan03\Lab03`, sau khi kích hoạt môi trường và đặt `PYTHONUTF8` như trên:

```powershell
jupyter execute test.ipynb
jupyter execute test_project.ipynb
```

`test.ipynb` xem cấu trúc và giá trị thiếu của dữ liệu. `test_project.ipynb` chạy smoke test ngắn cho feature engineering, preprocessing, metric và Ridge/MLP; nó không tạo submission E2E.

## Chạy từng bước thủ công

Các notebook trong `src/` dùng để xem riêng từng giai đoạn. Mở chúng trong cùng kernel theo thứ tự:

1. `src/01_eda.ipynb`
2. `src/02_preprocess.ipynb`
3. `src/05_feature_engineering.ipynb`
4. `src/03_train_ml.ipynb`
5. `src/04_train_mlp.ipynb` (tùy chọn)
6. `src/06_submission.ipynb`

Luồng này là quy trình minh họa theo bước, dùng cấu hình nhanh và tập mô hình khác với `run_experiments.ipynb`. Nên dùng một trong hai luồng cho một lần chạy; nếu chạy xen kẽ, một số file trong `outputs/` có thể phản ánh các lần chạy/cấu hình khác nhau.

`house_price_pipeline.ipynb` và `pytorch_mlp.ipynb` chứa các hàm/lớp dùng chung. `notebook_loader.py` nạp code cell từ các notebook này cho những notebook khác. Có thể mở pipeline và chạy riêng để nạp định nghĩa, nhưng chúng không tự chạy toàn bộ E2E hay tạo submission.

## Dữ liệu, thí nghiệm và kết quả

- `data/train.csv`: 1.460 căn nhà, gồm `Id`, 79 biến đầu vào và nhãn `SalePrice`.
- `data/test.csv`: 1.459 căn nhà, gồm `Id` và 79 biến đầu vào, không có nhãn.
- `data/data_description.txt`: giải thích các biến trong bộ dữ liệu.
- `data/sample_submission.csv`: ví dụ định dạng submission.
- `requirements.txt`: các thư viện Python cần cài.

`run_experiments.ipynb` so sánh các cấu hình sau:

| Thí nghiệm | Đặc trưng | Mô hình |
|---|---|---|
| `paper_core_ridge` | 6 biến cốt lõi | Ridge |
| `raw_ridge` | Biến gốc | Ridge |
| `engineered_ridge` | Biến gốc và biến dẫn xuất | Ridge |
| `engineered_random_forest` | Engineered | Random Forest |
| `engineered_extra_trees` | Engineered | Extra Trees |
| `engineered_gradient_boosting` | Engineered | Gradient Boosting |
| `engineered_gbr_remove_gt4000` | Engineered; loại nhà có `GrLivArea > 4000` khỏi tập train | Gradient Boosting |
| `engineered_mlp` | Engineered | MLP PyTorch |

Các file kết quả E2E:

- `outputs/cv_results.csv`: điểm từng cấu hình; `log_rmse_mean` thấp thường tốt hơn, `log_rmse_std` nhỏ hơn nghĩa là ổn định hơn giữa các fold.
- `outputs/feature_importance.csv`: mức độ quan trọng của các cột sau biến đổi, tính từ Extra Trees; đây không phải quan hệ nhân quả.
- `outputs/best_model.joblib` hoặc `outputs/best_model_mlp.joblib`: pipeline tốt nhất được fit lại trên toàn bộ train.
- `outputs/submission.csv`: hai cột `Id`, `SalePrice`; một dự đoán cho mỗi dòng trong test.
- `outputs/run_metadata.json`: model được chọn, điểm CV và kích thước dữ liệu từ lần chạy `run_experiments.ipynb` gần nhất.

Luồng thủ công trong `src/` tạo thêm `outputs/eda_summary.json`, `feature_engineering.csv`, `ml_results.csv`, `mlp_results.csv` và `train_log.txt`. Notebook submission thủ công dùng các file `ml_results.csv`/`mlp_results.csv`; các file này không đồng bộ toàn bộ kết quả `run_experiments.ipynb` như `cv_results.csv` và `run_metadata.json`.

Biểu đồ được hiển thị ngay dưới cell trong notebook và lưu dưới dạng PNG trong `outputs/`. E2E tạo `cv_results.png`, `feature_importance.png` và `submission_price_distribution.png`; luồng `src/` tạo biểu đồ riêng cho EDA, preprocessing, feature engineering, điểm CV/MLP và phân phối submission.

Điểm `log_rmse_mean` càng thấp thường càng tốt; `log_rmse_std` mô tả độ dao động giữa các fold. `--quick` dùng để kiểm tra luồng, không thay cho kết quả cấu hình đầy đủ.
