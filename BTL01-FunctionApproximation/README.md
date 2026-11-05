# BTL01 – Xấp xỉ hàm Gaussian 2 chiều bằng MLP (PyTorch)

Huấn luyện mạng perceptron đa tầng (MLP) để xấp xỉ

$$f(\mathbf{x}) = \frac{1}{2\pi\sqrt{\det C}}\exp\!\left(-\tfrac12(\mathbf{x}-\boldsymbol\mu)^T C^{-1}(\mathbf{x}-\boldsymbol\mu)\right),\quad
\boldsymbol\mu = (0,0)^T,\; C = \begin{pmatrix}0.3 & -0.5\\ -0.5 & 2\end{pmatrix}$$

trên miền $-2 \le x_1 \le 2,\ -4 \le x_2 \le 4$, và khảo sát ảnh hưởng của số mẫu train (A), số nơ-ron tầng ẩn (B), số tầng ẩn (C).

## Cài đặt

```bash
python -m venv .venv && source .venv/bin/activate     # tuỳ chọn
pip install -r requirements.txt                       # torch, numpy, pandas, scipy, matplotlib
```

Đã kiểm tra trên Python 3.10, PyTorch 2.13 (CPU). Mã chạy hoàn toàn trên CPU; không cần GPU.

## Chạy lại toàn bộ

```bash
python run_all.py                    # data → tune → A → B → B200 → C → final → plots → tables
python run_all.py --workers 8        # máy ít nhân CPU: giảm số tiến trình song song
```

Từng bước riêng (mỗi lần train đã có `metrics.json` sẽ được bỏ qua, nên chạy lại/tiếp tục an toàn):

| Lệnh | Việc làm |
|---|---|
| `python run_all.py --stages data` | Sinh `data/train_pool.csv` (2000), `valid.csv` (200), `test.csv` (100), seed 2024 |
| `python run_all.py --stages tune` | Lưới lr × activation × batch size (3 seed), chọn theo **val MSE** → `results/baseline.json` |
| `python run_all.py --stages A` | Nhóm A: A = 200 … 2000 mẫu train, MLP 1 tầng × 10 nơ-ron |
| `python run_all.py --stages B` | Nhóm B: B = 1 … 10, 15, 20, 30, 50 nơ-ron, A = 1000 |
| `python run_all.py --stages B200` | Bổ sung cho B: cùng lưới B nhưng A = 200 (để quan sát quá khớp) |
| `python run_all.py --stages C` | Nhóm C: C = 1 … 5 tầng ẩn × 10 nơ-ron, A = 1000 (đo thời gian) |
| `python run_all.py --stages final` | Tinh chỉnh cuối (A = 2000; 10/20/40 nơ-ron × 2/3/4 tầng, tối đa 6000 epoch), chọn theo val → `models/final_model.pt` |
| `python run_all.py --stages plots tables` | Xuất hình `results/figures/` và bảng `results/tables/` |
| `python report/build.py` | Ghép báo cáo/slide từ template + bảng/hình → `.docx`, `.pptx`, `.pdf` (cần `pandoc`, `xelatex`, `libreoffice`) |

Muốn chạy lại từ đầu: xoá `data/`, `runs/`, `results/`, `models/` rồi `python run_all.py`.

## Cấu trúc thư mục

```
├── run_all.py            # điều phối toàn bộ quy trình (CLI --stages)
├── src/
│   ├── data.py           # hàm Gaussian 2D, sinh/lưu/nạp CSV, chuẩn hoá, DataLoader
│   ├── model.py          # class MLP(nn.Module) tuỳ biến số tầng/nơ-ron/activation
│   ├── train.py          # vòng lặp train/val, early stopping, ReduceLROnPlateau, checkpoint best.pt
│   ├── evaluate.py       # predict() và MSE / RMSE / MAE / R²
│   ├── experiments.py    # định nghĩa nhóm tune/A/B/B200/C/final, chạy song song, gộp mean ± std
│   ├── plots.py          # toàn bộ hình
│   └── tables.py         # bảng Markdown sinh từ CSV
├── data/                 # train_pool.csv, valid.csv, test.csv
├── runs/<nhóm>/<cấu hình>_s<seed>/   # config.json, log.csv (loss theo epoch), best.pt, metrics.json
├── results/
│   ├── <nhóm>_runs.csv   # mỗi lần chạy một dòng (train/val/test MSE, RMSE, MAE, R², thời gian…)
│   ├── <nhóm>_summary.csv# mean ± std qua 3 seed
│   ├── baseline.json     # siêu tham số chọn được ở bước tune
│   ├── final_selection.json
│   ├── figures/  tables/
├── models/final_model.pt # trọng số mô hình được chọn (state_dict + config)
├── report/               # template báo cáo/slide, script build, file .docx/.pdf/.pptx
└── requirements.txt
```

## Quy trình đánh giá công bằng

* Dữ liệu sinh **một lần** với seed 2024 (cùng một `numpy.random.default_rng`), ba tập độc lập: pool train 2000, validation 200, test 100. Nhóm A dùng **A mẫu đầu tiên** của pool, nên tập nhỏ là tập con của tập lớn; validation và test giữ nguyên cho mọi cấu hình.
* Chuẩn hoá theo biên miền đã biết (x₁/2, x₂/4; y / f_max với f_max = 1/(2π√0.35) ≈ 0.2690) – không dùng thống kê của dữ liệu nên không rò rỉ. Mọi chỉ số báo cáo ở **thang gốc** của y.
* Validation dùng cho: early stopping, checkpoint, chọn siêu tham số (tune), chọn kiến trúc cuối. **Test chỉ dùng để báo cáo**, không tham gia lựa chọn nào.
* Mỗi cấu hình chạy 3 seed (0, 1, 2: khởi tạo trọng số + thứ tự xáo trộn mini-batch), báo cáo mean ± std.

## Cấu hình huấn luyện chung

Adam, MSELoss, tối đa 3000 epoch, early stopping patience 200 epoch (cải thiện tương đối ≥ 1e-4), ReduceLROnPlateau (×0.5 sau 50 epoch không cải thiện, lr tối thiểu 1e-5). lr / activation / batch size lấy từ `results/baseline.json`.

## Kết quả chính (sinh từ `results/*.csv`)

* Baseline chọn theo validation: **sigmoid, lr = 0.01, batch = 32** (1 tầng × 10, A = 1000).
* **A:** test RMSE 1.12e-02 (A = 200) → 4.00e-03 (A = 800), sau đó bão hoà ≈ 4–5e-3 (mô hình 10 nơ-ron thiếu khớp).
* **B:** B = 1: R² = 0.027; B = 10: test RMSE 4.21e-03; B = 50: 1.60e-03. Không quá khớp với A = 1000; khoảng cách val/train tăng khi A = 200 (B200).
* **C:** test RMSE 4.21e-03 (C = 1) → 4.86e-04 (C = 2) → 2.08e-04 (C = 5); thời gian/epoch 34.8 → 66.0 ms.
* **Mô hình cuối** (`models/final_model.pt`): tầng ẩn 40-40-40-40 (5081 tham số), A = 2000; test MSE 1.21e-09, RMSE 3.47e-05, MAE 2.74e-05, R² = 0.9999997 (trung bình 3 seed: test RMSE 5.76e-05).

Báo cáo: `report/BaoCao_MLP.docx`, `report/BaoCao_MLP.pdf`; slide: `report/Slide_MLP.pptx`, `report/Slide_MLP.pdf`.
