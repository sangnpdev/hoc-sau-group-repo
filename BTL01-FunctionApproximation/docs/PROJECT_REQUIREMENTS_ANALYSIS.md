# PROJECT REQUIREMENTS ANALYSIS
## BTL01 – Function Approximation using MLP (Deep Learning)

> Tài liệu phân tích yêu cầu dự án. Soạn từ bốn file PDF và các tài liệu bổ sung.  
> **Nguồn ưu tiên cao nhất:** `MLP_Exercise - Yeu Cau.pdf` (4 trang).  
> Ngày phân tích: 2026-10-25.

---

## 1. Tổng quan dự án

| Mục | Chi tiết |
|-----|---------|
| Tên bài tập | Bài tập xấp xỉ hàm số dùng mạng MLP |
| Môn học | Học sâu (Deep Learning) |
| Loại bài toán | Hồi quy / Xấp xỉ hàm (Regression / Function Approximation) |
| Framework bắt buộc | **PyTorch** (nn.Module, nn.Linear) |
| Framework bị cấm | TensorFlow, Keras, AutoML |
| Được dùng hỗ trợ | NumPy, Pandas, SciPy, Matplotlib |
| Tổng điểm | 100 điểm |

---

## 2. Phát biểu bài toán và T/P/E

### 2.1. Bài toán ứng dụng
Xấp xỉ (học) một hàm toán học đã biết (hàm mật độ Gaussian 2 chiều) bằng mạng MLP. Thay vì giải tích tường minh, mạng học từ các điểm dữ liệu được lấy mẫu ngẫu nhiên.

### 2.2. Bài toán Machine Learning
**Hồi quy (Regression):** Cho đầu vào (x₁, x₂), dự đoán giá trị liên tục y = f(x₁, x₂).

### 2.3. T – P – E

| Ký hiệu | Nội dung |
|---------|---------|
| **T (Task)** | Dự đoán giá trị y = f(x₁, x₂) cho bất kỳ điểm (x₁, x₂) nào trong miền dữ liệu |
| **P (Performance)** | MSE (bắt buộc) + MAE hoặc RMSE (ít nhất một chỉ số bổ sung); đánh giá trên tập **test độc lập** |
| **E (Experience)** | A mẫu huấn luyện được sinh ngẫu nhiên từ miền [-2,2] × [-4,4] với nhãn y = f(x₁, x₂) |

> Nguồn: `MLP_Exercise - Yeu Cau.pdf`, trang 1–2; `quy_trinh_bao_cao.md`, bước 3.

---

## 3. Đặc tả dữ liệu Gaussian

### 3.1. Công thức chính xác

$$f(\mathbf{x}) = \frac{1}{2\pi\sqrt{\det(\mathbf{C})}} \exp\!\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \mathbf{C}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

> **Lưu ý quan trọng:** Công thức trong `MLP_Exercise - Yeu Cau.pdf` (trang 1) dùng $2\pi\sqrt{\det(C)}$, tương đương với phân phối n=2 chiều: $(2\pi)^{n/2}[\det(C)]^{1/2}$. Hai cách viết là như nhau khi n=2.

### 3.2. Tham số cố định (KHÔNG được thay đổi)

| Tham số | Giá trị |
|---------|---------|
| μ (mean) | [0, 0]ᵀ |
| C (covariance) | [[0.3, -0.5], [-0.5, 2]] |
| Miền x₁ | −2 ≤ x₁ ≤ 2 |
| Miền x₂ | −4 ≤ x₂ ≤ 4 |
| Phạm vi y | 0 ≤ y ≤ 1 (tự nhiên từ hàm Gaussian) |

> Nguồn: `MLP_Exercise - Yeu Cau.pdf`, trang 1.

### 3.3. Đặc tả tập dữ liệu

| Tập | Cách tạo | Kích thước |
|-----|---------|-----------|
| Train | Sinh ngẫu nhiên, ghi random seed | A mẫu (biến theo thực nghiệm) |
| Validation | Sinh độc lập (không dùng lại train) | Không quy định cụ thể số lượng; khuyến nghị ~100–200 |
| Test | Sinh độc lập, **chỉ dùng đánh giá cuối** | **100 mẫu** (cố định) |

> Nguồn: `MLP_Exercise - Yeu Cau.pdf`, trang 1–2.  
> **Quan trọng:** Test set KHÔNG được dùng để lựa chọn siêu tham số. Validation mới dùng để chọn mô hình tốt nhất (checkpoint).

### 3.4. Đối tượng quan sát và Features

| | Nội dung |
|-|---------|
| Đối tượng quan sát | Một điểm dữ liệu (x₁, x₂, y) lấy mẫu ngẫu nhiên trong miền xác định |
| Features X | (x₁, x₂) – 2 chiều |
| Target y | f(x₁, x₂) – giá trị Gaussian tại điểm đó |
| Input MLP | 2 node (x₁, x₂) + bias |
| Output MLP | 1 node (ŷ, giá trị xấp xỉ) |

---

## 4. Yêu cầu kỹ thuật PyTorch

> Nguồn: `MLP_Exercise - Yeu Cau.pdf`, trang 1–2 (mục 3.2).

### 4.1. Thành phần bắt buộc

| Thành phần | Yêu cầu |
|-----------|---------|
| Mô hình | `torch.nn.Module` + `nn.Linear` |
| Kiến trúc | Tùy chỉnh được số tầng ẩn và số neuron/tầng |
| Hàm kích hoạt | Phù hợp (sigmoid, ReLU, tanh – theo lựa chọn nhóm; công bố rõ) |
| Dữ liệu | `torch.Tensor`, `TensorDataset` hoặc `Dataset`/`DataLoader` |
| Loss function | `nn.MSELoss` (bắt buộc) |
| Optimizer | Adam hoặc SGD (ví dụ được nêu) |
| Training loop | Tự viết: `forward → loss → zero_grad → backward → optimizer.step` |
| Context | `model.train()` khi train, `model.eval()` + `torch.no_grad()` khi eval |
| Checkpoint | Lưu mô hình tốt nhất theo validation loss (`.pt` hoặc `.pth`) |
| Evaluation | MSE + ít nhất MAE hoặc RMSE trên tập **test** |
| Visualization | y thật vs ŷ dự đoán; bề mặt xấp xỉ; loss curve theo epoch |
| Logging | Loss theo epoch (CSV) |
| Reproducibility | Random seed cố định; công bố đầy đủ hyperparameters |
| Device | CPU (bắt buộc hỗ trợ); GPU là tùy chọn |

### 4.2. Data leakage – Yêu cầu bắt buộc

- Tạo train/validation/test **hoàn toàn độc lập** (sinh riêng, không dùng lại)
- Nếu chuẩn hóa X: học mean/std trên train, áp dụng cho val và test (không fit trên val/test)
- Test KHÔNG được nhìn vào trước khi hoàn thành lựa chọn cấu hình

---

## 5. Đặc tả ba nhóm thực nghiệm A/B/C

> Nguồn: `MLP_Exercise - Yeu Cau.pdf`, trang 2 (bảng, mục 4).

### Baseline cấu hình chuẩn (phải công bố trước khi thực nghiệm)

Cần công bố rõ: seed, batch size, epochs, optimizer, learning rate, activation, số tầng/neuron, điều kiện dừng. Các thực nghiệm A/B/C chỉ thay đổi biến độc lập, giữ nguyên các thông số còn lại.

---

### 5.1. Experiment A – Training Samples (Quy mô dữ liệu)

| Thuộc tính | Giá trị |
|-----------|---------|
| Biến độc lập | Số mẫu train A |
| Cấu hình đề xuất | 200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000 |
| Mạng MLP | Cố định (baseline) |
| Validation | Cố định (100 mẫu hoặc theo baseline) |
| Test | Cố định (100 mẫu) |
| Kết quả bắt buộc | Bảng MSE/MAE (hoặc RMSE) theo A; đồ thị sai số vs A |
| Mục tiêu phân tích | Ảnh hưởng của quy mô dữ liệu đến hiệu suất; xu hướng giảm lỗi khi tăng A |
| Khuyến khích | ≥ 3 seeds, báo cáo mean ± std |

**Điều cần giữ cố định:** kiến trúc MLP, epochs, batch size, optimizer, learning rate, seed của mạng.

---

### 5.2. Experiment B – Hidden Nodes (Độ rộng)

| Thuộc tính | Giá trị |
|-----------|---------|
| Biến độc lập | Số neuron tầng ẩn B |
| Kiến trúc | **MLP 1 tầng ẩn** (bắt buộc trong Exp B) |
| Cấu hình đề xuất | B = 1, 2, 3, ..., 10 (có thể mở rộng thêm) |
| Dữ liệu | Cố định (A từ baseline hoặc giá trị tốt nhất từ Exp A) |
| Kết quả bắt buộc | Bảng/đồ thị so sánh train/val/test error theo B |
| Mục tiêu phân tích | Nhận xét underfitting (B nhỏ) và overfitting (B lớn) |
| Khuyến khích | ≥ 3 seeds |

**Điều cần giữ cố định:** số tầng (= 1), A, epochs, batch size, optimizer, learning rate.

---

### 5.3. Experiment C – Hidden Layers (Độ sâu)

| Thuộc tính | Giá trị |
|-----------|---------|
| Biến độc lập | Số tầng ẩn C |
| Cấu hình đề xuất | C = 1, 2, 3, 4 (ít nhất) |
| Số neuron/tầng | Cố định (giữ nguyên hoặc theo baseline) |
| Dữ liệu | Cố định |
| Kết quả bắt buộc | Bảng/đồ thị test error, loss hội tụ, **thời gian huấn luyện** |
| Mục tiêu phân tích | So sánh kết quả, tốc độ hội tụ, training time theo C |
| Rủi ro cần phân tích | Vanishing gradient khi C lớn |
| Khuyến khích | ≥ 3 seeds |

**Điều cần giữ cố định:** số neuron/tầng, A, epochs, batch size, optimizer, learning rate.

---

## 6. Quy trình triển khai end-to-end

> Mapping từ `quy_trinh_bao_cao.md` (11 bước) vào `MLP_Exercise - Yeu Cau.pdf`.

| Bước | Nội dung | Mapping rubric |
|------|---------|----------------|
| 1 | Bài toán ứng dụng: xấp xỉ hàm Gaussian | Báo cáo mục 1 |
| 2 | Bài toán ML: hồi quy 2D input → 1D output | Báo cáo mục 1 |
| 3 | T/P/E | Báo cáo mục 1 |
| 4 | Xác định và mô tả dữ liệu: Gaussian, μ, C, miền, cách lấy mẫu | Rubric #1 (10đ) |
| 5 | Lý thuyết: MLP, backpropagation, MSE loss, Adam/SGD | Báo cáo mục 3 |
| 6 | Tiền xử lý: sinh dữ liệu, chia train/val/test, (tuỳ chọn: chuẩn hoá X) | Rubric #1 |
| 7 | Xây dựng mô hình PyTorch: nn.Module tùy biến | Rubric #2 (25đ) |
| 8 | Huấn luyện: training loop, checkpoint, logging | Rubric #2 |
| 9 | Thực nghiệm A/B/C | Rubric #3/#4/#5 (30đ) |
| 10 | Đánh giá: MSE, MAE/RMSE, plot y vs ŷ, surface | Rubric #6 (10đ) |
| 11 | Bình luận và kết luận | Rubric #7, #8 (25đ) |

---

## 7. Cấu trúc repository đề xuất (chưa tạo code)

```
BTL01-FunctionApproximation/
├── docs/
│   └── PROJECT_REQUIREMENTS_ANALYSIS.md    ← file này
├── requirement/
│   ├── MLP_Exercise - Yeu Cau.pdf
│   ├── MLP_Exercise.pdf
│   ├── Sample_Code.pdf
│   └── Sample_Report.pdf
├── phan-cong-cong-viec.xlsx
│
│   === Phần dưới CHƯA TẠO – cấu trúc đề xuất ===
├── src/
│   ├── data_generator.py       # sinh Gaussian, chia tập, lưu CSV
│   ├── model.py                # MLP (nn.Module, tùy chỉnh layers/nodes)
│   ├── trainer.py              # training loop, validation, checkpoint
│   ├── evaluator.py            # MSE, MAE, RMSE, plots
│   ├── experiment_A.py         # Exp A: vary num_samples
│   ├── experiment_B.py         # Exp B: vary hidden_nodes (1 layer)
│   └── experiment_C.py         # Exp C: vary num_layers
├── notebooks/
│   └── main.ipynb              # (tùy chọn) notebook tổng hợp
├── data/
│   ├── train.csv
│   ├── valid.csv
│   └── test.csv
├── outputs/
│   ├── logs/                   # CSV logs mỗi run
│   ├── figures/                # plots A/B/C
│   └── models/                 # best_model_*.pt
├── reports/
│   ├── BTL01_report.docx
│   ├── BTL01_report.pdf
│   └── BTL01_slides.pptx
├── requirements.txt
└── README.md
```

---

## 8. Danh sách deliverables (Sản phẩm nộp)

> Nguồn: `MLP_Exercise - Yeu Cau.pdf`, trang 3 (mục 8).

| # | Sản phẩm | Format |
|---|---------|--------|
| 1 | Báo cáo viết | `.docx` + `.pdf` |
| 2 | Slide thuyết trình | `.pptx` (+ `.pdf` khuyến khích) |
| 3 | Mã nguồn | `.py` và/hoặc `.ipynb`, có thể chạy lại |
| 4 | Requirements | `requirements.txt` (hoặc `environment.yml`) |
| 5 | README | `README.md` với hướng dẫn chạy từng thí nghiệm |
| 6 | Dữ liệu / script sinh dữ liệu | CSV hoặc script |
| 7 | CSV/log kết quả | CSV per experiment |
| 8 | Hình biểu đồ | PNG/PDF |
| 9 | Mô hình lưu | `.pt` / `.pth` (nếu dung lượng phù hợp) |
| 10 | File nén | `BTL01_Nhom_HoTen_MSSV.zip` |

**Nội dung ZIP:** Ghi thành viên, nhiệm vụ, tỷ lệ đóng góp tổng 100%.

**Lưu ý:** Không được nộp duy nhất file HTML xuất từ notebook. Code phải chạy lại được trên Python + PyTorch, không phụ thuộc đường dẫn cá nhân.

---

## 9. Checklist theo Rubric 100 điểm

> Nguồn: `MLP_Exercise - Yeu Cau.pdf`, trang 3–4 (mục 9).

### Tiêu chí 1: Dữ liệu & bài toán (10 điểm)

| Sub-criterion | Điểm | Checklist |
|--------------|------|----------|
| Đúng hàm Gaussian, μ=[0,0], C=[[0.3,-0.5],[-0.5,2]], miền dữ liệu đúng | 5 | ☐ |
| Chia tập và seed hợp lý, tránh rò rỉ dữ liệu | 3 | ☐ |
| Trực quan hóa/kiểm tra dữ liệu (scatter 3D, phân bố x₁, x₂, y) | 2 | ☐ |

### Tiêu chí 2: Triển khai PyTorch (25 điểm)

| Sub-criterion | Điểm | Checklist |
|--------------|------|----------|
| MLP tùy biến đúng bằng nn.Module (tùy chỉnh layers + nodes/layer) | 8 | ☐ |
| Vòng lặp train/backprop/optimizer chuẩn (forward→loss→zero_grad→backward→step) | 8 | ☐ |
| Validation/checkpoint (lưu best model theo val loss) và dự đoán/đánh giá test | 5 | ☐ |
| Cấu trúc code, README, chạy lại được (requirements.txt, không hardcode path) | 4 | ☐ |

### Tiêu chí 3: Thực nghiệm A (10 điểm)

| Sub-criterion | Điểm | Checklist |
|--------------|------|----------|
| Cấu hình so sánh công bằng (chỉ thay A, cố định phần còn lại) | 3 | ☐ |
| Đầy đủ kết quả bảng + đồ thị sai số theo A | 5 | ☐ |
| Phân tích ảnh hưởng số mẫu (nhận xét xu hướng) | 2 | ☐ |

### Tiêu chí 4: Thực nghiệm B (10 điểm)

| Sub-criterion | Điểm | Checklist |
|--------------|------|----------|
| Cấu hình so sánh công bằng (chỉ thay B, MLP 1 tầng ẩn) | 3 | ☐ |
| Đầy đủ bảng + đồ thị so sánh | 5 | ☐ |
| Phân tích độ rộng: nhận xét underfitting/overfitting | 2 | ☐ |

### Tiêu chí 5: Thực nghiệm C (10 điểm)

| Sub-criterion | Điểm | Checklist |
|--------------|------|----------|
| Cấu hình so sánh công bằng (chỉ thay C, giữ nguyên phần còn lại) | 3 | ☐ |
| Bảng/đồ thị test error, loss hội tụ, **thời gian train** theo C | 5 | ☐ |
| Phân tích độ sâu (vanishing gradient, tốc độ hội tụ) | 2 | ☐ |

### Tiêu chí 6: Đánh giá & phân tích chung (10 điểm)

| Sub-criterion | Điểm | Checklist |
|--------------|------|----------|
| Chỉ số MSE (bắt buộc) + MAE hoặc RMSE; biểu đồ y–ŷ hoặc bề mặt xấp xỉ | 5 | ☐ |
| Diễn giải, tổng hợp lựa chọn mô hình và giới hạn | 5 | ☐ |

### Tiêu chí 7: Báo cáo viết (15 điểm)

| Sub-criterion | Điểm | Checklist |
|--------------|------|----------|
| Đầy đủ bố cục 6 mục, rõ phương pháp và tham số | 5 | ☐ |
| Hình/bảng đánh số, chú thích; số liệu truy vết được (có log/code) | 5 | ☐ |
| Trình bày, tài liệu tham khảo | 5 | ☐ |

### Tiêu chí 8: Slide & thuyết trình (10 điểm)

| Sub-criterion | Điểm | Checklist |
|--------------|------|----------|
| Slide rõ, đầy đủ nội dung trọng tâm (~10–15 slide) | 4 | ☐ |
| Thuyết trình/phân tích kết quả (~10–15 phút) | 3 | ☐ |
| Trả lời câu hỏi và demo/giải thích mã nguồn | 3 | ☐ |

**Nguyên tắc chấm:** Mục nào thiếu hoặc không chạy/không kiểm chứng được = không điểm tương ứng. Sao chép kết quả mẫu mà không có log/mã nguồn kiểm chứng = không tính.

---

## 10. Khác biệt giữa đề chính thức và tài liệu mẫu

> Đây là các điểm **KHÔNG nên theo tài liệu mẫu**.

| # | Điểm khác biệt | Sample nói gì | Đề chính thức nói gì |
|---|--------------|--------------|---------------------|
| 1 | **Framework** | `Sample_Code.pdf`: dùng Keras + TensorFlow 1.x (session-based); `Sample_Report.pdf`: dùng TensorFlow | **Bắt buộc PyTorch** (nn.Module) |
| 2 | **Tham số covariance** | `Sample_Report.pdf` (trang 3): `cov = [[1,0],[0,1]]` (identity matrix) | **C = [[0.3, -0.5], [-0.5, 2]]** |
| 3 | **Kích thước train** | `Sample_Code.pdf`: n_train=20000; `Sample_Report.pdf`: 100→5000 | **A ∈ {200,...,2000}** (Exp A), không phải 20000 |
| 4 | **Cấu trúc thực nghiệm** | Sample_Code.pdf: Exp1=epochs E, Exp2=samples N, Exp3=hidden nodes | Đề chính thức: **A=samples, B=hidden_nodes, C=hidden_layers** |
| 5 | **Validation** | Sample_Code.pdf dùng test set như validation trong lúc train | Đề chính thức: **validation riêng**, test KHÔNG được dùng chọn siêu tham số |
| 6 | **Số thực nghiệm** | Sample không có Exp C (hidden layers) rõ ràng với PyTorch | Đề chính thức: **C = 1, 2, 3, 4 tầng ẩn (ít nhất)** |
| 7 | **Activation output** | Sample dùng sigmoid ở output | Đề chính thức không quy định; y ∈ [0,1] nên sigmoid output có thể hợp lý — cần kiểm tra |
| 8 | **Chuẩn hóa X** | Sample Code có viết hàm normalize_data nhưng commented out | Đề chính thức không bắt buộc; nhóm tự quyết định và ghi rõ |
| 9 | **Báo cáo dùng số liệu mẫu** | Sample Report có kết quả thực nghiệm cụ thể | **Cấm sao chép kết quả mẫu** (phải tự chạy) |

---

## 11. Rủi ro kỹ thuật và quyết định cần xác nhận

| # | Rủi ro / Câu hỏi | Mức độ | Ghi chú |
|---|----------------|--------|---------|
| R1 | **Activation output**: dùng sigmoid (ép y ∈ [0,1]) hay linear? | Trung bình | Nếu y không chuẩn hóa thì linear phù hợp hơn; nếu chuẩn hóa thì sigmoid. Cần quyết định nhất quán cho tất cả exp. |
| R2 | **Chuẩn hóa X**: có nên chuẩn hóa input trước khi đưa vào MLP? | Thấp | Giúp training ổn định hơn; nếu áp dụng phải học trên train, transform val/test. |
| R3 | **Số neuron trong Exp C**: giữ cố định bao nhiêu neuron/tầng khi thay đổi C? | Trung bình | Cần quyết định: giữ tổng params gần nhau hay giữ neuron/tầng cố định. |
| R4 | **Validation size**: không được quy định cụ thể số lượng | Thấp | Đề xuất: 100–200 mẫu, sinh độc lập. |
| R5 | **Early stopping**: có dùng không? | Thấp | Không bắt buộc; nếu dùng phải công bố patience. |
| R6 | **Seed management**: dùng seed cho cả numpy, random, torch | Cao | Bắt buộc để tái lập. `torch.manual_seed()`, `np.random.seed()`, `random.seed()`. |
| R7 | **Reproducibility trên GPU**: `torch.backends.cudnn.deterministic=True` | Thấp | Nếu dùng GPU thì cần thiết. |
| R8 | **Training time Exp C**: đo như thế nào (wall clock? per epoch?)** | Thấp | Dùng `time.time()` bao quanh training loop. |
| R9 | **Seeds ≥ 3**: khuyến khích nhưng không bắt buộc | Thấp | Tùy thời gian nhóm. |

---

## 12. Mapping yêu cầu → tài liệu nguồn → bằng chứng nghiệm thu

| Yêu cầu | Nguồn tài liệu / Trang | Bằng chứng nghiệm thu |
|---------|----------------------|----------------------|
| Đúng công thức Gaussian, μ, C | `Yeu Cau.pdf` trang 1 | Code sinh data + kiểm tra giá trị y |
| Miền dữ liệu [-2,2] × [-4,4] | `Yeu Cau.pdf` trang 1; `MLP_Exercise.pdf` slide 2 | assert min/max x1, x2 |
| Test = 100 mẫu độc lập | `Yeu Cau.pdf` trang 1 | CSV test.csv với 100 dòng |
| nn.Module + nn.Linear | `Yeu Cau.pdf` trang 1 (mục 3.2) | model.py, kiểm tra `isinstance(model, nn.Module)` |
| Training loop tự viết | `Yeu Cau.pdf` trang 1 (mục 3.2) | trainer.py không dùng `.fit()` |
| model.train() / model.eval() | `Yeu Cau.pdf` trang 1 | Review code |
| Checkpoint best val loss | `Yeu Cau.pdf` trang 1 | File `.pt` trong outputs/models/ |
| MSE + MAE hoặc RMSE | `Yeu Cau.pdf` trang 1 | evaluator.py, kết quả bảng |
| Exp A: A = 200,400,...,2000 | `Yeu Cau.pdf` trang 2 (bảng) | CSV log Exp A, 10 dòng |
| Exp B: B = 1,...,10, 1 tầng | `Yeu Cau.pdf` trang 2 (bảng) | CSV log Exp B, 10 dòng |
| Exp C: C = 1,2,3,4 (ít nhất) | `Yeu Cau.pdf` trang 2 (bảng) | CSV log Exp C, thời gian train |
| Không dùng test chọn hyperparams | `Yeu Cau.pdf` trang 2 (mục 4) | Code review: test chỉ gọi sau khi chọn model |
| requirements.txt + README.md | `Yeu Cau.pdf` trang 2 (mục 5) | File tồn tại trong ZIP |
| Báo cáo 6 mục, 10–20 trang | `Yeu Cau.pdf` trang 2–3 (mục 6) | Word/PDF với bố cục đầy đủ |
| Slide 10–15 slide, .pptx | `Yeu Cau.pdf` trang 3 (mục 7) | File .pptx trong ZIP |
| ZIP với tên đúng quy định | `Yeu Cau.pdf` trang 3 (mục 8) | `BTL01_Nhom_HoTen_MSSV.zip` |

---

## 13. Thành phần có thể tham khảo từ tài liệu mẫu (về mặt phương pháp)

> Các thành phần sau có thể học tư duy tổ chức, nhưng **phải tự triển khai bằng PyTorch**.

| Từ Sample_Code.pdf | Ý tưởng tham khảo | Phải viết lại bằng PyTorch |
|------------------|------------------|--------------------------|
| `generate_data()` với scipy.stats.multivariate_normal | Cách sinh dữ liệu Gaussian | ✅ Có thể dùng scipy để tính y, nhưng data pipeline là PyTorch |
| `visualize_dataset()` 3D scatter | Cách vẽ scatter 3D | ✅ Giữ matplotlib, không cần đổi |
| `normalize_data()` | Ý tưởng chuẩn hóa | ✅ Tự chọn có/không |
| `visualize_logs()` | Cách plot loss curve | ✅ Dùng matplotlib |
| Cấu trúc thư mục data/, model/ | Tổ chức file | ✅ Tham khảo |

| Từ Sample_Report.pdf | Ý tưởng tham khảo | Chú ý |
|--------------------|------------------|-------|
| Bố cục: Data, MLP, Experiments, References | Cấu trúc báo cáo | ✅ Theo mục 6 của đề chính thức |
| Hình scatter 3D Training/Testing | Loại hình minh họa | ✅ |
| Bảng CSV kết quả thực nghiệm | Định dạng bảng | ✅ Nhưng dùng số liệu tự chạy |

---

## Trạng thái sẵn sàng triển khai

**ĐÃ HIỂU:** Toàn bộ yêu cầu bài toán, dữ liệu, kỹ thuật PyTorch, 3 nhóm thực nghiệm A/B/C, rubric 100 điểm và các deliverables.

**CẦN CHÚ Ý:**
1. Sample_Report dùng covariance **sai** (identity matrix thay vì [[0.3,-0.5],[-0.5,2]])
2. Sample_Code dùng Keras/TensorFlow – toàn bộ mô hình + training phải viết lại bằng PyTorch
3. Validation set phải tách độc lập khỏi test; test KHÔNG được dùng trong quá trình lựa chọn mô hình

**THÔNG TIN CÒN THIẾU:**
- `phan-cong-cong-viec.xlsx`: chưa đọc được (binary Excel) – cần xem để biết phân công thành viên và deadline
- Deadline chính thức: không ghi trong `MLP_Exercise - Yeu Cau.pdf` (slide gốc `MLP_Exercise.pdf` cũng không nêu ngày cụ thể)
- Activation function cho hidden layers: đề không bắt buộc cụ thể (nhóm tự chọn và công bố)

**SẴN SÀNG TRIỂN KHAI:** Có thể bắt đầu lập trình khi nhận lệnh tiếp theo.
