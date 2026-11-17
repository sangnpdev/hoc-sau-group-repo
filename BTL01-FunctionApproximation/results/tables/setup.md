| Thành phần | Giá trị |
|---------------------|---------------------------------------------------------------------------------------------------------------------------|
| Optimizer | Adam (β₁ = 0.9, β₂ = 0.999) |
| Learning rate | 0.01 (chọn bằng tinh chỉnh; lưới: lr: {0.001, 0.003, 0.01, 0.03}; activation: {tanh, relu, sigmoid}; batch_size: {32, 128}) |
| Hàm kích hoạt tầng ẩn | sigmoid (chọn bằng tinh chỉnh) |
| Batch size | 32 (chọn bằng tinh chỉnh) |
| Đầu ra | 1 nơ-ron tuyến tính |
| Kiến trúc baseline | 1 tầng ẩn × 10 nơ-ron, A = 1000 mẫu train |
| Hàm mất mát | MSELoss (trên y chuẩn hoá y / f_max) |
| Số epoch tối đa | 3000 |
| Điều kiện dừng | early stopping: val loss không giảm ≥ 0.0001 (tương đối) sau 200 epoch |
| Lịch learning rate | ReduceLROnPlateau: ×0.5 sau 50 epoch không cải thiện, lr tối thiểu 1e-05 |
| Checkpoint | trọng số có val loss nhỏ nhất (best.pt) |
| Khởi tạo trọng số | mặc định của nn.Linear (Kaiming-uniform) |
| Seeds | dữ liệu: 2024; huấn luyện: 0, 1, 2 |
