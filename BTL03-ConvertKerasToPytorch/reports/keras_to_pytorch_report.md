# Báo cáo chuyển FER2013 CNN từ Keras sang PyTorch

## 1. Mục tiêu

Chuyển hai notebook FER2013 CNN có sẵn trong thư mục examples sang PyTorch, giữ nguyên nhiệm vụ phân loại 7 biểu cảm và các thiết lập thực nghiệm chính. Phạm vi gồm CNN cơ bản và phiên bản có tăng cường dữ liệu cùng early stopping.

## 2. Mã nguồn được chuyển

- `examples/2. CNN + MLP/3_Emotion_Reg_CNN.ipynb`: CNN cơ bản.
- `examples/4. Training Utilities/3_Emotion_Reg_CNN.ipynb`: cùng kiến trúc, bổ sung data augmentation và early stopping.

Bản PyTorch có hai notebook trình bày theo từng bước trong `notebooks/pytorch/`, cùng script tái sử dụng `src/fer2013_emotion_cnn.py`. Thư mục `examples/` chỉ giữ notebook Keras gốc để tham khảo và đối chiếu.

## 3. Kiến trúc tương đương

Đầu vào là ảnh grayscale 48×48. Lớp nhãn theo thứ tự chỉ số thực tế trong decoder của notebook: `0 angry`, `1 disgust`, `2 fear`, `3 happy`, `4 sad`, `5 surprise`, `6 neutral`.

| Khối | Keras | PyTorch | Kích thước không gian |
|---|---|---|---:|
| Đầu vào | `(48, 48, 1)` NHWC | `(1, 48, 48)` NCHW | 48×48 |
| Conv 1 | Conv2D 64, 3×3, same, ReLU | Conv2d 1→64, padding 1 + ReLU | 48×48 |
| Conv 2 | Conv2D 64, 3×3, valid, ReLU | Conv2d 64→64, padding 0 + ReLU | 46×46 |
| Pool/drop | MaxPool 2×2; Dropout 0,25 | MaxPool2d 2×2; Dropout 0,25 | 23×23 |
| Conv 3–6 | 128 same; 128 valid; 218 same; 218 valid | Conv2d cùng số kênh, kernel và padding | 19×19 |
| Pool/drop | MaxPool 2×2; Dropout 0,25 | MaxPool2d 2×2; Dropout 0,25 | 9×9 |
| Fully connected | Flatten → Dense 1024 ReLU → Dropout 0,5 | NHWC-order flatten → Linear 17.658→1.024 → Dropout 0,5 | vector 1.024 |
| Đầu ra | Dense 7 + softmax | Linear 1.024→7 logits | 7 lớp |

Tổng số phần tử sau convolution là `218 × 9 × 9 = 17.658`, khớp với đầu vào của lớp Dense 1024. Dữ liệu được chuyển sang NCHW theo chuẩn PyTorch. Trước khi flatten, tensor được sắp lại theo NHWC để giữ thứ tự đặc trưng của Keras.

## 4. Ánh xạ thành phần Keras → PyTorch

| Keras | PyTorch |
|---|---|
| `Conv2D` | `nn.Conv2d` |
| `MaxPooling2D` | `nn.MaxPool2d` |
| `Dropout` | `nn.Dropout` |
| `Flatten` | `permute(NCHW→NHWC)` rồi `view` |
| `Dense` | `nn.Linear` |
| `categorical_crossentropy` với one-hot | `nn.CrossEntropyLoss` với class index |
| `Adam(lr=0.0001)` | `torch.optim.Adam(lr=0.0001)` |
| `model.fit(...)` | vòng lặp forward, loss, backward, optimizer step |
| `model.evaluate(...)` | vòng lặp `eval()` + `torch.no_grad()` |
| `softmax` đầu ra | áp dụng `torch.softmax(logits, dim=1)` khi suy luận |
| `ImageDataGenerator` | phép biến đổi affine theo batch bằng `affine_grid`/`grid_sample` |
| `EarlyStopping(monitor='val_loss', patience=5)` | theo dõi validation loss, dừng sau 5 epoch không cải thiện |

Khi train, PyTorch nhận logits trực tiếp với cross-entropy; softmax chỉ dùng để lấy xác suất khi dự đoán. Cách này tương đương cross-entropy + softmax trong Keras nhưng ổn định số hơn.

## 5. Thiết lập được giữ lại

| Thiết lập | CNN cơ bản | CNN utilities |
|---|---:|---:|
| Loss | Categorical cross-entropy | Categorical cross-entropy |
| Optimizer | Adam, learning rate 0,0001 | Adam, learning rate 0,0001 |
| Batch size | 128 | 128 |
| Epoch | 15 | Tối đa 100 |
| Augmentation | Không | Xoay ±45°, dịch ngang/dọc tối đa 20%, lật ngang |
| Early stopping | Không | Theo dõi `val_loss`, patience 5 |

Ảnh giữ ở thang pixel 0–255, như notebook gốc. Augmentation dùng nội suy bilinear và padding biên gần nhất để gần với `ImageDataGenerator` của Keras. Do hai framework có thể lấy mẫu tham số biến đổi khác nhau, kết quả augmentation không bảo đảm giống từng pixel.

Keras `EarlyStopping` mặc định không khôi phục trọng số của epoch tốt nhất. Bản chuyển cũng dừng và giữ trọng số của epoch cuối cùng đã chạy; `best_epoch` chỉ được ghi lại để tham khảo.

Các Conv2D/Dense dùng khởi tạo Glorot uniform và bias 0 như mặc định Keras. Learning rate Adam được giữ ở 0,0001; beta/epsilon không được nêu trong source nên dùng mặc định của `torch.optim.Adam`.

## 6. Dữ liệu và đánh giá

Code đọc hai file `fer2013_training_onehot.csv` và `fer2013_publictest_onehot.csv`. Mỗi dòng phải có 2.304 pixel theo thứ tự hàng và 7 cột one-hot. One-hot được đổi thành class index trước khi tính loss.

Notebook nguồn dùng PublicTest làm validation trong quá trình train và sau đó gọi `evaluate()` trên cùng PublicTest. Bản chuyển giữ đúng luồng này để so sánh với ví dụ Keras, nhưng đặt tên metric là `final_publictest_*`. Đây không phải đánh giá trên test độc lập; không có file PrivateTest trong đầu vào được yêu cầu của ví dụ.

Kết quả train (loss, accuracy, thời gian, epoch cuối), checkpoint, lịch sử theo epoch, biểu đồ và một dự đoán minh họa được lưu trong thư mục output. Chưa ghi số đo thực nghiệm trong báo cáo này vì các CSV dữ liệu không nằm trong thư mục BTL03 ở workspace tại thời điểm chuyển đổi.

## 7. Các vấn đề được phát hiện và xử lý

1. Notebook gốc gọi `np.expand_dims(..., axis=4)` sau khi reshape thành mảng 3 chiều. Với mảng `(N, 48, 48)`, trục 4 không hợp lệ. Bản PyTorch reshape trực tiếp thành `(N, 1, 48, 48)`.
2. Phần mô tả tổng quan liệt kê `neutral` trước `sad`, nhưng hàm giải mã nhãn trong notebook ánh xạ chỉ số 4=`sad`, 5=`surprise`, 6=`neutral`. Bản chuyển theo thứ tự decoder.
3. Nhãn one-hot phải có đúng một lớp hoạt động. Loader kiểm tra số cột, giá trị hữu hạn và tổng từng hàng one-hot trước khi train.
4. PublicTest vừa tham gia chọn mô hình vừa được dùng để báo cáo cuối; báo cáo không gọi đây là test độc lập.

## 8. Cách chạy

Đặt hai CSV cạnh thư mục làm việc, cài thư viện trong `requirements-pytorch.txt`, sau đó chạy:

```bash
python src/fer2013_emotion_cnn.py --variant both --train-csv fer2013_training_onehot.csv --validation-csv fer2013_publictest_onehot.csv --output-dir fer2013_pytorch_results
```

Notebook theo từng bước: `notebooks/pytorch/3_Emotion_Reg_CNN_PyTorch.ipynb` và `notebooks/pytorch/3_Emotion_Reg_CNN_PyTorch_Utilities.ipynb`. Có thêm notebook Colab tiện chạy trong `notebooks/fer2013_emotion_cnn_pytorch.ipynb`; hướng dẫn và CLI nằm trong `README_PyTorch.md`.

## 9. Giới hạn

- Chưa có checkpoint Keras trong tài liệu, nên đây là chuyển mã kiến trúc và quy trình train, không phải chuyển trực tiếp trọng số đã huấn luyện.
- Không có CSV FER2013 trong workspace BTL03; vì vậy chưa thể lập bảng so sánh accuracy Keras/PyTorch chạy trên cùng dữ liệu.
- PublicTest không phải test độc lập trong luồng nguồn; muốn có đánh giá cuối không thiên lệch cần một test split chưa dùng khi chọn mô hình.
