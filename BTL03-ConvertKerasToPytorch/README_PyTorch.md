# FER2013 CNN: bản PyTorch

Thư mục này chứa bản chuyển hai notebook FER2013 Keras sang PyTorch: CNN cơ bản và CNN có data augmentation + early stopping.

## Notebook theo từng bước

Hai notebook PyTorch nằm riêng trong `notebooks/pytorch/`; thư mục `examples/` giữ các notebook Keras gốc để tham khảo. Notebook PyTorch giữ cùng thứ tự trình bày: data, khai báo model, train, evaluate và dự đoán một ảnh. Kiến trúc được khai báo trực tiếp trong notebook; `src/fer2013_emotion_cnn.py` là bản dùng lại qua CLI.

- [CNN cơ bản](notebooks/pytorch/3_Emotion_Reg_CNN_PyTorch.ipynb)
- [CNN với utilities](notebooks/pytorch/3_Emotion_Reg_CNN_PyTorch_Utilities.ipynb)

## Dữ liệu cần có

Đặt hai file CSV cùng thư mục chạy lệnh:

- `fer2013_training_onehot.csv`
- `fer2013_publictest_onehot.csv`

Mỗi dòng gồm 2.304 giá trị pixel theo thứ tự hàng của ảnh 48×48, tiếp theo là 7 giá trị one-hot. Bản chuyển giữ pixel ở thang 0–255 như notebook Keras gốc và chuyển ảnh sang định dạng PyTorch NCHW.

## Cài thư viện

```bash
python -m pip install -r requirements-pytorch.txt
```

Nếu cần CUDA, hãy cài gói PyTorch phù hợp với môi trường theo hướng dẫn cài PyTorch đang dùng.

## Chạy

Chạy cả hai phiên bản:

```bash
python src/fer2013_emotion_cnn.py --variant both \
  --train-csv fer2013_training_onehot.csv \
  --validation-csv fer2013_publictest_onehot.csv \
  --output-dir fer2013_pytorch_results
```

Chỉ chạy CNN cơ bản hoặc bản augmentation:

```bash
python src/fer2013_emotion_cnn.py --variant basic
python src/fer2013_emotion_cnn.py --variant utilities
```

Mặc định, CNN cơ bản chạy 15 epoch; bản utilities chạy tối đa 100 epoch và dừng sớm sau 5 epoch liên tiếp không cải thiện validation loss. Có thể giới hạn thời gian thử nhanh bằng `--epochs`, hoặc chọn `--device cpu` / `--device cuda`.

## Kết quả đầu ra

Mỗi biến thể lưu checkpoint PyTorch, lịch sử train/validation dạng JSON, biểu đồ loss/accuracy, dự đoán minh họa trên một ảnh PublicTest và metrics dạng JSON.

## Lưu ý về đánh giá

Notebook Keras gốc dùng PublicTest làm `validation_data` rồi cũng gọi `evaluate()` trên chính tập đó. Vì vậy kết quả được đặt tên là PublicTest/validation, không phải kết quả trên một tập test độc lập. Bài nộp nên ghi rõ giới hạn này.
