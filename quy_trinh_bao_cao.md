# QUY TRÌNH THỰC HIỆN VÀ BÁO CÁO MÔN HỌC SÂU

> Quy trình tổng hợp từ ghi chú môn học và nội dung đã thống nhất. Bắt đầu từ xác định bài toán ứng dụng; không bao gồm bước chọn đề tài/dataset riêng biệt.

## 1. Xác định bài toán ứng dụng
- Trình bày vấn đề thực tế cần giải quyết, bối cảnh và mục tiêu ứng dụng.
- Nêu rõ kết quả đầu ra mong muốn và ý nghĩa thực tiễn.

## 2. Xác định bài toán máy học / học sâu
- Chuyển bài toán ứng dụng thành nhiệm vụ học máy cụ thể.
- Xác định loại bài toán: phân loại, hồi quy hoặc loại khác.
- Giải thích vì sao cách phát biểu này phù hợp.

## 3. Xác định T – P – E
- **T (Task):** Nhiệm vụ mô hình phải thực hiện.
- **P (Performance measure):** Thước đo hiệu năng phù hợp với nhiệm vụ.
- **E (Experience):** Dữ liệu/kinh nghiệm mô hình sử dụng để học.
- Liên hệ rõ T, P, E với bài toán đang nghiên cứu.

## 4. Xác định và mô tả dữ liệu
- Giới thiệu nguồn và ý nghĩa của dữ liệu được sử dụng.
- Xác định **đối tượng quan sát** (mỗi mẫu/dòng biểu diễn điều gì).
- Xác định **input/output**, **features (X)** và **target (y)**.
- Mô tả số mẫu, thuộc tính, kiểu dữ liệu, phân phối, dữ liệu thiếu và các vấn đề chất lượng dữ liệu.

## 5. Tổng quan lý thuyết liên quan
- Trình bày các khái niệm ML/DL cần thiết để hiểu bài toán.
- Giới thiệu thuật toán, kiến trúc mô hình và nguyên lý hoạt động có liên quan.
- Nếu có, tổng hợp các nghiên cứu/paper tham khảo và nêu cơ sở lựa chọn phương pháp.

## 6. Tiền xử lý dữ liệu
- Kiểm tra, làm sạch dữ liệu và xử lý giá trị thiếu/bất thường.
- Mã hóa thuộc tính phân loại, chuẩn hóa dữ liệu khi phù hợp.
- Chia train/validation/test; tránh rò rỉ dữ liệu bằng cách học các phép biến đổi trên tập train.
- Thực hiện feature engineering hoặc feature selection khi cần.

## 7. Lựa chọn thuật toán và xây dựng mô hình
- Xây dựng **baseline** làm mốc so sánh.
- Lựa chọn các mô hình Machine Learning và/hoặc Deep Learning phù hợp yêu cầu bài tập.
- Mô tả kiến trúc, đầu vào/đầu ra, lý do chọn mô hình và các giả định chính.

## 8. Thực thi và huấn luyện mô hình
- Trình bày môi trường, thư viện và cấu hình chạy thực nghiệm.
- Nêu loss function, optimizer, learning rate, batch size, epochs và các siêu tham số liên quan (nếu áp dụng).
- Huấn luyện, theo dõi kết quả validation và chọn mô hình theo quy tắc đã xác định.
- Lưu cấu hình và kết quả để có thể tái lập.

## 9. Đánh giá kết quả
- Đánh giá theo **P** đã xác định ở bước 3, trên tập dữ liệu phù hợp.
- Với **phân loại**: cân nhắc Accuracy, Precision, Recall, F1, Confusion Matrix.
- Với **hồi quy**: cân nhắc MAE, MSE, RMSE, R².
- Trình bày bảng/biểu đồ kết quả và so sánh với baseline.

## 10. Bình luận và thảo luận
- Giải thích mô hình nào tốt hơn và các nguyên nhân có thể dẫn đến chênh lệch.
- Phân tích tác động của tiền xử lý, đặc trưng và siêu tham số.
- Thảo luận overfitting/underfitting, hạn chế dữ liệu, sai số và khả năng ứng dụng thực tế.

## 11. Kết luận
- Tóm tắt bài toán, cách tiếp cận và kết quả chính.
- Nêu mức độ đáp ứng mục tiêu ban đầu, hạn chế và hướng cải tiến/phát triển.

---

## Sơ đồ quy trình rút gọn

**Bài toán ứng dụng → Bài toán ML/DL → T/P/E → Dữ liệu → Lý thuyết → Tiền xử lý → Xây dựng mô hình → Huấn luyện → Đánh giá → Bình luận → Kết luận**

## Gợi ý cấu trúc báo cáo

1. Giới thiệu và phát biểu bài toán (bước 1–3).
2. Dữ liệu và cơ sở lý thuyết (bước 4–5).
3. Phương pháp, tiền xử lý và xây dựng mô hình (bước 6–7).
4. Thiết lập và tiến hành thực nghiệm (bước 8).
5. Kết quả, đánh giá và thảo luận (bước 9–10).
6. Kết luận và hướng phát triển (bước 11).

*Lưu ý: Đây là khung quy trình tổng hợp, không phải bản chép nguyên văn quy định chính thức của giảng viên.*
