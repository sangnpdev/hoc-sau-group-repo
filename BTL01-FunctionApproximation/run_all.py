"""Chạy toàn bộ quy trình: sinh dữ liệu → tinh chỉnh siêu tham số → A/B/C → chọn mô hình cuối → hình/bảng.

Ví dụ:
    python run_all.py                    # chạy tất cả các bước
    python run_all.py --stages A         # chỉ chạy thực nghiệm A (cần baseline.json đã có)
    python run_all.py --stages plots tables
Các lần chạy đã có metrics.json sẽ được bỏ qua, nên có thể chạy lại/tiếp tục an toàn.
"""
import argparse
import os

# Mạng rất nhỏ → CPU nhanh hơn GPU. Ẩn GPU để hàng chục tiến trình song song không tạo CUDA context
# (PyTorch có thể khởi tạo CUDA trong optimizer.step() và làm tràn bộ nhớ GPU dùng chung).
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

STAGES = ["data", "tune", "A", "B", "B200", "C", "final", "plots", "tables"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stages", nargs="+", default=STAGES, choices=STAGES)
    ap.add_argument("--workers", type=int, default=32, help="số tiến trình song song (mỗi tiến trình 1 luồng)")
    args = ap.parse_args()

    from src import data, experiments as ex
    for st in args.stages:
        print(f"===== stage: {st}", flush=True)
        if st == "data":
            if not os.path.exists(os.path.join(ex.DATA_DIR, "test.csv")):
                data.generate_datasets(ex.DATA_DIR)
            print("data in", ex.DATA_DIR)
        elif st == "tune":
            ex.run_group("tune", args.workers)
            ex.select_baseline()
        elif st == "C":
            # Thời gian huấn luyện là kết quả bắt buộc của nhóm C: mỗi cấu hình chạy trên 1 tiến trình riêng
            # (đủ số worker cho mọi run cùng lúc, không chạy chung với nhóm khác) để thời gian so sánh được.
            ex.run_group("C", len(ex.C_VALUES) * len(ex.SEEDS))
        elif st in ("A", "B", "B200", "final"):
            ex.run_group(st, args.workers)
            if st == "final":
                ex.select_final()
        elif st == "plots":
            from src import plots
            plots.make_all()
        elif st == "tables":
            from src import tables
            tables.make_all()


if __name__ == "__main__":
    main()
