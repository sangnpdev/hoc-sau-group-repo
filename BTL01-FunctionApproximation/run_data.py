"""
run_data.py – Mục 3.1: Sinh, lưu và khảo sát dữ liệu Gaussian hai chiều.

Cách chạy (từ thư mục BTL01-FunctionApproximation):
    python run_data.py

Các bước:
    1. Kiểm tra C xác định dương và hàm gaussian_2d khớp scipy + giá trị đỉnh lý thuyết.
    2. Sinh một lần N_TOTAL điểm với SEED, chia tuần tự train / valid / test (xem src/data.py).
    3. Kiểm tra kích thước, miền giá trị, không trùng điểm giữa các tập.
    4. Lưu CSV từng tập + all_data.csv + metadata.json (ghi rõ seed, cách chia).
    5. In thống kê mô tả và vẽ 3 hình khảo sát.

Kết quả:
    data/train.csv, data/valid.csv, data/test.csv   – 3 tập cố định (x1, x2, y)
    data/all_data.csv                               – gộp 3 tập, thêm cột split
    data/metadata.json                              – μ, C, miền, seed, kích thước, quy trình chia
    outputs/figures/data/fig1_scatter3d_surface.png – scatter 3D theo tập + bề mặt thật
    outputs/figures/data/fig2_overlay.png           – mẫu train chồng lên bề mặt thật
    outputs/figures/data/fig3_distributions.png     – phân bố x1, x2, y + contour và vị trí mẫu
    outputs/figures/data/data_summary.csv           – thống kê mô tả theo tập
"""
import json
import os
import sys

# Đảm bảo terminal Windows hiển thị tiếng Việt (UTF-8)
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal

from src.data import (
    COV, DATA_DIR, MU, N_TOTAL, SEED, SPLIT_SIZES, X1_RANGE, X2_RANGE,
    gaussian_2d, generate_splits, load_splits, save_csv,
)
from src.data_viz import describe_splits, plot_distributions, plot_overlay, plot_scatter3d_surface

# Đường dẫn tính theo vị trí file → chạy được trên máy bất kỳ.
ROOT    = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(ROOT, "outputs", "figures", "data")


def check_gaussian() -> None:
    """Kiểm chứng tham số và công thức.

    - C phải đối xứng, xác định dương (mọi trị riêng > 0) thì mới là ma trận hiệp phương sai hợp lệ.
    - gaussian_2d khớp scipy.stats.multivariate_normal trên 1000 điểm (sai số tương đối ≤ 1e-12).
    - Giá trị đỉnh f(μ) = 1 / (2π·sqrt(det C)).
    """
    eig = np.linalg.eigvalsh(COV)
    assert np.allclose(COV, COV.T) and np.all(eig > 0), "C không xác định dương"
    print(f"det(C) = {np.linalg.det(COV):.4f}, trị riêng C = {np.round(eig, 4)} → C xác định dương")

    lo = [X1_RANGE[0], X2_RANGE[0]]
    hi = [X1_RANGE[1], X2_RANGE[1]]
    pts = np.random.default_rng(0).uniform(lo, hi, (1000, 2))
    ref = multivariate_normal.pdf(pts, mean=MU, cov=COV)
    assert np.allclose(gaussian_2d(pts), ref, rtol=1e-12, atol=0), "gaussian_2d khác scipy"
    peak = 1.0 / (2 * np.pi * np.sqrt(np.linalg.det(COV)))
    assert np.isclose(gaussian_2d(MU)[0], peak), "giá trị đỉnh f(μ) sai"
    print(f"gaussian_2d OK (khớp scipy; đỉnh f(μ) = {peak:.6f})")


def check_splits(splits: dict) -> None:
    """Kiểm tra 3 tập: đúng kích thước, đúng miền, không có điểm nào nằm ở hai tập (chống rò rỉ)."""
    for name, (X, _) in splits.items():
        assert len(X) == SPLIT_SIZES[name], f"{name}: sai kích thước"
        assert X1_RANGE[0] <= X[:, 0].min() and X[:, 0].max() <= X1_RANGE[1], f"{name}: x1 ngoài miền"
        assert X2_RANGE[0] <= X[:, 1].min() and X[:, 1].max() <= X2_RANGE[1], f"{name}: x2 ngoài miền"
    keys = {name: {tuple(p) for p in X} for name, (X, _) in splits.items()}
    names = list(keys)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = keys[names[i]] & keys[names[j]]
            assert not overlap, f"{names[i]} và {names[j]} trùng {len(overlap)} điểm"
    print("Các tập OK (đúng kích thước, đúng miền, không trùng điểm giữa train/valid/test)\n")


def main() -> None:
    os.makedirs(DATA_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)

    # 1–3. Kiểm chứng, sinh dữ liệu, kiểm tra các tập
    check_gaussian()
    splits = generate_splits()
    check_splits(splits)

    # 4a. Lưu CSV từng tập và bảng gộp all_data.csv (có cột split)
    frames = []
    for name, (X, y) in splits.items():
        save_csv(X, y, os.path.join(DATA_DIR, f"{name}.csv"))
        df = pd.DataFrame({"split": name, "x1": X[:, 0], "x2": X[:, 1], "y": y})
        frames.append(df)
        print(f"{name:5s}: n={len(df):4d}  y ∈ [{df.y.min():.4f}, {df.y.max():.4f}]  mean={df.y.mean():.4f}")
    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(os.path.join(DATA_DIR, "all_data.csv"), index=False, float_format="%.8f")

    # Đọc lại CSV: dữ liệu các thực nghiệm sẽ dùng phải giống hệt dữ liệu vừa sinh
    loaded = load_splits()
    for name in splits:
        assert np.allclose(loaded[name][0], splits[name][0], atol=1e-7), f"{name}: X đọc lại bị lệch"
        assert np.allclose(loaded[name][1], splits[name][1], atol=1e-7), f"{name}: y đọc lại bị lệch"

    # 4b. Metadata: công bố seed và quy trình chia để người khác tái lập
    bounds, start = {}, 0
    for name, n in SPLIT_SIZES.items():
        bounds[name] = {"file": f"{name}.csv", "n": n, "rows": [start, start + n]}
        start += n
    metadata = {
        "function": "f(x) = 1/(2*pi*sqrt(det C)) * exp(-0.5 (x-mu)^T C^-1 (x-mu))",
        "mu": MU.tolist(),
        "cov": COV.tolist(),
        "domain": {"x1": list(X1_RANGE), "x2": list(X2_RANGE)},
        "seed": SEED,
        "n_total": N_TOTAL,
        "sampling": "np.random.default_rng(seed): sinh N_TOTAL giá trị x1 ~ U[-2,2] rồi N_TOTAL giá trị x2 ~ U[-4,4]",
        "splits": bounds,
        "protocol": (
            "Sinh một lần N_TOTAL điểm rồi chia tuần tự theo 'rows' (chỉ số [đầu, cuối)). "
            "train là kho 2000 mẫu; Exp A dùng A dòng đầu tiên. valid dùng để chọn "
            "checkpoint/siêu tham số; test chỉ dùng để đánh giá cuối cùng. Nhãn y không thêm nhiễu."
        ),
    }
    with open(os.path.join(DATA_DIR, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    # 5a. Thống kê mô tả
    print("\nThống kê mô tả (toàn bộ):")
    print(all_df[["x1", "x2", "y"]].describe().round(4).to_string())
    describe_splits(all_df).to_csv(os.path.join(FIG_DIR, "data_summary.csv"), index=False, float_format="%.6f")

    # 5b. Hình khảo sát
    plot_scatter3d_surface(all_df, os.path.join(FIG_DIR, "fig1_scatter3d_surface.png"))
    plot_overlay(all_df, os.path.join(FIG_DIR, "fig2_overlay.png"))
    plot_distributions(all_df, os.path.join(FIG_DIR, "fig3_distributions.png"))

    print(f"\nĐã lưu CSV vào {os.path.relpath(DATA_DIR, ROOT)}/, hình vào {os.path.relpath(FIG_DIR, ROOT)}/")


if __name__ == "__main__":
    main()
