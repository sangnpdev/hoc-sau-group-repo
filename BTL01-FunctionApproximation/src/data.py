"""
src/data.py – Sinh, lưu và nạp dữ liệu cho bài toán xấp xỉ hàm Gaussian 2 chiều (mục 3.1).

Cách chia dữ liệu (cố định, tái lập được):
  - Seed toàn cục: SEED = 42 (np.random.default_rng(SEED)).
  - Sinh MỘT lần N_TOTAL = 2300 điểm (x1, x2) độc lập, phân bố đều:
        x1 ~ U[-2, 2],  x2 ~ U[-4, 4]
  - Chia theo thứ tự sau khi sinh (các điểm đã độc lập nên không cần xáo trộn):
        train : 2000 mẫu (kho cho Exp A: A = 200..2000 lấy A dòng đầu)
        valid :  200 mẫu (chọn checkpoint / siêu tham số / early stopping)
        test  :  100 mẫu (CHỈ dùng đánh giá cuối cùng)
  - Nhãn mục tiêu: y = f(x1, x2) (không thêm nhiễu).

Nội dung module:
    - gaussian_2d()    : tính f(x) tự viết đúng công thức của đề.
    - generate_data()  : sinh n điểm đều trong miền + nhãn y.
    - generate_splits(): sinh N_TOTAL điểm rồi chia train / valid / test.
    - save_csv() / load_csv() / load_splits(): lưu – đọc CSV.
    - make_loader()    : numpy → TensorDataset → DataLoader của PyTorch.
"""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import TensorDataset, DataLoader

# ── Tham số cố định của đề bài (KHÔNG thay đổi) ─────────────────────────────
MU = np.array([0.0, 0.0])                    # vector kỳ vọng μ
COV = np.array([[0.3, -0.5], [-0.5, 2.0]])   # ma trận hiệp phương sai C
X1_RANGE = (-2.0, 2.0)                       # miền x1: -2 ≤ x1 ≤ 2
X2_RANGE = (-4.0, 4.0)                       # miền x2: -4 ≤ x2 ≤ 4

# ── Quy trình chia tập cố định (dùng chung cho baseline và Exp A/B/C) ───────
SEED        = 42
SPLIT_SIZES = {"train": 2000, "valid": 200, "test": 100}   # thứ tự = thứ tự cắt
N_TOTAL     = sum(SPLIT_SIZES.values())
# Thư mục data/ tính theo vị trí file này → không phụ thuộc đường dẫn cá nhân.
DATA_DIR    = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data")
# ─────────────────────────────────────────────────────────────────────────────


def gaussian_2d(X: np.ndarray, mu: np.ndarray = MU, cov: np.ndarray = COV) -> np.ndarray:
    """Hàm mật độ Gaussian 2 chiều theo đúng công thức của đề:

        f(x) = 1 / (2π·sqrt(det C)) · exp( -½ (x-μ)ᵀ C⁻¹ (x-μ) )

    Tham số:
        X  : mảng (n, 2) hoặc một điểm (2,).
        mu : vector kỳ vọng μ.
        cov: ma trận hiệp phương sai C.
    Trả về:
        Mảng (n,) chứa giá trị f tại từng điểm.
    """
    X = np.atleast_2d(np.asarray(X, dtype=float))
    diff = X - mu                                           # (x - μ)
    cov_inv = np.linalg.inv(cov)                            # C⁻¹
    mahal = np.einsum("ni,ij,nj->n", diff, cov_inv, diff)   # (x-μ)ᵀ C⁻¹ (x-μ) cho từng dòng
    norm_const = 2.0 * np.pi * np.sqrt(np.linalg.det(cov))  # 2π·sqrt(det C)
    return np.exp(-0.5 * mahal) / norm_const


def generate_data(n: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sinh n điểm phân bố đều trong miền và tính nhãn y = f(x1, x2).

    Sinh toàn bộ x1 trước rồi mới tới x2 bằng numpy default_rng(seed),
    nên cùng seed luôn cho cùng dữ liệu.

    Trả về:
        X: float32 (n, 2) – đặc trưng (x1, x2).
        y: float32 (n,)   – nhãn mục tiêu.
    """
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(*X1_RANGE, n)
    x2 = rng.uniform(*X2_RANGE, n)
    pts = np.column_stack([x1, x2])
    y = gaussian_2d(pts)                # tính y ở float64 rồi mới ép kiểu → chính xác hơn
    return pts.astype(np.float32), y.astype(np.float32)


def generate_splits() -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Sinh một lần N_TOTAL điểm với SEED rồi cắt tuần tự thành train / valid / test.

    Vì mỗi chỉ số chỉ thuộc đúng một đoạn nên 3 tập không bao giờ trùng điểm.
    Trả về dict {"train": (X, y), "valid": (X, y), "test": (X, y)}.
    """
    X, y = generate_data(N_TOTAL, seed=SEED)
    splits, start = {}, 0
    for name, n in SPLIT_SIZES.items():
        splits[name] = (X[start:start + n], y[start:start + n])
        start += n
    return splits


def save_csv(X: np.ndarray, y: np.ndarray, path: str) -> None:
    """Lưu một tập ra CSV với 3 cột x1, x2, y (8 chữ số thập phân)."""
    df = pd.DataFrame({"x1": X[:, 0], "x2": X[:, 1], "y": y})
    df.to_csv(path, index=False, float_format="%.8f")


def load_csv(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Đọc CSV (x1, x2, y) → (X float32 (n, 2), y float32 (n,))."""
    df = pd.read_csv(path)
    X = df[["x1", "x2"]].to_numpy(dtype=np.float32)
    y = df["y"].to_numpy(dtype=np.float32)
    return X, y


def load_splits(
    n_train: int | None = None,
    data_dir: str = DATA_DIR,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Nạp 3 tập đã lưu (train.csv, valid.csv, test.csv) – các thực nghiệm nên dùng hàm này.

    Tham số:
        n_train : nếu khác None, chỉ lấy n_train dòng ĐẦU của kho train (dùng cho Exp A).
                  valid và test luôn giữ nguyên.
        data_dir: thư mục chứa CSV, mặc định là data/ của dự án.
    """
    splits = {name: load_csv(os.path.join(data_dir, f"{name}.csv")) for name in SPLIT_SIZES}
    if n_train is not None:
        X, y = splits["train"]
        if n_train > len(X):
            raise ValueError(f"n_train={n_train} lớn hơn kích thước kho train ({len(X)})")
        splits["train"] = (X[:n_train], y[:n_train])
    return splits


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False) -> DataLoader:
    """Đóng gói (X, y) thành DataLoader của PyTorch.

    y được thêm một chiều (n,) → (n, 1) để khớp với đầu ra 1 neuron của MLP.
    shuffle=True chỉ nên dùng cho tập train.
    """
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
