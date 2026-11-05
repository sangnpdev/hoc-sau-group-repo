"""Sinh, lưu, nạp dữ liệu Gaussian 2 chiều và tạo DataLoader."""
import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import BatchSampler, DataLoader, RandomSampler, SequentialSampler, TensorDataset

MU = np.array([0.0, 0.0])
COV = np.array([[0.3, -0.5], [-0.5, 2.0]])
X1_RANGE = (-2.0, 2.0)
X2_RANGE = (-4.0, 4.0)

DATA_SEED = 2024          # seed duy nhất dùng để sinh toàn bộ dữ liệu
N_TRAIN_POOL = 2000       # pool train; nhóm A dùng A mẫu đầu tiên của pool
N_VALID = 200             # validation dùng cho early stopping / chọn mô hình
N_TEST = 100              # test độc lập, chỉ dùng đánh giá cuối


def gaussian2d(x, mu=MU, cov=COV):
    """f(x) = 1 / (2*pi*sqrt(det C)) * exp(-1/2 (x-mu)^T C^-1 (x-mu)), x có shape (N, 2)."""
    x = np.asarray(x, dtype=np.float64)
    d = x - mu
    inv = np.linalg.inv(cov)
    mahal = np.einsum("ni,ij,nj->n", d, inv, d)
    return np.exp(-0.5 * mahal) / (2.0 * np.pi * np.sqrt(np.linalg.det(cov)))


Y_MAX = float(gaussian2d(MU[None, :])[0])  # giá trị cực đại của f, = 1/(2*pi*sqrt(0.35)) ~ 0.269


def sample_inputs(n, rng):
    """Lấy mẫu đều x1 ~ U[-2, 2], x2 ~ U[-4, 4]."""
    x1 = rng.uniform(*X1_RANGE, size=n)
    x2 = rng.uniform(*X2_RANGE, size=n)
    return np.stack([x1, x2], axis=1)


def generate_datasets(data_dir, seed=DATA_SEED):
    """Sinh 3 tập độc lập từ cùng một Generator (các lần gọi liên tiếp → không trùng mẫu) và lưu CSV."""
    os.makedirs(data_dir, exist_ok=True)
    rng = np.random.default_rng(seed)
    sizes = {"train_pool": N_TRAIN_POOL, "valid": N_VALID, "test": N_TEST}
    for name, n in sizes.items():
        x = sample_inputs(n, rng)
        y = gaussian2d(x)
        pd.DataFrame({"x1": x[:, 0], "x2": x[:, 1], "y": y}).to_csv(
            os.path.join(data_dir, f"{name}.csv"), index=False)


def load_split(data_dir, name, n=None):
    df = pd.read_csv(os.path.join(data_dir, f"{name}.csv"))
    if n is not None:
        df = df.iloc[:n]
    return df[["x1", "x2"]].to_numpy(np.float32), df["y"].to_numpy(np.float32)


# --- Chuẩn hoá: dùng biên miền đã biết (không phụ thuộc dữ liệu → không rò rỉ thông tin) ---
def scale_x(x):
    return np.stack([x[:, 0] / X1_RANGE[1], x[:, 1] / X2_RANGE[1]], axis=1).astype(np.float32)


def scale_y(y):
    return (y / Y_MAX).astype(np.float32)


def unscale_y(y_scaled):
    return y_scaled * Y_MAX


def to_tensors(x, y):
    return torch.from_numpy(scale_x(x)), torch.from_numpy(scale_y(y)).unsqueeze(1)


def make_loader(x, y, batch_size, shuffle, seed=0):
    g = torch.Generator()
    g.manual_seed(seed)
    ds = TensorDataset(*to_tensors(x, y))
    # BatchSampler lấy cả lô chỉ số một lần (nhanh hơn gom từng mẫu); shuffle dùng Generator có seed.
    base = RandomSampler(ds, generator=g) if shuffle else SequentialSampler(ds)
    return DataLoader(ds, batch_size=None, sampler=BatchSampler(base, batch_size, drop_last=False))
