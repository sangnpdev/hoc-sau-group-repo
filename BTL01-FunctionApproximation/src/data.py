import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader
from scipy.stats import multivariate_normal

MU = np.array([0.0, 0.0])
COV = np.array([[0.3, -0.5], [-0.5, 2.0]])


def generate_data(n: int, seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Sample n points uniformly from domain, compute Gaussian targets."""
    rng = np.random.default_rng(seed)
    x1 = rng.uniform(-2.0, 2.0, n)
    x2 = rng.uniform(-4.0, 4.0, n)
    pts = np.column_stack([x1, x2])
    y = multivariate_normal.pdf(pts, mean=MU, cov=COV)
    X = pts.astype(np.float32)
    y = y.astype(np.float32)
    return X, y


def make_loader(X: np.ndarray, y: np.ndarray, batch_size: int, shuffle: bool = False) -> DataLoader:
    ds = TensorDataset(torch.from_numpy(X), torch.from_numpy(y).unsqueeze(1))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle)
