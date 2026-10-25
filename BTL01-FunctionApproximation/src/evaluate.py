import matplotlib
matplotlib.use("Agg")  # headless-safe; no display required

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader


def compute_metrics(
    model: nn.Module,
    loader: DataLoader,
    device: str = "cpu",
) -> tuple[dict, np.ndarray, np.ndarray]:
    """Return (metrics_dict, y_true, y_pred) on the given loader."""
    model.eval()
    yt, yp = [], []
    with torch.no_grad():
        for X_b, y_b in loader:
            yp.append(model(X_b.to(device)).cpu().numpy())
            yt.append(y_b.numpy())
    y_true = np.concatenate(yt).flatten()
    y_pred = np.concatenate(yp).flatten()
    mse  = float(np.mean((y_true - y_pred) ** 2))
    mae  = float(np.mean(np.abs(y_true - y_pred)))
    rmse = float(np.sqrt(mse))
    return {"MSE": mse, "MAE": mae, "RMSE": rmse}, y_true, y_pred


def plot_loss(logs: list[dict], save_path: str) -> None:
    epochs = [r["epoch"] for r in logs]
    plt.figure(figsize=(8, 4))
    plt.plot(epochs, [r["train_loss"] for r in logs], label="Train")
    plt.plot(epochs, [r["val_loss"]   for r in logs], label="Val")
    plt.xlabel("Epoch")
    plt.ylabel("MSE Loss")
    plt.title("Loss curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()


def plot_pred_vs_true(y_true: np.ndarray, y_pred: np.ndarray, save_path: str) -> None:
    lo = min(y_true.min(), y_pred.min())
    hi = max(y_true.max(), y_pred.max())
    plt.figure(figsize=(5, 5))
    plt.scatter(y_true, y_pred, alpha=0.4, s=10)
    plt.plot([lo, hi], [lo, hi], "r--", linewidth=1)
    plt.xlabel("y true")
    plt.ylabel("y pred")
    plt.title("Predicted vs True")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
