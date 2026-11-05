"""Dự đoán và tính chỉ số đánh giá (trên thang đo gốc của y)."""
import numpy as np
import torch

from src.data import scale_x, unscale_y


@torch.no_grad()
def predict(model, x, device="cpu"):
    """x: numpy (N, 2) thang gốc → ŷ numpy (N,) thang gốc."""
    model.eval()
    xt = torch.from_numpy(scale_x(x)).to(device)
    return unscale_y(model(xt).squeeze(1).cpu().numpy().astype(np.float64))


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    err = np.asarray(y_pred, dtype=np.float64) - y_true
    mse = float(np.mean(err ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return {"mse": mse, "rmse": mse ** 0.5, "mae": float(np.mean(np.abs(err))),
            "r2": 1.0 - float(np.sum(err ** 2)) / ss_tot}
