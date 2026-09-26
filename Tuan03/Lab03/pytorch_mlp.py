from __future__ import annotations

import copy
import random
from typing import Optional

import numpy as np
import torch
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.model_selection import train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Mạng nơ-ron nhỏ dùng để so sánh với các mô hình hồi quy của scikit-learn.

class _MLP(nn.Module):
    def __init__(self, input_dim: int, hidden1: int, hidden2: int, hidden3: int, dropout: float):
        super().__init__()
        # Các lớp ẩn học quan hệ phi tuyến; Dropout giúp giảm học thuộc dữ liệu train.
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden1),
            nn.BatchNorm1d(hidden1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden2, hidden3),
            nn.ReLU(),
            nn.Linear(hidden3, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


class TorchMLPRegressor(BaseEstimator, RegressorMixin):
    """Small sklearn-compatible PyTorch regressor for log1p(SalePrice).

    It performs an internal validation split for early stopping. The outer CV still remains
    leak-free because preprocessing is fit only on each outer training fold by sklearn Pipeline.
    """

    def __init__(
        self,
        hidden1: int = 256,
        hidden2: int = 128,
        hidden3: int = 64,
        dropout: float = 0.18,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 280,
        patience: int = 28,
        validation_fraction: float = 0.15,
        random_state: int = 42,
        verbose: bool = False,
    ):
        self.hidden1 = hidden1
        self.hidden2 = hidden2
        self.hidden3 = hidden3
        self.dropout = dropout
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.validation_fraction = validation_fraction
        self.random_state = random_state
        self.verbose = verbose

    def _seed(self):
        # Cố định seed để các lần chạy có thể tái lập kết quả gần giống nhau.
        random.seed(self.random_state)
        np.random.seed(self.random_state)
        torch.manual_seed(self.random_state)

    def fit(self, X, y):
        # X đã được pipeline tiền xử lý; y là log1p(SalePrice).
        self._seed()
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)
        self.n_features_in_ = X.shape[1]

        X_tr, X_va, y_tr, y_va = train_test_split(
            X,
            y,
            test_size=self.validation_fraction,
            random_state=self.random_state,
        )
        # Tách một phần train làm validation nội bộ để quyết định early stopping.

        self.model_ = _MLP(
            self.n_features_in_, self.hidden1, self.hidden2, self.hidden3, self.dropout
        )
        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=max(4, self.patience // 4)
        )
        # Khi validation không tốt hơn, scheduler tự giảm learning rate.
        loss_fn = nn.MSELoss()

        train_ds = TensorDataset(torch.from_numpy(X_tr), torch.from_numpy(y_tr))
        generator = torch.Generator().manual_seed(self.random_state)
        loader = DataLoader(
            train_ds,
            batch_size=min(self.batch_size, len(train_ds)),
            shuffle=True,
            generator=generator,
        )
        X_va_t = torch.from_numpy(X_va)
        y_va_t = torch.from_numpy(y_va)

        best_loss = float("inf")
        best_state: Optional[dict] = None
        stale = 0
        self.history_ = []

        for epoch in range(1, self.max_epochs + 1):
            # Mỗi epoch gồm một lượt cập nhật trên toàn bộ mini-batch huấn luyện.
            self.model_.train()
            for xb, yb in loader:
                optimizer.zero_grad(set_to_none=True)
                pred = self.model_(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                optimizer.step()

            self.model_.eval()
            with torch.no_grad():
                va_pred = self.model_(X_va_t)
                va_mse = float(loss_fn(va_pred, y_va_t).item())
                va_rmse = va_mse ** 0.5
            scheduler.step(va_rmse)
            self.history_.append(va_rmse)

            if va_rmse < best_loss - 1e-6:
                best_loss = va_rmse
                best_state = copy.deepcopy(self.model_.state_dict())
                stale = 0
            else:
                stale += 1

            if self.verbose and (epoch == 1 or epoch % 25 == 0):
                print(f"epoch={epoch:4d} val_log_rmse={va_rmse:.6f}")
            if stale >= self.patience:
                # Dừng sớm khi nhiều epoch liên tiếp không cải thiện validation loss.
                break

        if best_state is not None:
            self.model_.load_state_dict(best_state)
        self.best_validation_rmse_ = float(best_loss)
        self.epochs_trained_ = len(self.history_)
        return self

    def predict(self, X):
        # Tắt dropout và không tính gradient khi dự đoán để tiết kiệm bộ nhớ/thời gian.
        if not hasattr(self, "model_"):
            raise RuntimeError("Estimator has not been fitted.")
        X = np.asarray(X, dtype=np.float32)
        self.model_.eval()
        with torch.no_grad():
            pred = self.model_(torch.from_numpy(X)).cpu().numpy()
        return pred.astype(np.float64)
