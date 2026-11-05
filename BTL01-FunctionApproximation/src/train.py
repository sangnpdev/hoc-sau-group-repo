"""Vòng lặp huấn luyện / validation viết tay, early stopping và checkpoint theo validation loss."""
import copy
import json
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from src.data import load_split, make_loader, to_tensors
from src.evaluate import metrics, predict
from src.model import MLP, count_params

# Cấu hình mặc định; tune.py sẽ ghi đè lr/activation/batch_size bằng kết quả tìm kiếm trên validation.
DEFAULTS = {
    "n_train": 1000,
    "hidden_sizes": [10],
    "activation": "tanh",
    "optimizer": "adam",
    "lr": 3e-3,
    "batch_size": 32,
    "max_epochs": 3000,
    "patience": 200,          # early stopping: dừng nếu val loss không giảm sau 200 epoch
    "min_rel_improve": 1e-4,  # cải thiện tương đối tối thiểu để tính là "tốt hơn"
    "plateau_patience": 50,   # ReduceLROnPlateau: giảm lr x0.5 nếu val không giảm sau 50 epoch
    "plateau_factor": 0.5,
    "min_lr": 1e-5,
    "seed": 0,
    "device": "cpu",
}


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_optimizer(name, params, lr):
    if name == "adam":
        return torch.optim.Adam(params, lr=lr, foreach=False)
    if name == "sgd":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    raise ValueError(name)


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    """Một epoch train (optimizer != None) hoặc đánh giá (optimizer is None). Trả về loss trung bình theo mẫu."""
    training = optimizer is not None
    model.train(training)
    total, n = 0.0, 0
    with torch.set_grad_enabled(training):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)                 # forward
            loss = loss_fn(pred, yb)         # MSE
            if training:
                optimizer.zero_grad()        # xoá gradient cũ
                loss.backward()              # backpropagation
                optimizer.step()             # cập nhật trọng số
            total += loss.item() * len(xb)
            n += len(xb)
    return total / n


def train_run(cfg, data_dir, out_dir):
    """Huấn luyện một cấu hình. Lưu config.json, log.csv, best.pt, metrics.json vào out_dir.
    Nếu metrics.json đã tồn tại thì bỏ qua (cho phép chạy tiếp khi bị ngắt)."""
    cfg = {**DEFAULTS, **cfg}
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as f:
            return json.load(f)
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(cfg, f, indent=2)

    set_seed(cfg["seed"])
    device = cfg["device"]
    x_tr, y_tr = load_split(data_dir, "train_pool", cfg["n_train"])
    x_va, y_va = load_split(data_dir, "valid")
    x_te, y_te = load_split(data_dir, "test")
    train_loader = make_loader(x_tr, y_tr, cfg["batch_size"], shuffle=True, seed=cfg["seed"])
    xv, yv = (t.to(device) for t in to_tensors(x_va, y_va))

    model = MLP(cfg["hidden_sizes"], cfg["activation"]).to(device)
    loss_fn = nn.MSELoss()
    optimizer = make_optimizer(cfg["optimizer"], model.parameters(), cfg["lr"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=cfg["plateau_factor"],
        patience=cfg["plateau_patience"], min_lr=cfg["min_lr"])

    best_val, best_epoch, best_state, wait = float("inf"), -1, None, 0
    logs = []
    t0 = time.perf_counter()
    for epoch in range(1, cfg["max_epochs"] + 1):
        train_loss = run_epoch(model, train_loader, loss_fn, device, optimizer)
        model.eval()
        with torch.no_grad():
            val_loss = loss_fn(model(xv), yv).item()
        scheduler.step(val_loss)
        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
                     "lr": optimizer.param_groups[0]["lr"], "time_s": time.perf_counter() - t0})
        if val_loss < best_val * (1 - cfg["min_rel_improve"]):
            best_val, best_epoch, wait = val_loss, epoch, 0
            best_state = copy.deepcopy(model.state_dict())   # checkpoint tốt nhất theo validation
        else:
            wait += 1
            if wait >= cfg["patience"]:
                break
    train_time = time.perf_counter() - t0
    n_epochs = len(logs)

    model.load_state_dict(best_state)
    torch.save({"state_dict": best_state, "config": cfg, "best_epoch": best_epoch},
               os.path.join(out_dir, "best.pt"))
    pd.DataFrame(logs).to_csv(os.path.join(out_dir, "log.csv"), index=False)

    result = {"name": cfg.get("name", os.path.basename(out_dir)), "n_train": cfg["n_train"],
              "hidden_sizes": "-".join(map(str, cfg["hidden_sizes"])), "depth": len(cfg["hidden_sizes"]),
              "width": cfg["hidden_sizes"][0], "activation": cfg["activation"], "lr": cfg["lr"],
              "batch_size": cfg["batch_size"], "seed": cfg["seed"], "n_params": count_params(model),
              "best_epoch": best_epoch, "epochs_run": n_epochs, "train_time_s": train_time,
              "time_per_epoch_ms": 1000 * train_time / n_epochs,
              "n_steps": n_epochs * len(train_loader)}
    for split, (x, y) in {"train": (x_tr, y_tr), "val": (x_va, y_va), "test": (x_te, y_te)}.items():
        for k, v in metrics(y, predict(model, x, device)).items():
            result[f"{split}_{k}"] = v
    with open(metrics_path, "w") as f:
        json.dump(result, f, indent=2)
    return result
