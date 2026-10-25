"""Baseline run: 1 hidden layer, 10 nodes, A=1000 train samples."""
import csv
import os
import random

import numpy as np
import torch

from src.data import generate_data, make_loader
from src.evaluate import compute_metrics, plot_loss, plot_pred_vs_true
from src.model import MLP
from src.train import train

# ── Config ──────────────────────────────────────────────────────────────────
SEED         = 42
N_TRAIN      = 1000
N_VAL        = 200
N_TEST       = 100
HIDDEN_SIZES = [10]   # 1 hidden layer, 10 neurons
EPOCHS       = 200
BATCH_SIZE   = 128
LR           = 1e-3
OUT_DIR      = "outputs/baseline"
MODEL_PATH   = "outputs/models/baseline_best.pt"
# ─────────────────────────────────────────────────────────────────────────────


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    set_seed(SEED)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs("outputs/models", exist_ok=True)

    # Independent seeds for each split so sets never overlap
    X_train, y_train = generate_data(N_TRAIN, seed=SEED)
    X_val,   y_val   = generate_data(N_VAL,   seed=SEED + 1)
    X_test,  y_test  = generate_data(N_TEST,  seed=SEED + 2)

    train_loader = make_loader(X_train, y_train, BATCH_SIZE, shuffle=True)
    val_loader   = make_loader(X_val,   y_val,   BATCH_SIZE)
    test_loader  = make_loader(X_test,  y_test,  BATCH_SIZE)

    model = MLP(hidden_sizes=HIDDEN_SIZES)
    print(model)
    print(f"\nTraining: A={N_TRAIN}, hidden={HIDDEN_SIZES}, epochs={EPOCHS}, lr={LR}")
    print("-" * 60)

    logs, elapsed = train(model, train_loader, val_loader, EPOCHS, LR, MODEL_PATH)

    # Save epoch log
    log_path = os.path.join(OUT_DIR, "log.csv")
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss"])
        writer.writeheader()
        writer.writerows(logs)

    # Load best checkpoint and evaluate on test
    model.load_state_dict(torch.load(MODEL_PATH, weights_only=True))
    metrics, y_true, y_pred = compute_metrics(model, test_loader)

    print("-" * 60)
    print(f"Test  | MSE={metrics['MSE']:.6f} | MAE={metrics['MAE']:.6f} | RMSE={metrics['RMSE']:.6f}")
    print(f"Training time: {elapsed:.1f}s")

    plot_loss(logs, save_path=os.path.join(OUT_DIR, "loss_curve.png"))
    plot_pred_vs_true(y_true, y_pred, save_path=os.path.join(OUT_DIR, "pred_vs_true.png"))
    print(f"\nOutputs saved to {OUT_DIR}/")


if __name__ == "__main__":
    main()
