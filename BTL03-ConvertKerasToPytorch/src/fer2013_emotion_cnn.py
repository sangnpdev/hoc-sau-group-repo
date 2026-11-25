"""PyTorch port of the FER2013 Keras CNN examples in this project.

Expected CSV rows contain 2,304 grayscale pixels followed by 7 one-hot labels.
The pixels are kept on their original 0..255 scale to match the Keras notebook.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


CLASS_NAMES = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]
PIXELS_PER_IMAGE = 48 * 48
CSV_COLUMNS = PIXELS_PER_IMAGE + len(CLASS_NAMES)


class EmotionCNN(nn.Module):
    """CNN with the same convolution, pooling, dropout, and dense layers as Keras."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_dropout1 = nn.Dropout(p=0.25)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=0)
        self.conv2_3 = nn.Conv2d(128, 218, kernel_size=3, padding=1)
        self.conv2_4 = nn.Conv2d(218, 218, kernel_size=3, padding=0)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_dropout2 = nn.Dropout(p=0.25)

        # 48 -> 48 -> 46 -> 23 -> 23 -> 21 -> 21 -> 19 -> 9
        self.fc1 = nn.Linear(218 * 9 * 9, 1024)
        self.fc_dropout1 = nn.Dropout(p=0.5)
        self.fc3_7ways = nn.Linear(1024, len(CLASS_NAMES))

        # Match Keras Conv2D/Dense defaults: Glorot uniform kernels, zero biases.
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1_1(images))
        x = F.relu(self.conv1_2(x))
        x = self.conv_dropout1(self.pool1(x))

        x = F.relu(self.conv2_1(x))
        x = F.relu(self.conv2_2(x))
        x = F.relu(self.conv2_3(x))
        x = F.relu(self.conv2_4(x))
        x = self.conv_dropout2(self.pool2(x))

        # Keras Flatten traverses NHWC; transpose first to preserve that ordering.
        x = x.permute(0, 2, 3, 1).contiguous().view(x.shape[0], -1)
        x = self.fc_dropout1(F.relu(self.fc1(x)))
        # Return logits. CrossEntropyLoss applies the equivalent log-softmax safely.
        return self.fc3_7ways(x)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_fer_csv(csv_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
    """Load pixel + one-hot-label rows and return NCHW float pixels and class IDs."""
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    rows = np.loadtxt(csv_path, delimiter=",", dtype=np.float32, ndmin=2)
    if rows.shape[1] != CSV_COLUMNS:
        raise ValueError(
            f"Expected {CSV_COLUMNS} columns (2304 pixels + 7 one-hot labels), "
            f"found {rows.shape[1]} in {csv_path}"
        )
    if rows.shape[0] == 0 or not np.isfinite(rows).all():
        raise ValueError(f"CSV is empty or contains non-finite values: {csv_path}")

    pixels = rows[:, :PIXELS_PER_IMAGE].reshape(-1, 1, 48, 48)
    one_hot = rows[:, PIXELS_PER_IMAGE:]
    labels = one_hot.argmax(axis=1).astype(np.int64)
    if not np.allclose(one_hot.sum(axis=1), 1.0):
        raise ValueError(f"Expected one-hot labels with exactly one active class: {csv_path}")
    return torch.from_numpy(pixels.copy()), torch.from_numpy(labels)


def make_loader(images: torch.Tensor, labels: torch.Tensor, batch_size: int,
                shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(images, labels), batch_size=batch_size, shuffle=shuffle,
        num_workers=0, pin_memory=torch.cuda.is_available(), generator=generator,
    )


def augment_batch(images: torch.Tensor) -> torch.Tensor:
    """Approximate Keras ImageDataGenerator rotation/shift/flip using grid_sample.

    Uses bilinear interpolation and border padding (Keras fill_mode='nearest').
    Rotation range is ±45 degrees; horizontal/vertical shifts are up to 20%.
    """
    batch = images.shape[0]
    device = images.device
    angles = (torch.rand(batch, device=device) * 2.0 - 1.0) * math.radians(45)
    cos_a, sin_a = torch.cos(angles), torch.sin(angles)
    flip = torch.where(torch.rand(batch, device=device) < 0.5, -1.0, 1.0)
    tx = (torch.rand(batch, device=device) * 2.0 - 1.0) * 0.4
    ty = (torch.rand(batch, device=device) * 2.0 - 1.0) * 0.4

    theta = torch.zeros(batch, 2, 3, dtype=images.dtype, device=device)
    theta[:, 0, 0] = cos_a * flip
    theta[:, 0, 1] = -sin_a
    theta[:, 1, 0] = sin_a * flip
    theta[:, 1, 1] = cos_a
    theta[:, 0, 2] = tx
    theta[:, 1, 2] = ty
    grid = F.affine_grid(theta, images.shape, align_corners=False)
    return F.grid_sample(images, grid, mode="bilinear", padding_mode="border", align_corners=False)


def run_epoch(model: EmotionCNN, loader: DataLoader, device: torch.device,
              optimizer: torch.optim.Optimizer | None = None,
              use_augmentation: bool = False) -> tuple[float, float]:
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if training and use_augmentation:
            images = augment_batch(images)
        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = F.cross_entropy(logits, labels)
            if training:
                loss.backward()
                optimizer.step()

        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(dim=1) == labels).sum().item()
        count += labels.size(0)

    return total_loss / count, correct / count


def save_history_plot(history: dict[str, list[float]], output_path: Path, title: str) -> None:
    epochs = range(1, len(history["train_loss"]) + 1)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, history["train_loss"], label="Train")
    axes[0].plot(epochs, history["val_loss"], label="PublicTest / validation")
    axes[0].set(title="Cross-entropy loss", xlabel="Epoch", ylabel="Loss")
    axes[1].plot(epochs, history["train_accuracy"], label="Train")
    axes[1].plot(epochs, history["val_accuracy"], label="PublicTest / validation")
    axes[1].set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def save_sample_prediction(model: EmotionCNN, images: torch.Tensor, labels: torch.Tensor,
                          device: torch.device, output_path: Path, seed: int) -> dict[str, object]:
    index = int(np.random.default_rng(seed).integers(0, len(images)))
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(images[index:index + 1].to(device)), dim=1)[0].cpu()
    predicted = int(probabilities.argmax())
    true_label = int(labels[index])
    fig, axis = plt.subplots(figsize=(4, 4))
    axis.imshow(images[index, 0].numpy(), cmap="gray", vmin=0, vmax=255)
    axis.set_title(f"True: {CLASS_NAMES[true_label]} | Predicted: {CLASS_NAMES[predicted]}")
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    return {
        "index": index,
        "true_class": CLASS_NAMES[true_label],
        "predicted_class": CLASS_NAMES[predicted],
        "probabilities": probabilities.tolist(),
    }


def run_variant(variant: str, train_csv: Path, validation_csv: Path,
                output_dir: Path, batch_size: int, seed: int,
                device: torch.device, epochs_override: int | None = None,
                patience: int = 5) -> dict[str, object]:
    if variant not in {"basic", "utilities"}:
        raise ValueError(f"Unknown variant: {variant}")
    set_seed(seed)
    train_images, train_labels = load_fer_csv(train_csv)
    val_images, val_labels = load_fer_csv(validation_csv)
    train_loader = make_loader(train_images, train_labels, batch_size, True, seed)
    val_loader = make_loader(val_images, val_labels, batch_size, False, seed)

    use_augmentation = variant == "utilities"
    epochs = epochs_override if epochs_override is not None else (15 if variant == "basic" else 100)
    model = EmotionCNN().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    history: dict[str, list[float]] = {
        "train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []
    }
    best_val_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    started = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, device, optimizer, use_augmentation
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, device)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(
            f"{variant} {epoch:03d}/{epochs}: "
            f"train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}"
        )

        if use_augmentation:
            if val_loss < best_val_loss:
                best_val_loss, best_epoch, epochs_without_improvement = val_loss, epoch, 0
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= patience:
                    print(f"Early stopping at epoch {epoch}; best validation loss at {best_epoch}.")
                    break

    elapsed_minutes = (time.time() - started) / 60.0
    final_val_loss, final_val_accuracy = run_epoch(model, val_loader, device)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"fer2013_{variant}_final.pt"
    torch.save({
        "state_dict": model.state_dict(),
        "variant": variant,
        "class_names": CLASS_NAMES,
        "input_shape": [1, 48, 48],
        "seed": seed,
        "final_epoch": len(history["train_loss"]),
    }, model_path)
    (output_dir / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    save_history_plot(history, output_dir / "training_history.png", f"FER2013 CNN · {variant}")
    sample_prediction = save_sample_prediction(
        model, val_images, val_labels, device, output_dir / "sample_prediction.png", seed
    )

    metrics = {
        "variant": variant,
        "device": str(device),
        "epochs_completed": len(history["train_loss"]),
        "best_validation_loss_epoch": best_epoch if use_augmentation else None,
        "final_publictest_loss": final_val_loss,
        "final_publictest_accuracy": final_val_accuracy,
        "training_time_minutes": elapsed_minutes,
        "batch_size": batch_size,
        "learning_rate": 1e-4,
        "seed": seed,
        "augmentation": use_augmentation,
        "early_stopping_patience": patience if use_augmentation else None,
        "model_file": model_path.name,
        "class_names": CLASS_NAMES,
        "sample_prediction": sample_prediction,
    }
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model, history, plot, and metrics to {output_dir.resolve()}")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="PyTorch port of the FER2013 Keras CNN notebooks.")
    parser.add_argument("--train-csv", type=Path, default=Path("fer2013_training_onehot.csv"))
    parser.add_argument("--validation-csv", type=Path, default=Path("fer2013_publictest_onehot.csv"))
    parser.add_argument("--variant", choices=("basic", "utilities", "both"), default="both")
    parser.add_argument("--output-dir", type=Path, default=Path("fer2013_pytorch_results"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=None,
                        help="Override source defaults (15 basic / 100 utilities).")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if args.batch_size <= 0 or (args.epochs is not None and args.epochs <= 0) or args.patience <= 0:
        parser.error("batch-size, epochs, and patience must be positive integers")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or
                          (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    variants = ["basic", "utilities"] if args.variant == "both" else [args.variant]
    results = {}
    for variant in variants:
        results[variant] = run_variant(
            variant=variant,
            train_csv=args.train_csv,
            validation_csv=args.validation_csv,
            output_dir=args.output_dir / variant,
            batch_size=args.batch_size,
            seed=args.seed,
            device=device,
            epochs_override=args.epochs,
            patience=args.patience,
        )
    (args.output_dir / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
