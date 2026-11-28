"""Train and evaluate PyTorch CNNs on the folder-form FER2013 dataset.

BTL02 - Nhan dien cam xuc khuon mat (FER2013). Phan cong:
- 3123411255 Nguyen Phuoc Sang (nhom truong): kien truc EmotionCNN, vong lap huan luyen,
  early stopping, checkpoint; tong hop ket qua va bao cao.
- 3123411009 Tran Quoc An: xu ly du lieu, tien xu ly va tang cuong du lieu
  (make_transform, make_loaders).
- 3123411042 Nguyen Nguyen Chuong: thiet ke va chay 3 thi nghiem tren Colab, so sanh duong cong hoc.
- 3123411279 Nguyen Gia Thieu: danh gia tren PrivateTest (confusion matrix, ca du doan sai,
  truc quan hoa bo loc), slide thuyet trinh.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


EXPERIMENTS = [
    {"name": "small", "channels": (32, 64, 128), "dropout": 0.25,
     "learning_rate": 1e-3, "weight_decay": 0.0, "augmentation": False},
    {"name": "medium_aug", "channels": (32, 64, 128, 256), "dropout": 0.35,
     "learning_rate": 1e-3, "weight_decay": 0.0, "augmentation": True},
    {"name": "medium_adamw", "channels": (32, 64, 128, 256), "dropout": 0.40,
     "learning_rate": 5e-4, "weight_decay": 1e-4, "augmentation": True},
]
CLASS_ORDER = ["angry", "disgust", "fear", "happy", "sad", "surprise", "neutral"]


class EmotionCNN(nn.Module):
    def __init__(self, channels: tuple[int, ...], dropout: float, num_classes: int):
        super().__init__()
        blocks: list[nn.Module] = []
        in_channels = 1
        for out_channels in channels:
            blocks.extend([
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Dropout2d(dropout),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*blocks)
        self.pool = nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_channels * 4, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.pool(self.features(images)))


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_transform(augment: bool) -> transforms.Compose:
    steps: list[object] = [transforms.Grayscale(num_output_channels=1)]
    if augment:
        steps.extend([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        ])
    steps.extend([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    return transforms.Compose(steps)


def make_loaders(data_dir: Path, batch_size: int, workers: int, augment: bool):
    train_root, test_root = data_dir / "train", data_dir / "test"
    if not train_root.is_dir() or not test_root.is_dir():
        raise FileNotFoundError(f"Expected class folders under {train_root} and {test_root}")

    train_augmented = datasets.ImageFolder(train_root, transform=make_transform(augment))
    train_evaluation = datasets.ImageFolder(train_root, transform=make_transform(False))
    test_dataset = datasets.ImageFolder(test_root, transform=make_transform(False))
    if set(train_augmented.classes) != set(CLASS_ORDER) or set(test_dataset.classes) != set(CLASS_ORDER):
        raise ValueError(f"Expected FER2013 classes {CLASS_ORDER}; found {train_augmented.classes} and {test_dataset.classes}")

    def set_class_order(dataset):
        old_classes = dataset.classes
        old_to_new = {old_index: CLASS_ORDER.index(class_name)
                      for old_index, class_name in enumerate(old_classes)}
        dataset.samples = [(path, old_to_new[label]) for path, label in dataset.samples]
        dataset.targets = [label for _, label in dataset.samples]
        dataset.classes = list(CLASS_ORDER)
        dataset.class_to_idx = {name: index for index, name in enumerate(CLASS_ORDER)}

    for folder in (train_augmented, train_evaluation, test_dataset):
        set_class_order(folder)
    classes = list(CLASS_ORDER)

    validation_indices, private_indices = [], []
    for index, (path, _) in enumerate(test_dataset.samples):
        name = Path(path).name
        if name.startswith("PublicTest_"):
            validation_indices.append(index)
        elif name.startswith("PrivateTest_"):
            private_indices.append(index)
    if not validation_indices or not private_indices:
        raise ValueError("Expected PublicTest_ and PrivateTest_ images in data/test")

    pin_memory = torch.cuda.is_available()
    common = {"batch_size": batch_size, "num_workers": workers,
              "pin_memory": pin_memory, "persistent_workers": workers > 0}
    train_loader = DataLoader(train_augmented, shuffle=True, **common)
    validation_loader = DataLoader(Subset(test_dataset, validation_indices), shuffle=False, **common)
    private_loader = DataLoader(Subset(test_dataset, private_indices), shuffle=False, **common)
    train_eval_loader = DataLoader(train_evaluation, shuffle=False, **common)
    return train_loader, validation_loader, private_loader, train_eval_loader, classes


def run_epoch(model, loader, loss_fn, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, correct, count = 0.0, 0, 0
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = loss_fn(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        total_loss += loss.item() * labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()
        count += labels.size(0)
    return total_loss / count, correct / count


def train_experiment(config, args, device, classes, output_dir):
    train_loader, val_loader, _, _, _ = make_loaders(
        args.data_dir, args.batch_size, args.workers, config["augmentation"])
    model = EmotionCNN(config["channels"], config["dropout"], len(classes)).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"],
                                  weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=2)
    history = {"train_loss": [], "train_accuracy": [], "val_loss": [], "val_accuracy": []}
    best_loss, best_accuracy, best_epoch, stale_epochs = float("inf"), 0.0, 0, 0
    checkpoint_path = output_dir / f"{config['name']}_best.pt"

    print(f"\n=== {config['name']} | channels={config['channels']} | augmentation={config['augmentation']} ===")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(model, train_loader, loss_fn, device, optimizer)
        val_loss, val_accuracy = run_epoch(model, val_loader, loss_fn, device)
        scheduler.step(val_loss)
        history["train_loss"].append(train_loss)
        history["train_accuracy"].append(train_accuracy)
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        print(f"{epoch:02d}/{args.epochs} train_loss={train_loss:.4f} train_acc={train_accuracy:.4f} "
              f"val_loss={val_loss:.4f} val_acc={val_accuracy:.4f}", flush=True)
        if val_loss < best_loss:
            best_loss, best_accuracy, best_epoch, stale_epochs = val_loss, val_accuracy, epoch, 0
            torch.save({"state_dict": model.state_dict(), "config": config,
                        "classes": classes, "epoch": epoch}, checkpoint_path)
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping at epoch {epoch} (best epoch {best_epoch}).")
                break

    return {"name": config["name"], "config": config, "history": history,
            "best_val_loss": best_loss, "best_val_accuracy": best_accuracy,
            "best_epoch": best_epoch, "checkpoint": str(checkpoint_path)}


def predict(model, loader, device):
    model.eval()
    all_labels, all_predictions, all_images = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            all_images.append(images.cpu())
            all_labels.append(labels.cpu())
            all_predictions.append(logits.argmax(1).cpu())
    return (torch.cat(all_images), torch.cat(all_labels), torch.cat(all_predictions))


def save_plots(results, model, private_loader, device, classes, output_dir):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for result in results:
        epochs = range(1, len(result["history"]["train_accuracy"]) + 1)
        axes[0].plot(epochs, result["history"]["val_accuracy"], label=result["name"])
        axes[1].plot(epochs, result["history"]["val_loss"], label=result["name"])
    axes[0].set(title="Validation accuracy", xlabel="Epoch", ylabel="Accuracy")
    axes[1].set(title="Validation loss", xlabel="Epoch", ylabel="Cross-entropy loss")
    for axis in axes:
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "experiment_comparison.png", dpi=160)
    plt.show()
    plt.close(fig)

    images, labels, predictions = predict(model, private_loader, device)
    matrix = np.zeros((len(classes), len(classes)), dtype=np.int64)
    np.add.at(matrix, (labels.numpy(), predictions.numpy()), 1)
    row_sums = matrix.sum(axis=1, keepdims=True)
    normalized = np.divide(matrix, row_sums, out=np.zeros_like(matrix, dtype=float), where=row_sums != 0)
    fig, axis = plt.subplots(figsize=(8, 7))
    image = axis.imshow(normalized, cmap="Blues", vmin=0, vmax=1)
    axis.set(xticks=range(len(classes)), yticks=range(len(classes)),
             xticklabels=classes, yticklabels=classes,
             xlabel="Predicted emotion", ylabel="True emotion",
             title="Private test normalized confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=40, ha="right", rotation_mode="anchor")
    for row in range(len(classes)):
        for column in range(len(classes)):
            axis.text(column, row, f"{normalized[row, column]:.2f}", ha="center", va="center",
                      color="white" if normalized[row, column] > 0.55 else "black", fontsize=8)
    fig.colorbar(image, ax=axis, label="Fraction per true class")
    fig.tight_layout()
    fig.savefig(output_dir / "private_confusion_matrix.png", dpi=160)
    plt.show()
    plt.close(fig)

    wrong = torch.cat([
        torch.where((labels == class_index) & (labels != predictions))[0][:3]
        for class_index in range(len(classes))
    ])
    if len(wrong):
        columns = 5
        rows = int(np.ceil(len(wrong) / columns))
        fig, axes = plt.subplots(rows, columns, figsize=(12, 2.5 * rows))
        axes = np.array(axes).reshape(-1)
        for axis in axes:
            axis.axis("off")
        for axis, index in zip(axes, wrong):
            axis.imshow(images[index, 0] * 0.5 + 0.5, cmap="gray", vmin=0, vmax=1)
            axis.set_title(f"true: {classes[labels[index]]}\npred: {classes[predictions[index]]}", fontsize=8)
        fig.suptitle("Private test misclassified examples")
        fig.tight_layout()
        fig.savefig(output_dir / "private_misclassified.png", dpi=160)
        plt.show()
        plt.close(fig)

    last_conv = [module for module in model.modules() if isinstance(module, nn.Conv2d)][-1]
    # Last-layer kernels span many input channels; mean absolute weight per output channel
    # gives a compact 2-D view of each learned spatial kernel.
    filters = last_conv.weight.detach().abs().mean(dim=1).cpu().numpy()
    columns = 8
    rows = int(np.ceil(len(filters) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(12, 1.45 * rows))
    axes = np.array(axes).reshape(-1)
    for index, axis in enumerate(axes):
        axis.axis("off")
        if index < len(filters):
            axis.imshow(filters[index], cmap="viridis")
            axis.set_title(f"filter {index}", fontsize=7)
    fig.suptitle("Final convolution layer: mean absolute kernel weights", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(output_dir / "final_conv_filters.png", dpi=180)
    plt.show()
    plt.close(fig)

    return {"test_accuracy": float((labels == predictions).float().mean()),
            "confusion_matrix": matrix.tolist(), "classes": classes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True,
                        help="Directory containing train/ and test/ class folders")
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    _, _, _, _, classes = make_loaders(args.data_dir, args.batch_size, args.workers, False)
    results = [train_experiment(config, args, device, classes, args.output_dir)
               for config in EXPERIMENTS]
    best = max(results, key=lambda result: result["best_val_accuracy"])
    saved = torch.load(best["checkpoint"], map_location=device, weights_only=False)
    best_model = EmotionCNN(saved["config"]["channels"], saved["config"]["dropout"], len(classes)).to(device)
    best_model.load_state_dict(saved["state_dict"])
    _, _, private_loader, _, _ = make_loaders(
        args.data_dir, args.batch_size, args.workers, False)
    final_metrics = save_plots(results, best_model, private_loader, device, classes, args.output_dir)
    summary = {
        "device": str(device), "classes": classes,
        "experiments": [{key: value for key, value in result.items() if key != "history"}
                        for result in results],
        "best_experiment": best["name"], "best_epoch": best["best_epoch"],
        "best_validation_accuracy": best["best_val_accuracy"],
        "private_test_accuracy": final_metrics["test_accuracy"],
        "private_test_confusion_matrix": final_metrics["confusion_matrix"],
    }
    (args.output_dir / "results.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (args.output_dir / "training_history.json").open("w", encoding="utf-8") as stream:
        json.dump({result["name"]: result["history"] for result in results}, stream, indent=2)
    print("\nExperiment summary:")
    for result in results:
        print(f"{result['name']}: best val acc={result['best_val_accuracy']:.4f} "
              f"at epoch {result['best_epoch']}")
    print(f"Selected: {best['name']} | private test acc={final_metrics['test_accuracy']:.4f}")
    print(f"Artifacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
