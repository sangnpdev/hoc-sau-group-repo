import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int,
    lr: float,
    checkpoint_path: str,
    device: str = "cpu",
    print_every: int = 20,
) -> tuple[list[dict], float]:
    """Train model, save best checkpoint by val loss.

    Returns:
        logs: list of {epoch, train_loss, val_loss} per epoch.
        elapsed: total training time in seconds.
    """
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float("inf")
    logs: list[dict] = []
    start = time.perf_counter()

    for epoch in range(1, epochs + 1):
        model.train()
        running = 0.0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_b), y_b)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(X_b)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        running = 0.0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                running += criterion(model(X_b), y_b).item() * len(X_b)
        val_loss = running / len(val_loader.dataset)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), checkpoint_path)

        logs.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss})
        if epoch % print_every == 0:
            print(f"Epoch {epoch:4d} | train={train_loss:.6f} | val={val_loss:.6f}")

    elapsed = time.perf_counter() - start
    return logs, elapsed
