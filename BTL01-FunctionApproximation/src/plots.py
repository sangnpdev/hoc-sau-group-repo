"""Xuất toàn bộ hình vẽ vào results/figures/."""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

from src import data  # noqa: E402
from src.evaluate import predict  # noqa: E402
from src.experiments import DATA_DIR, MODELS_DIR, RESULTS_DIR, ROOT, RUNS_DIR  # noqa: E402
from src.model import MLP  # noqa: E402

FIG_DIR = os.path.join(RESULTS_DIR, "figures")
# Bảng màu phân loại cố định: train / validation / test luôn cùng một màu ở mọi hình.
C_TRAIN, C_VAL, C_TEST = "#2a78d6", "#eb6834", "#1baf7a"
SPLIT_COLORS = {"train": C_TRAIN, "val": C_VAL, "test": C_TEST}
SPLIT_LABELS = {"train": "Train", "val": "Validation", "test": "Test"}
SEQ_CMAP = "Blues"
GRID = dict(color="#d9d9d6", linewidth=0.6)

plt.rcParams.update({"figure.dpi": 110, "savefig.dpi": 150, "font.size": 10, "axes.titlesize": 11,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#8a8984",
                     "axes.labelcolor": "#0b0b0b", "xtick.color": "#52514e", "ytick.color": "#52514e",
                     "legend.frameon": False})


def _save(fig, name):
    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(os.path.join(FIG_DIR, name), bbox_inches="tight")
    plt.close(fig)


def _grid_xy(n=121):
    g1 = np.linspace(*data.X1_RANGE, n)
    g2 = np.linspace(*data.X2_RANGE, n)
    G1, G2 = np.meshgrid(g1, g2)
    return G1, G2, np.stack([G1.ravel(), G2.ravel()], axis=1)


def load_model(path):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    m = MLP(ck["config"]["hidden_sizes"], ck["config"]["activation"])
    m.load_state_dict(ck["state_dict"])
    return m, ck


# ---------------------------------------------------------------- dữ liệu
def plot_data():
    splits = {"train": load_split_full("train_pool"), "val": load_split_full("valid"), "test": load_split_full("test")}
    fig = plt.figure(figsize=(15, 4.6))
    for i, (k, (x, y)) in enumerate(splits.items()):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.scatter(x[:, 0], x[:, 1], y, s=4, c=SPLIT_COLORS[k], alpha=0.8)
        ax.set_title(f"{SPLIT_LABELS[k]} (N = {len(y)})")
        ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.set_zlabel("y")
        ax.view_init(28, -55)
    _save(fig, "data_scatter3d.png")

    G1, G2, X = _grid_xy()
    Z = data.gaussian2d(X).reshape(G1.shape)
    fig = plt.figure(figsize=(12, 4.8))
    ax = fig.add_subplot(1, 2, 1, projection="3d")
    ax.plot_surface(G1, G2, Z, cmap=SEQ_CMAP, linewidth=0, antialiased=True)
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.set_zlabel("f(x)")
    ax.set_title("Bề mặt hàm thật f(x₁, x₂)"); ax.view_init(28, -55)
    ax = fig.add_subplot(1, 2, 2)
    cs = ax.contourf(G1, G2, Z, levels=20, cmap=SEQ_CMAP)
    fig.colorbar(cs, ax=ax, label="f(x)")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.set_title("Đường đồng mức (trục chính nghiêng do hiệp phương sai âm)")
    _save(fig, "true_surface.png")

    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    for j, (col, lab) in enumerate([(0, "x₁"), (1, "x₂"), (2, "y")]):
        ax = axes[j]
        for k, (x, y) in splits.items():
            v = y if col == 2 else x[:, col]
            ax.hist(v, bins=20, density=True, histtype="step", linewidth=2, color=SPLIT_COLORS[k],
                    label=SPLIT_LABELS[k])
        ax.set_xlabel(lab); ax.set_ylabel("mật độ"); ax.grid(axis="y", **GRID)
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle("Phân bố x₁, x₂ (đều) và y (lệch phải: phần lớn mẫu nằm xa đỉnh Gaussian)", y=1.08)
    _save(fig, "data_distributions.png")


def load_split_full(name):
    return data.load_split(DATA_DIR, name)


# ---------------------------------------------------------------- tiện ích vẽ sai số theo biến
def _err_plot(ax, summ, xcol, metric="rmse", splits=("train", "val", "test"), logy=True):
    for s in splits:
        m, sd = summ[f"{s}_{metric}_mean"], summ[f"{s}_{metric}_std"].fillna(0)
        ax.errorbar(summ[xcol], m, yerr=sd, marker="o", markersize=5, linewidth=2, capsize=3,
                    color=SPLIT_COLORS[s], label=SPLIT_LABELS[s])
    if logy:
        ax.set_yscale("log")
    ax.grid(**GRID)
    ax.set_ylabel(f"{metric.upper()} (mean ± std, 3 seeds)")


def _curves(ax, group, names, label_fmt, seed=0, key="val_loss"):
    cmap = plt.get_cmap(SEQ_CMAP)
    for i, n in enumerate(names):
        p = os.path.join(RUNS_DIR, group, f"{n}_s{seed}", "log.csv")
        log = pd.read_csv(p)
        ax.plot(log.epoch, log[key] * data.Y_MAX ** 2, linewidth=1.6,
                color=cmap(0.35 + 0.65 * i / max(1, len(names) - 1)), label=label_fmt(n))
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.grid(**GRID)
    ax.set_ylabel("Validation MSE (thang gốc)")


# ---------------------------------------------------------------- tinh chỉnh
def plot_tune():
    s = pd.read_csv(os.path.join(RESULTS_DIR, "tune_summary.csv")).sort_values("val_rmse_mean")
    labels = [f"{r.activation}, lr={r.lr:g}, bs={int(r.batch_size)}" for r in s.itertuples()]
    fig, ax = plt.subplots(figsize=(8, 6))
    y = np.arange(len(s))[::-1]
    # dạng dot plot (trục log không phù hợp với cột bắt đầu từ 0)
    ax.errorbar(s.val_rmse_mean, y, xerr=s.val_rmse_std, fmt="o", color=C_VAL, markersize=6, capsize=2)
    ax.plot(s.val_rmse_mean.iloc[0], y[0], "o", markersize=10, color="#b04a1f", label="được chọn (baseline)")
    ax.legend(loc="lower right")
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xscale("log"); ax.set_xlabel("Validation RMSE (mean ± std, 3 seeds)")
    ax.set_title("Tinh chỉnh siêu tham số (A=1000, 1 tầng ẩn × 10 nơ-ron) – chọn theo validation")
    ax.grid(axis="x", **GRID)
    _save(fig, "tune_val_rmse.png")


# ---------------------------------------------------------------- A, B, C
def plot_A():
    s = pd.read_csv(os.path.join(RESULTS_DIR, "A_summary.csv"))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    _err_plot(axes[0], s, "n_train", "mse")
    _err_plot(axes[1], s, "n_train", "rmse")
    for ax in axes:
        ax.set_xlabel("Số mẫu huấn luyện A"); ax.set_xticks(s.n_train)
    axes[0].legend(); fig.suptitle("Nhóm A – sai số theo số mẫu huấn luyện (1 tầng ẩn × 10)")
    _save(fig, "A_error.png")
    fig, ax = plt.subplots(figsize=(7, 4.2))
    _curves(ax, "A", ["A200", "A600", "A1000", "A1400", "A2000"], lambda n: f"A = {n[1:]}")
    ax.legend(); ax.set_title("Nhóm A – đường cong validation loss (seed 0)")
    _save(fig, "A_curves.png")


def plot_B():
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, grp, n in [(axes[0], "B", 1000), (axes[1], "B200", 200)]:
        s = pd.read_csv(os.path.join(RESULTS_DIR, f"{grp}_summary.csv"))
        _err_plot(ax, s, "width", "rmse")
        ax.set_xscale("log"); ax.set_xticks(s.width); ax.set_xticklabels(s.width)
        ax.set_xlabel("Số nơ-ron tầng ẩn B"); ax.set_title(f"A = {n} mẫu train")
    axes[0].legend(); fig.suptitle("Nhóm B – sai số theo độ rộng (MLP 1 tầng ẩn)")
    _save(fig, "B_error.png")

    G1, G2, X = _grid_xy(81)
    Ztrue = data.gaussian2d(X).reshape(G1.shape)
    widths = [1, 2, 3, 5, 10, 50]
    fig, axes = plt.subplots(1, len(widths) + 1, figsize=(18, 3.4), sharey=True)
    lv = np.linspace(0, data.Y_MAX * 1.05, 15)
    axes[0].contourf(G1, G2, Ztrue, levels=lv, cmap=SEQ_CMAP, extend="both"); axes[0].set_title("Hàm thật")
    for ax, b in zip(axes[1:], widths):
        m, _ = load_model(os.path.join(RUNS_DIR, "B", f"B{b}_s0", "best.pt"))
        ax.contourf(G1, G2, predict(m, X).reshape(G1.shape), levels=lv, cmap=SEQ_CMAP, extend="both")
        ax.set_title(f"B = {b}")
    for ax in axes:
        ax.set_xlabel("x₁")
    axes[0].set_ylabel("x₂")
    fig.suptitle("Bề mặt xấp xỉ theo số nơ-ron B (A=1000, seed 0) – B nhỏ thiếu khớp rõ rệt", y=1.04)
    _save(fig, "B_surfaces.png")

    # Đường cong train/val của B200 (A = 200): khoảng cách train–val tăng theo B
    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
    for ax, b in zip(axes, [3, 10, 50]):
        log = pd.read_csv(os.path.join(RUNS_DIR, "B200", f"B{b}_s0", "log.csv"))
        ax.plot(log.epoch, log.train_loss * data.Y_MAX ** 2, color=C_TRAIN, linewidth=1.4, label="Train")
        ax.plot(log.epoch, log.val_loss * data.Y_MAX ** 2, color=C_VAL, linewidth=1.4, label="Validation")
        ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.grid(**GRID); ax.set_title(f"B = {b}")
    axes[0].set_ylabel("MSE (thang gốc)"); axes[0].legend()
    fig.suptitle("B200 (A = 200, seed 0): loss train / validation theo epoch", y=1.02)
    _save(fig, "B200_curves.png")


def plot_C():
    s = pd.read_csv(os.path.join(RESULTS_DIR, "C_summary.csv"))
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    _err_plot(ax, s, "depth", "rmse")
    ax.set_xlabel("Số tầng ẩn C (10 nơ-ron/tầng)"); ax.set_xticks(s.depth); ax.legend()
    ax.set_title("Nhóm C – sai số theo độ sâu")
    _save(fig, "C_error.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 3.8))
    for ax, col, lab in [(axes[0], "train_time_s", "Thời gian huấn luyện (s)"),
                         (axes[1], "time_per_epoch_ms", "Thời gian / epoch (ms)"),
                         (axes[2], "best_epoch", "Epoch tốt nhất (hội tụ)")]:
        ax.bar(s.depth, s[f"{col}_mean"], yerr=s[f"{col}_std"].fillna(0), color=C_TRAIN, width=0.6, capsize=3)
        ax.set_xlabel("Số tầng ẩn C"); ax.set_ylabel(lab); ax.set_xticks(s.depth); ax.grid(axis="y", **GRID)
    fig.suptitle("Nhóm C – chi phí huấn luyện theo độ sâu (mean ± std, 3 seeds; 1 luồng CPU / lần chạy)")
    _save(fig, "C_time.png")

    fig, ax = plt.subplots(figsize=(7, 4.2))
    _curves(ax, "C", [f"C{c}" for c in s.depth], lambda n: f"C = {n[1:]}")
    ax.legend(); ax.set_title("Nhóm C – đường cong validation loss (seed 0)")
    _save(fig, "C_curves.png")


# ---------------------------------------------------------------- mô hình cuối
def plot_final():
    s = pd.read_csv(os.path.join(RESULTS_DIR, "final_summary.csv"))
    piv = s.pivot(index="depth", columns="width", values="val_rmse_mean")
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    im = ax.imshow(piv.values, cmap=SEQ_CMAP + "_r", aspect="auto")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            v = piv.values[i, j]
            ax.text(j, i, f"{v:.2e}", ha="center", va="center", fontsize=9,
                    color="white" if v < np.median(piv.values) else "#0b0b0b")
    ax.set_xticks(range(piv.shape[1])); ax.set_xticklabels(piv.columns)
    ax.set_yticks(range(piv.shape[0])); ax.set_yticklabels(piv.index)
    ax.set_xlabel("Số nơ-ron mỗi tầng"); ax.set_ylabel("Số tầng ẩn")
    ax.set_title("Tinh chỉnh cuối (A=2000): validation RMSE")
    fig.colorbar(im, ax=ax)
    _save(fig, "final_grid.png")

    sel = json.load(open(os.path.join(RESULTS_DIR, "final_selection.json")))
    model, ck = load_model(os.path.join(MODELS_DIR, "final_model.pt"))
    arch = "-".join(map(str, ck["config"]["hidden_sizes"]))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    for ax, name, key in [(axes[0], "train_pool", "train"), (axes[1], "valid", "val"), (axes[2], "test", "test")]:
        x, y = data.load_split(DATA_DIR, name)
        yh = predict(model, x)
        ax.scatter(y, yh, s=8, color=SPLIT_COLORS[key], alpha=0.7)
        lim = [min(y.min(), yh.min()) - 0.005, max(y.max(), yh.max()) + 0.005]
        ax.plot(lim, lim, color="#52514e", linewidth=1, linestyle="--", label="y = ŷ")
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        ax.set_xlabel("y thật"); ax.set_ylabel("ŷ dự đoán")
        ax.set_title(f"{SPLIT_LABELS[key]}: RMSE = {sel[key + '_rmse']:.2e}, R² = {sel[key + '_r2']:.6f}")
        ax.grid(**GRID)
    axes[0].legend(loc="upper left")
    fig.suptitle(f"Mô hình cuối (tầng ẩn {arch}): y thật – ŷ dự đoán", y=1.02)
    _save(fig, "final_y_vs_yhat.png")

    G1, G2, X = _grid_xy()
    Z = data.gaussian2d(X).reshape(G1.shape)
    Zh = predict(model, X).reshape(G1.shape)
    xt, yt = data.load_split(DATA_DIR, "test")
    fig = plt.figure(figsize=(16, 4.6))
    for i, (zz, t) in enumerate([(Z, "Bề mặt hàm thật"), (Zh, "Bề mặt xấp xỉ bởi MLP")]):
        ax = fig.add_subplot(1, 3, i + 1, projection="3d")
        ax.plot_surface(G1, G2, zz, cmap=SEQ_CMAP, linewidth=0, alpha=0.9)
        if i == 1:
            ax.scatter(xt[:, 0], xt[:, 1], yt, s=6, color=C_TEST, label="test (y thật)")
            ax.legend(loc="upper left")
        ax.set_title(t); ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.set_zlabel("y"); ax.view_init(28, -55)
    ax = fig.add_subplot(1, 3, 3)
    err = Zh - Z
    v = np.abs(err).max()
    cs = ax.contourf(G1, G2, err, levels=np.linspace(-v, v, 21), cmap="RdBu_r")
    fig.colorbar(cs, ax=ax, label="ŷ − y", format="%.1e")
    ax.set_xlabel("x₁"); ax.set_ylabel("x₂"); ax.set_title("Sai số ŷ − y trên toàn miền")
    _save(fig, "final_surface.png")

    log = pd.read_csv(os.path.join(ROOT, sel["run_dir"], "log.csv"))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(log.epoch, log.train_loss * data.Y_MAX ** 2, color=C_TRAIN, linewidth=1.6, label="Train")
    ax.plot(log.epoch, log.val_loss * data.Y_MAX ** 2, color=C_VAL, linewidth=1.6, label="Validation")
    ax.axvline(ck["best_epoch"], color="#52514e", linestyle="--", linewidth=1, label=f"best epoch = {ck['best_epoch']}")
    ax.set_yscale("log"); ax.set_xlabel("epoch"); ax.set_ylabel("MSE (thang gốc)"); ax.grid(**GRID); ax.legend()
    ax.set_title("Mô hình cuối – loss theo epoch")
    _save(fig, "final_loss.png")


def make_all():
    plot_data()
    plot_formula()
    plot_architecture((10,), "mlp_architecture.png", "MLP 1 tầng ẩn (B nơ-ron; baseline B = 10)")
    plot_architecture((10, 10, 10), "mlp_deep.png", "MLP C tầng ẩn (ví dụ C = 3, 10 nơ-ron/tầng)")
    for name, fn in [("tune", plot_tune), ("A", plot_A), ("B", plot_B), ("C", plot_C), ("final", plot_final)]:
        if os.path.exists(os.path.join(RESULTS_DIR, f"{name}_summary.csv")):
            fn()
    print("figures in", FIG_DIR)


def plot_architecture(hidden=(10,), fname="mlp_architecture.png", title=None):
    """Sơ đồ MLP: 2 đầu vào → các tầng ẩn → 1 đầu ra (hiển thị tối đa 6 nút mỗi tầng)."""
    sizes = [2, *hidden, 1]
    fig, ax = plt.subplots(figsize=(2.2 * len(sizes) + 1, 4.4))
    pos, dx = [], 2.6
    for li, n in enumerate(sizes):
        shown = min(n, 6)
        ys = np.linspace(-(shown - 1) / 2, (shown - 1) / 2, shown)
        pos.append([(li * dx, y) for y in ys])
        for (x, y) in pos[-1]:
            color = C_TRAIN if li == 0 else (C_TEST if li == len(sizes) - 1 else "#ffffff")
            ax.add_patch(plt.Circle((x, y), 0.18, facecolor=color, edgecolor="#0b0b0b", linewidth=1.2, zorder=3))
        if shown < n:
            ax.text(li * dx, ys[0] - 0.75, f"… ({n} nơ-ron)", ha="center", fontsize=9, color="#52514e")
        name = "Đầu vào\n(x₁, x₂)" if li == 0 else ("Đầu ra ŷ\n(tuyến tính)" if li == len(sizes) - 1
                                                    else f"Tầng ẩn {li}\nLinear + σ")
        ax.text(li * dx, 3.0 if li not in (0, len(sizes) - 1) else 1.3, name, ha="center", va="bottom", fontsize=9)
    for a, b in zip(pos[:-1], pos[1:]):
        for (x1, y1) in a:
            for (x2, y2) in b:
                ax.plot([x1, x2], [y1, y2], color="#b5b4af", linewidth=0.6, zorder=1)
    ax.set_xlim(-1.2, (len(sizes) - 1) * dx + 1.2); ax.set_ylim(-3.6, 4.4); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(title or f"MLP {'-'.join(map(str, sizes))}")
    _save(fig, fname)


def plot_formula():
    """Ảnh công thức Gaussian cho slide (công thức OMML trong .pptx hiển thị không ổn định khi xuất PDF).
    Biên dịch bằng xelatex (standalone) rồi chuyển sang PNG bằng pdftoppm; bỏ qua nếu thiếu công cụ."""
    import shutil
    import subprocess
    import tempfile
    if not (shutil.which("xelatex") and shutil.which("pdftoppm")):
        return
    tex = r"""\documentclass[border=6pt]{standalone}
\usepackage{amsmath,bm}
\begin{document}
$\displaystyle f(\mathbf{x})=\frac{1}{2\pi\sqrt{\det C}}\exp\!\Big(-\frac{1}{2}(\mathbf{x}-\bm{\mu})^{T}
C^{-1}(\mathbf{x}-\bm{\mu})\Big),\qquad \bm{\mu}=\begin{pmatrix}0\\0\end{pmatrix},\quad
C=\begin{pmatrix}0.3&-0.5\\-0.5&2\end{pmatrix}$
\end{document}
"""
    with tempfile.TemporaryDirectory() as d:
        open(os.path.join(d, "f.tex"), "w").write(tex)
        subprocess.run(["xelatex", "-interaction=batchmode", "f.tex"], cwd=d, check=True, stdout=subprocess.DEVNULL)
        os.makedirs(FIG_DIR, exist_ok=True)
        subprocess.run(["pdftoppm", "-r", "300", "-png", "-singlefile", os.path.join(d, "f.pdf"),
                        os.path.join(FIG_DIR, "formula")], check=True)
