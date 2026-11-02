"""
src/data_viz.py – Trực quan hoá và thống kê mô tả dữ liệu (mục 3.1).

Các hàm (đầu vào là DataFrame all_df có cột split, x1, x2, y):
    - plot_scatter3d_surface(): Hình 1 – scatter 3D train/valid/test + bề mặt hàm thật.
    - plot_overlay()          : Hình 2 – mẫu train chồng lên bề mặt thật.
    - plot_distributions()    : Hình 3 – histogram x1, x2, y + contour f và vị trí mẫu.
    - describe_splits()       : bảng thống kê mô tả theo từng tập và toàn bộ.

Quy ước màu: mỗi tập luôn có một màu cố định ở mọi hình
(train = xanh dương, valid = cam, test = xanh ngọc).
"""
import matplotlib
matplotlib.use("Agg")  # backend không cần màn hình → chạy được trên server/terminal

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from src.data import X1_RANGE, X2_RANGE, gaussian_2d

# Màu cố định cho từng tập (bảng màu phân loại an toàn với người mù màu).
SPLIT_COLORS = {"train": "#2a78d6", "valid": "#eb6834", "test": "#1baf7a"}
# Thang màu tuần tự một sắc độ (xanh nhạt → xanh đậm) cho độ lớn của f.
SEQ_CMAP = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#184f95", "#0d366b"]
)
HIST_COLOR = "#5f5e5a"   # màu trung tính cho histogram toàn bộ dữ liệu (không gắn với tập nào)
VIEW = dict(elev=25, azim=-60)   # góc nhìn chung cho mọi hình 3D


def true_grid(n: int = 120) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Lưới n×n phủ toàn miền và giá trị f trên lưới (để vẽ bề mặt / contour)."""
    G1, G2 = np.meshgrid(np.linspace(*X1_RANGE, n), np.linspace(*X2_RANGE, n))
    Z = gaussian_2d(np.column_stack([G1.ravel(), G2.ravel()])).reshape(G1.shape)
    return G1, G2, Z


def _label_3d(ax, zlabel: str = "y") -> None:
    ax.set_xlabel("$x_1$")
    ax.set_ylabel("$x_2$")
    ax.set_zlabel(zlabel)
    ax.view_init(**VIEW)


def plot_scatter3d_surface(all_df: pd.DataFrame, save_path: str) -> None:
    """Hình 1. Trái: scatter 3D dữ liệu theo tập. Phải: bề mặt hàm Gaussian thật."""
    G1, G2, Z = true_grid()
    fig = plt.figure(figsize=(16, 6.5))

    ax = fig.add_subplot(1, 2, 1, projection="3d")
    for name, color in SPLIT_COLORS.items():
        d = all_df[all_df.split == name]
        ax.scatter(d.x1, d.x2, d.y, s=8 if name == "train" else 14,
                   color=color, alpha=0.6, linewidths=0, label=f"{name} ({len(d)})")
    _label_3d(ax)
    ax.set_title("Scatter 3D dữ liệu train / valid / test")
    ax.legend(loc="upper left")

    ax = fig.add_subplot(1, 2, 2, projection="3d")
    ax.plot_surface(G1, G2, Z, cmap=SEQ_CMAP, alpha=0.9, linewidth=0)
    _label_3d(ax, "f")
    ax.set_title("Bề mặt hàm Gaussian thật $f(x_1, x_2)$")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_overlay(all_df: pd.DataFrame, save_path: str) -> None:
    """Hình 2. Mẫu train chồng lên bề mặt thật – kiểm tra trực quan nhãn y nằm đúng trên f."""
    G1, G2, Z = true_grid()
    fig = plt.figure(figsize=(8, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(G1, G2, Z, cmap=SEQ_CMAP, alpha=0.35, linewidth=0)
    tr = all_df[all_df.split == "train"]
    ax.scatter(tr.x1, tr.x2, tr.y, s=6, color="#1a1a19", alpha=0.7, linewidths=0)
    _label_3d(ax)
    ax.set_title("Mẫu train nằm trên bề mặt thật")
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def plot_distributions(all_df: pd.DataFrame, save_path: str) -> None:
    """Hình 3. Phân bố x1, x2, y (toàn bộ dữ liệu) + contour f với vị trí mẫu theo tập.

    Kỳ vọng: x1, x2 gần đều (uniform); y lệch phải mạnh (đa số điểm xa đỉnh nên y ≈ 0).
    """
    G1, G2, Z = true_grid()
    fig, axs = plt.subplots(1, 4, figsize=(19, 4.3))

    hist_kw = dict(bins=30, color=HIST_COLOR, edgecolor="white", linewidth=1)
    axs[0].hist(all_df.x1, **hist_kw)
    axs[0].set_title("Phân bố $x_1$ (đều trên [-2, 2])")
    axs[0].set_xlabel("$x_1$")
    axs[1].hist(all_df.x2, **hist_kw)
    axs[1].set_title("Phân bố $x_2$ (đều trên [-4, 4])")
    axs[1].set_xlabel("$x_2$")
    axs[2].hist(all_df.y, **hist_kw)
    axs[2].set_title("Phân bố nhãn y (lệch phải, nhiều giá trị ≈ 0)")
    axs[2].set_xlabel("y")
    for ax in axs[:3]:
        ax.set_ylabel("Số mẫu")
        ax.grid(axis="y", alpha=0.3)
        ax.set_axisbelow(True)

    # Contour f + vị trí mẫu: cho thấy 3 tập cùng phủ đều miền
    cs = axs[3].contourf(G1, G2, Z, levels=20, cmap=SEQ_CMAP)
    for name, color in SPLIT_COLORS.items():
        d = all_df[all_df.split == name]
        # viền trắng mảnh để điểm không chìm vào nền contour cùng tông xanh
        axs[3].scatter(d.x1, d.x2, s=8, color=color, edgecolors="white", linewidths=0.4, label=name)
    axs[3].set_xlabel("$x_1$")
    axs[3].set_ylabel("$x_2$")
    axs[3].set_title("Contour f và vị trí mẫu")
    axs[3].legend(markerscale=3)
    fig.colorbar(cs, ax=axs[3], label="f")

    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)


def describe_splits(all_df: pd.DataFrame) -> pd.DataFrame:
    """Bảng thống kê mô tả x1, x2, y cho từng tập và cho toàn bộ ("all").

    Mỗi dòng = (tập, biến) với cột: count, mean, std, min, q25, median, q75, max.
    """
    groups = [(name, all_df[all_df.split == name]) for name in all_df.split.unique()]
    groups.append(("all", all_df))
    rows = []
    for name, d in groups:
        for var in ("x1", "x2", "y"):
            v = d[var].to_numpy()
            rows.append({
                "split": name, "variable": var, "count": len(v),
                "mean": v.mean(), "std": v.std(ddof=1), "min": v.min(),
                "q25": np.quantile(v, 0.25), "median": np.median(v),
                "q75": np.quantile(v, 0.75), "max": v.max(),
            })
    return pd.DataFrame(rows)
