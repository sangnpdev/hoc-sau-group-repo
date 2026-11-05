"""Sinh bảng Markdown (results/tables/*.md) từ các file *_summary.csv – không gõ số liệu bằng tay."""
import json
import os

import pandas as pd

from src.experiments import RESULTS_DIR

TABLE_DIR = os.path.join(RESULTS_DIR, "tables")


def pm(mean, std, fmt):
    std = 0.0 if pd.isna(std) else std
    return f"{mean:{fmt}} ± {std:{fmt}}"


def metric_table(summ, key_cols, key_names, extra=()):
    """RMSE/MAE ghi theo đơn vị 10⁻³ (mean ± std qua seed) cho gọn; MSE và R² ghi giá trị trung bình.
    extra: các cột phụ (tên cột gốc, tiêu đề, định dạng)."""
    head = list(key_names) + ["Train RMSE ×10⁻³", "Val RMSE ×10⁻³", "Test RMSE ×10⁻³", "Test MAE ×10⁻³",
                              "Test MSE", "Test R²"]
    head += [h for _, h, _ in extra]
    rows = []
    for _, r in summ.iterrows():
        row = [f"{r[k]:g}" for k in key_cols]
        row += [pm(1e3 * r[f"{s}_rmse_mean"], 1e3 * r[f"{s}_rmse_std"], ".2f") for s in ("train", "val", "test")]
        row += [pm(1e3 * r["test_mae_mean"], 1e3 * r["test_mae_std"], ".2f"), f"{r['test_mse_mean']:.2e}",
                f"{r['test_r2_mean']:.5f}"]
        row += [f"{r[f'{c}_mean']:{f}}" for c, _, f in extra]
        rows.append(row)
    return to_md(head, rows)


def to_md(head, rows):
    # số dấu '-' tỉ lệ với độ dài nội dung → pandoc chia độ rộng cột hợp lý khi bảng rộng
    widths = [max(len(head[j]) // 2, *(len(r[j]) for r in rows), 3) for j in range(len(head))]
    lines = ["| " + " | ".join(head) + " |", "|" + "|".join("-" * w for w in widths) + "|"]
    lines += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(lines) + "\n"


def write(name, text):
    os.makedirs(TABLE_DIR, exist_ok=True)
    with open(os.path.join(TABLE_DIR, name), "w") as f:
        f.write(text)


def read(name):
    return pd.read_csv(os.path.join(RESULTS_DIR, f"{name}_summary.csv"))


def data_stats():
    from src.data import load_split
    from src.experiments import DATA_DIR
    rows = []
    for name, lab in [("train_pool", "Train pool"), ("valid", "Validation"), ("test", "Test")]:
        x, y = load_split(DATA_DIR, name)
        for col, v in [("x₁", x[:, 0]), ("x₂", x[:, 1]), ("y", y)]:
            rows.append([lab, col, str(len(v)), f"{v.min():.4g}", f"{v.max():.4g}", f"{v.mean():.4g}",
                         f"{v.std():.4g}", f"{pd.Series(v).median():.4g}"])
    write("data_stats.md", to_md(["Tập", "Biến", "N", "Min", "Max", "Mean", "Std", "Median"], rows))


def setup_table():
    from src.experiments import load_baseline, SEEDS, TUNE_GRID
    from src.train import DEFAULTS
    b = load_baseline()
    grid = "; ".join(f"{k}: {{{', '.join(map(str, v))}}}" for k, v in TUNE_GRID.items())
    rows = [
        ["Optimizer", "Adam (β₁ = 0.9, β₂ = 0.999)"],
        ["Learning rate", f"{b['lr']:g} (chọn bằng tinh chỉnh; lưới: {grid})"],
        ["Hàm kích hoạt tầng ẩn", f"{b['activation']} (chọn bằng tinh chỉnh)"],
        ["Batch size", f"{b['batch_size']} (chọn bằng tinh chỉnh)"],
        ["Đầu ra", "1 nơ-ron tuyến tính"],
        ["Kiến trúc baseline", f"1 tầng ẩn × {b['hidden_sizes'][0]} nơ-ron, A = {b['n_train']} mẫu train"],
        ["Hàm mất mát", "MSELoss (trên y chuẩn hoá y / f_max)"],
        ["Số epoch tối đa", str(DEFAULTS["max_epochs"])],
        ["Điều kiện dừng", f"early stopping: val loss không giảm ≥ {DEFAULTS['min_rel_improve']:g} (tương đối) "
                           f"sau {DEFAULTS['patience']} epoch"],
        ["Lịch learning rate", f"ReduceLROnPlateau: ×{DEFAULTS['plateau_factor']} sau {DEFAULTS['plateau_patience']} "
                               f"epoch không cải thiện, lr tối thiểu {DEFAULTS['min_lr']:g}"],
        ["Checkpoint", "trọng số có val loss nhỏ nhất (best.pt)"],
        ["Khởi tạo trọng số", "mặc định của nn.Linear (Kaiming-uniform)"],
        ["Seeds", f"dữ liệu: 2024; huấn luyện: {', '.join(map(str, SEEDS))}"],
    ]
    write("setup.md", to_md(["Thành phần", "Giá trị"], rows))


def make_all():
    data_stats()
    setup_table()
    s = read("tune").sort_values("val_mse_mean")
    rows = [[r.activation, f"{r.lr:g}", str(int(r.batch_size)), pm(r.val_mse_mean, r.val_mse_std, ".2e"),
             pm(r.val_rmse_mean, r.val_rmse_std, ".2e"), pm(r.best_epoch_mean, r.best_epoch_std, ".0f")]
            for r in s.itertuples()]
    write("tune.md", to_md(["Activation", "lr", "Batch", "Val MSE", "Val RMSE", "Best epoch"], rows))

    epochs = [("best_epoch", "Best epoch", ".0f"), ("n_steps", "Số bước cập nhật", ".0f")]
    write("A.md", metric_table(read("A"), ["n_train"], ["A"], epochs))
    write("B.md", metric_table(read("B"), ["width"], ["B"], [("n_params", "Số tham số", ".0f")]))
    write("B200.md", metric_table(read("B200"), ["width"], ["B"], [("n_params", "Số tham số", ".0f")]))
    write("C.md", metric_table(read("C"), ["depth"], ["C"], [("n_params", "Số tham số", ".0f")]))
    c = read("C")
    rows = [[str(r.depth), f"{r.n_params_mean:.0f}", pm(r.best_epoch_mean, r.best_epoch_std, ".0f"),
             pm(r.epochs_run_mean, r.epochs_run_std, ".0f"), pm(r.time_per_epoch_ms_mean, r.time_per_epoch_ms_std, ".1f"),
             pm(r.train_time_s_mean, r.train_time_s_std, ".1f")] for r in c.itertuples()]
    write("C_time.md", to_md(["C", "Số tham số", "Best epoch", "Số epoch đã chạy", "ms / epoch", "Tổng thời gian (s)"], rows))

    f = read("final").sort_values("val_mse_mean")
    rows = [[str(r.width), str(r.depth), f"{r.n_params_mean:.0f}", pm(r.val_mse_mean, r.val_mse_std, ".2e"),
             pm(r.val_rmse_mean, r.val_rmse_std, ".2e")] for r in f.itertuples()]
    write("final_grid.md", to_md(["Nơ-ron/tầng", "Số tầng", "Số tham số", "Val MSE", "Val RMSE"], rows))
    sel = json.load(open(os.path.join(RESULTS_DIR, "final_selection.json")))
    rows = [[sp, f"{sel[sp + '_mse']:.3e}", f"{sel[sp + '_rmse']:.3e}", f"{sel[sp + '_mae']:.3e}", f"{sel[sp + '_r2']:.7f}"]
            for sp in ("train", "val", "test")]
    write("final_model.md", to_md(["Tập", "MSE", "RMSE", "MAE", "R²"], rows))
    print("tables in", TABLE_DIR)
