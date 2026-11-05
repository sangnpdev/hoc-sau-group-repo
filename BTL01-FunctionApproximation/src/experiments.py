"""Định nghĩa và chạy các nhóm thực nghiệm: tune (tinh chỉnh siêu tham số), A, B, B200, C, final.

Mỗi lần chạy lưu ở runs/<nhóm>/<tên>/ (config.json, log.csv, best.pt, metrics.json).
Kết quả gộp: results/<nhóm>_runs.csv (từng lần chạy) và results/<nhóm>_summary.csv (mean ± std theo seed).
"""
import itertools
import json
import os
import shutil
from concurrent.futures import ProcessPoolExecutor

import pandas as pd
import torch

from src.train import train_run

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(ROOT, "data")
RUNS_DIR = os.path.join(ROOT, "runs")
RESULTS_DIR = os.path.join(ROOT, "results")
MODELS_DIR = os.path.join(ROOT, "models")
BASELINE_PATH = os.path.join(RESULTS_DIR, "baseline.json")

SEEDS = [0, 1, 2]
BASE_N_TRAIN = 1000
BASE_WIDTH = 10

# Lưới tinh chỉnh siêu tham số "phụ" (không phải biến khảo sát A/B/C), chọn theo val MSE.
TUNE_GRID = {"lr": [1e-3, 3e-3, 1e-2, 3e-2], "activation": ["tanh", "relu", "sigmoid"], "batch_size": [32, 128]}
A_VALUES = [200, 400, 600, 800, 1000, 1200, 1400, 1600, 1800, 2000]
B_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 30, 50]
C_VALUES = [1, 2, 3, 4, 5]
# Tinh chỉnh cuối: dựa trên kết quả C (sâu ≥ 2 tốt hơn hẳn, C = 4–5 chạm trần 3000 epoch) → xét độ sâu 2–4
# và nâng trần epoch; dùng toàn bộ 2000 mẫu train (kết quả A: thêm dữ liệu không làm xấu đi).
FINAL_GRID = {"width": [10, 20, 40], "depth": [2, 3, 4]}
FINAL_MAX_EPOCHS = 6000

# Cột gom nhóm khi tính mean ± std qua các seed
GROUP_KEYS = {"tune": ["lr", "activation", "batch_size"], "A": ["n_train"], "B": ["width"],
              "B200": ["width"], "C": ["depth"], "final": ["width", "depth"]}


def load_baseline():
    with open(BASELINE_PATH) as f:
        return json.load(f)


def base_cfg():
    b = load_baseline()
    return {"lr": b["lr"], "activation": b["activation"], "batch_size": b["batch_size"],
            "n_train": BASE_N_TRAIN, "hidden_sizes": [BASE_WIDTH]}


def build_configs(group):
    cfgs = []
    if group == "tune":
        for lr, act, bs in itertools.product(*TUNE_GRID.values()):
            cfgs.append({"name": f"lr{lr}_{act}_bs{bs}", "lr": lr, "activation": act, "batch_size": bs,
                         "n_train": BASE_N_TRAIN, "hidden_sizes": [BASE_WIDTH]})
    elif group == "A":
        cfgs = [{**base_cfg(), "name": f"A{a}", "n_train": a} for a in A_VALUES]
    elif group in ("B", "B200"):
        n = BASE_N_TRAIN if group == "B" else 200
        cfgs = [{**base_cfg(), "name": f"B{b}", "n_train": n, "hidden_sizes": [b]} for b in B_VALUES]
    elif group == "C":
        cfgs = [{**base_cfg(), "name": f"C{c}", "hidden_sizes": [BASE_WIDTH] * c} for c in C_VALUES]
    elif group == "final":
        for w, d in itertools.product(FINAL_GRID["width"], FINAL_GRID["depth"]):
            cfgs.append({**base_cfg(), "name": f"W{w}_D{d}", "n_train": 2000, "hidden_sizes": [w] * d,
                         "max_epochs": FINAL_MAX_EPOCHS})
    else:
        raise ValueError(group)
    return [{**c, "seed": s} for c in cfgs for s in SEEDS]


def _init_worker():
    torch.set_num_threads(1)   # mỗi tiến trình 1 luồng → không tranh chấp CPU, thời gian so sánh được


def _job(args):
    cfg, group = args
    out_dir = os.path.join(RUNS_DIR, group, f"{cfg['name']}_s{cfg['seed']}")
    return train_run(cfg, DATA_DIR, out_dir)


def run_group(group, workers=32):
    cfgs = build_configs(group)
    print(f"[{group}] {len(cfgs)} runs, {workers} workers", flush=True)
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as ex:
        results = list(ex.map(_job, [(c, group) for c in cfgs]))
    df = pd.DataFrame(results)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    df.to_csv(os.path.join(RESULTS_DIR, f"{group}_runs.csv"), index=False)
    summ = summarize(df, GROUP_KEYS[group])
    summ.to_csv(os.path.join(RESULTS_DIR, f"{group}_summary.csv"), index=False)
    return df, summ


def summarize(df, keys):
    cols = [c for c in df.columns if c.split("_")[0] in ("train", "val", "test")
            and c.split("_")[-1] in ("mse", "rmse", "mae", "r2")]
    cols += ["n_params", "best_epoch", "epochs_run", "train_time_s", "time_per_epoch_ms", "n_steps"]
    agg = df.groupby(keys)[cols].agg(["mean", "std"])
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    return agg.reset_index()


def select_baseline():
    """Chọn lr/activation/batch_size có val MSE trung bình nhỏ nhất (KHÔNG dùng test)."""
    summ = pd.read_csv(os.path.join(RESULTS_DIR, "tune_summary.csv"))
    best = summ.sort_values("val_mse_mean").iloc[0]
    baseline = {"lr": float(best["lr"]), "activation": best["activation"], "batch_size": int(best["batch_size"]),
                "n_train": BASE_N_TRAIN, "hidden_sizes": [BASE_WIDTH], "seeds": SEEDS,
                "selected_by": "min mean val_mse over seeds", "val_mse_mean": float(best["val_mse_mean"])}
    with open(BASELINE_PATH, "w") as f:
        json.dump(baseline, f, indent=2)
    print("baseline:", baseline)
    return baseline


def select_final():
    """Chọn kiến trúc cuối theo val MSE trung bình; lấy seed có val MSE nhỏ nhất làm mô hình nộp."""
    runs = pd.read_csv(os.path.join(RESULTS_DIR, "final_runs.csv"))
    summ = pd.read_csv(os.path.join(RESULTS_DIR, "final_summary.csv"))
    best = summ.sort_values("val_mse_mean").iloc[0]
    cand = runs[(runs.width == best.width) & (runs.depth == best.depth)].sort_values("val_mse")
    run = cand.iloc[0]
    run_dir = os.path.join(RUNS_DIR, "final", f"{run['name']}_s{run['seed']}")
    os.makedirs(MODELS_DIR, exist_ok=True)
    shutil.copy(os.path.join(run_dir, "best.pt"), os.path.join(MODELS_DIR, "final_model.pt"))
    info = {k: (v.item() if hasattr(v, "item") else v) for k, v in run.to_dict().items()}
    info["run_dir"] = os.path.relpath(run_dir, ROOT)
    with open(os.path.join(RESULTS_DIR, "final_selection.json"), "w") as f:
        json.dump(info, f, indent=2)
    print("final model:", info)
    return info
