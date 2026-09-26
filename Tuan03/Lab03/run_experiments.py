from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline

from house_price_pipeline import (
    ExperimentSpec,
    TARGET,
    cross_validate_pipeline,
    fit_full_sklearn,
    kaggle_log_rmse,
    load_competition_data,
    make_feature_frame,
    make_preprocessor,
    paper_outlier_mask,
    save_feature_importance,
    save_pipeline,
    write_metadata,
)
from pytorch_mlp import TorchMLPRegressor

# File này điều phối toàn bộ thí nghiệm, chọn mô hình tốt nhất và tạo submission.

def mlp_cv(train: pd.DataFrame, folds: int, random_state: int, quick: bool):
    # Cross-validation riêng cho MLP PyTorch nhưng vẫn dùng tiền xử lý của sklearn.
    x = make_feature_frame(train, "engineered")
    y = np.log1p(train[TARGET].astype(float).to_numpy())
    kf = KFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(x), start=1):
        estimator = Pipeline(
            [
                ("preprocess", make_preprocessor(x.iloc[tr_idx])),
                (
                    "model",
                    TorchMLPRegressor(
                        max_epochs=90 if quick else 280,
                        patience=14 if quick else 28,
                        random_state=random_state + fold,
                    ),
                ),
            ]
        )
        estimator.fit(x.iloc[tr_idx], y[tr_idx])
        pred = estimator.predict(x.iloc[va_idx])
        score = kaggle_log_rmse(y[va_idx], pred)
        print(f"MLP fold {fold}/{folds}: log-RMSE={score:.6f}")
        scores.append(score)

    return {
        "experiment": "engineered_mlp",
        "feature_profile": "engineered",
        "model": "pytorch_mlp",
        "remove_outliers": False,
        "rows_used": int(len(train)),
        "rows_removed": 0,
        "cv_folds": int(folds),
        "log_rmse_mean": float(np.mean(scores)),
        "log_rmse_std": float(np.std(scores, ddof=1)) if len(scores) > 1 else 0.0,
        "fold_scores": [float(s) for s in scores],
    }


def fit_full_mlp(train: pd.DataFrame, test: pd.DataFrame, random_state: int, quick: bool):
    # Huấn luyện lại MLP tốt nhất trên toàn bộ tập train trước khi dự đoán test.
    x_train = make_feature_frame(train, "engineered")
    x_test = make_feature_frame(test, "engineered")
    y = np.log1p(train[TARGET].astype(float).to_numpy())
    pipe = Pipeline(
        [
            ("preprocess", make_preprocessor(x_train)),
            (
                "model",
                TorchMLPRegressor(
                    max_epochs=110 if quick else 340,
                    patience=18 if quick else 35,
                    random_state=random_state,
                ),
            ),
        ]
    )
    pipe.fit(x_train, y)
    pred = np.expm1(pipe.predict(x_test)).clip(min=0)
    return pipe, pred


def main():
    # Các tham số dòng lệnh cho phép đổi dữ liệu, số fold, seed và chế độ chạy nhanh.
    p = argparse.ArgumentParser(description="Kaggle House Prices: sklearn + PyTorch MLP experiments")
    p.add_argument("--data-dir", default="data")
    p.add_argument("--output-dir", default="outputs")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--mlp-folds", type=int, default=3)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--skip-mlp", action="store_true")
    p.add_argument("--quick", action="store_true", help="Faster settings for a local smoke run")
    args = p.parse_args()

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    train, test = load_competition_data(args.data_dir)
    # Sau bước này, train có SalePrice còn test chỉ chứa các biến đầu vào.
    print(f"train shape={train.shape}; test shape={test.shape}")

    specs = [
        # Mỗi ExperimentSpec là một cấu hình đặc trưng + mô hình cần đánh giá.
        ExperimentSpec("paper_core_ridge", "paper_core", "ridge", False),
        ExperimentSpec("raw_ridge", "raw", "ridge", False),
        ExperimentSpec("engineered_ridge", "engineered", "ridge", False),
        ExperimentSpec("engineered_random_forest", "engineered", "random_forest", False),
        ExperimentSpec("engineered_extra_trees", "engineered", "extra_trees", False),
        ExperimentSpec("engineered_gradient_boosting", "engineered", "gradient_boosting", False),
        # Kiểm chứng riêng quy tắc loại nhà có GrLivArea > 4000 thay vì mặc định áp dụng.
        ExperimentSpec("engineered_gbr_remove_gt4000", "engineered", "gradient_boosting", True),
    ]

    results = []
    for spec in specs:
        print(f"\n=== {spec.name} ===")
        result = cross_validate_pipeline(
            train,
            spec,
            n_splits=args.folds,
            random_state=args.seed,
            quick=args.quick,
        )
        print(
            f"mean={result['log_rmse_mean']:.6f} +/- {result['log_rmse_std']:.6f}; "
            f"folds={result['fold_scores']}"
        )
        results.append(result)

    if not args.skip_mlp:
        print("\n=== engineered_mlp ===")
        results.append(mlp_cv(train, args.mlp_folds, args.seed, args.quick))

    results_df = pd.DataFrame(results).sort_values("log_rmse_mean").reset_index(drop=True)
    # Điểm log-RMSE càng thấp thì mô hình càng tốt.
    results_df.to_csv(out / "cv_results.csv", index=False)
    print("\nCV ranking (lower is better):")
    print(results_df[["experiment", "log_rmse_mean", "log_rmse_std"]].to_string(index=False))

    # Lưu danh sách độ quan trọng của đặc trưng sau biến đổi để phân tích/báo cáo.
    importance = save_feature_importance(train, out / "feature_importance.csv", args.seed, args.quick)
    print("\nTop transformed features from ExtraTrees:")
    print(importance.head(20).to_string(index=False))

    best = results_df.iloc[0].to_dict()
    # Chọn cấu hình đứng đầu rồi fit lại trên toàn bộ dữ liệu train.
    if best["model"] == "pytorch_mlp":
        best_pipe, pred = fit_full_mlp(train, test, args.seed, args.quick)
        model_path = out / "best_model_mlp.joblib"
    else:
        best_pipe, pred = fit_full_sklearn(
            train,
            test,
            feature_profile=best["feature_profile"],
            model_name=best["model"],
            remove_outliers=bool(best["remove_outliers"]),
            random_state=args.seed,
            quick=args.quick,
        )
        model_path = out / "best_model.joblib"
    save_pipeline(best_pipe, model_path)

    submission = pd.DataFrame({"Id": test["Id"].astype(int), "SalePrice": pred})
    # File submission phải giữ đúng hai cột Id và SalePrice theo định dạng Kaggle.
    submission.to_csv(out / "submission.csv", index=False)

    metadata = {
        "best_experiment": best["experiment"],
        "best_cv_log_rmse": float(best["log_rmse_mean"]),
        "best_cv_std": float(best["log_rmse_std"]),
        "metric": "RMSE on log1p(SalePrice), equivalent to Kaggle log-RMSE convention",
        "train_shape": list(train.shape),
        "test_shape": list(test.shape),
        "paper_outliers_gt4000_count": int((~paper_outlier_mask(train)).sum()),
    }
    write_metadata(out / "run_metadata.json", metadata)
    print(f"\nSaved results to {out.resolve()}")
    print("Submission columns:", submission.columns.tolist(), "rows:", len(submission))


if __name__ == "__main__":
    main()
