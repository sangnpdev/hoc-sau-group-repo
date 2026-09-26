from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Tên cột mục tiêu và cột mã dòng được dùng thống nhất trong toàn bộ pipeline.

TARGET = "SalePrice"
ID_COL = "Id"

# Bộ đặc trưng rút gọn, dùng để tái hiện nhóm thí nghiệm đơn giản trong tài liệu Ames.
PAPER_CORE_FEATURES = [
    "Neighborhood",
    "LotArea",
    "TotalBsmtSF",
    "GrLivArea",
    "GarageCars",
    "Fireplaces",
]

# Hai cột này là số nhưng về ý nghĩa lại là mã phân loại, nên sẽ được xử lý như category.
CODE_AS_CATEGORY = ["MSSubClass", "MoSold"]


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    feature_profile: str
    model_name: str
    remove_outliers: bool = False


def kaggle_log_rmse(y_true_log: np.ndarray, y_pred_log: np.ndarray) -> float:
    """Tính log-RMSE theo cách Kaggle dùng khi mô hình dự đoán log1p(SalePrice)."""
    return float(math.sqrt(mean_squared_error(y_true_log, y_pred_log)))


def load_competition_data(data_dir: str | Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    # Đọc hai file dữ liệu và kiểm tra nhanh cấu trúc trước khi chạy mô hình.
    data_dir = Path(data_dir)
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    missing = [str(p) for p in (train_path, test_path) if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing Kaggle data files: "
            + ", ".join(missing)
            + ". Download train.csv and test.csv from the competition Data page and place them in the data directory."
        )

    train = pd.read_csv(train_path)
    test = pd.read_csv(test_path)
    if TARGET not in train.columns:
        raise ValueError(f"{train_path} must contain target column '{TARGET}'.")
    if TARGET in test.columns:
        raise ValueError(f"{test_path} should not contain target column '{TARGET}'.")
    return train, test


def _sum_existing(df: pd.DataFrame, columns: Iterable[str]) -> pd.Series:
    # Chỉ cộng những cột thực sự có mặt để hàm dùng được cho cả train và test.
    cols = [c for c in columns if c in df.columns]
    if not cols:
        return pd.Series(0.0, index=df.index)
    return df[cols].fillna(0).sum(axis=1)


def add_domain_features(df: pd.DataFrame) -> pd.DataFrame:
    # Tạo thêm các biến mang ý nghĩa thực tế về diện tích, tuổi nhà và tiện nghi.
    x = df.copy()

    x["TotalSF"] = _sum_existing(x, ["TotalBsmtSF", "1stFlrSF", "2ndFlrSF"])
    x["TotalBathrooms"] = (
        x.get("FullBath", 0).fillna(0) if isinstance(x.get("FullBath", 0), pd.Series) else 0
    )
    for col, weight in [("HalfBath", 0.5), ("BsmtFullBath", 1.0), ("BsmtHalfBath", 0.5)]:
        if col in x.columns:
            x["TotalBathrooms"] = x["TotalBathrooms"] + weight * x[col].fillna(0)

    x["TotalPorchSF"] = _sum_existing(
        x, ["OpenPorchSF", "3SsnPorch", "EnclosedPorch", "ScreenPorch", "WoodDeckSF"]
    )

    if "YrSold" in x.columns and "YearBuilt" in x.columns:
        x["HouseAge"] = (x["YrSold"] - x["YearBuilt"]).clip(lower=0)
    if "YrSold" in x.columns and "YearRemodAdd" in x.columns:
        x["RemodAge"] = (x["YrSold"] - x["YearRemodAdd"]).clip(lower=0)
    if "YrSold" in x.columns and "GarageYrBlt" in x.columns:
        x["GarageAge"] = (x["YrSold"] - x["GarageYrBlt"]).clip(lower=0)

    if "YearBuilt" in x.columns and "YearRemodAdd" in x.columns:
        x["Remodeled"] = (x["YearRemodAdd"] > x["YearBuilt"]).astype(int)

    flag_sources = {
        "HasGarage": "GarageArea",
        "HasBsmt": "TotalBsmtSF",
        "HasFireplace": "Fireplaces",
        "Has2ndFloor": "2ndFlrSF",
        "HasPool": "PoolArea",
    }
    for new_col, source in flag_sources.items():
        if source in x.columns:
            x[new_col] = (x[source].fillna(0) > 0).astype(int)

    if "OverallQual" in x.columns and "GrLivArea" in x.columns:
        x["OverallQual_x_GrLivArea"] = x["OverallQual"].fillna(0) * x["GrLivArea"].fillna(0)
    if "OverallQual" in x.columns:
        x["OverallQual_x_TotalSF"] = x["OverallQual"].fillna(0) * x["TotalSF"].fillna(0)
    if "GarageCars" in x.columns and "GarageArea" in x.columns:
        x["GarageScore"] = x["GarageCars"].fillna(0) * x["GarageArea"].fillna(0)

    # Tháng bán có tính chu kỳ; giữ sin/cos để tháng 12 và tháng 1 gần nhau về mặt số học.
    if "MoSold" in x.columns:
        month = pd.to_numeric(x["MoSold"], errors="coerce")
        x["MoSold_sin"] = np.sin(2 * np.pi * month / 12.0)
        x["MoSold_cos"] = np.cos(2 * np.pi * month / 12.0)

    for col in ["GrLivArea", "LotArea", "TotalBsmtSF", "1stFlrSF", "GarageArea", "TotalSF"]:
        if col in x.columns:
            values = pd.to_numeric(x[col], errors="coerce").clip(lower=0)
            x[f"Log1p_{col}"] = np.log1p(values)

    return x


def make_feature_frame(df: pd.DataFrame, profile: str) -> pd.DataFrame:
    """Return X according to a named feature profile.

    Profiles:
      - paper_core: small subset inspired by the paper's example regressions.
      - raw: all original competition predictors, no manual feature engineering.
      - engineered: raw predictors plus domain features above.
    """
    x = df.drop(columns=[TARGET], errors="ignore").copy()
    x = x.drop(columns=[ID_COL], errors="ignore")

    if profile == "paper_core":
        cols = [c for c in PAPER_CORE_FEATURES if c in x.columns]
        if not cols:
            raise ValueError("No PAPER_CORE_FEATURES were found in the dataframe.")
        x = x[cols].copy()
    elif profile == "raw":
        pass
    elif profile == "engineered":
        x = add_domain_features(x)
    else:
        raise ValueError(f"Unknown feature profile: {profile}")

    # Ép mã số phân loại sang kiểu chuỗi để OneHotEncoder xử lý đúng bản chất dữ liệu.
    for col in CODE_AS_CATEGORY:
        if col in x.columns:
            x[col] = x[col].astype("string")
    return x


def make_preprocessor(x: pd.DataFrame) -> ColumnTransformer:
    # Tách cột số và cột phân loại, sau đó xây dựng phép tiền xử lý cho từng nhóm.
    numeric_cols = x.select_dtypes(include=[np.number, "bool"]).columns.tolist()
    categorical_cols = [c for c in x.columns if c not in numeric_cols]

    numeric_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipe = Pipeline(
        steps=[
            # Điền một giá trị cố định để vẫn giữ thông tin "thiếu/không có" của dữ liệu.
            ("imputer", SimpleImputer(strategy="constant", fill_value="Missing")),
            (
                "onehot",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False, dtype=np.float32),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipe, numeric_cols),
            ("cat", categorical_pipe, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=True,
    )


def make_model(model_name: str, random_state: int = 42, quick: bool = False):
    # Chế độ quick giảm số cây để kiểm tra nhanh; chế độ thường ưu tiên chất lượng.
    trees = 220 if quick else 700
    if model_name == "ridge":
        return Ridge(alpha=15.0)
    if model_name == "random_forest":
        return RandomForestRegressor(
            n_estimators=trees,
            max_features=0.75,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "extra_trees":
        return ExtraTreesRegressor(
            n_estimators=trees,
            max_features=0.9,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=random_state,
        )
    if model_name == "gradient_boosting":
        return GradientBoostingRegressor(
            n_estimators=320 if quick else 1200,
            learning_rate=0.04 if quick else 0.025,
            max_depth=3,
            max_features="sqrt",
            loss="huber",
            random_state=random_state,
        )
    raise ValueError(f"Unknown model name: {model_name}")


def make_sklearn_pipeline(x: pd.DataFrame, model_name: str, random_state: int = 42, quick: bool = False) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocess", make_preprocessor(x)),
            ("model", make_model(model_name, random_state=random_state, quick=quick)),
        ]
    )


def paper_outlier_mask(train: pd.DataFrame) -> pd.Series:
    """Conservative training-only rule derived from the paper's note on very large living areas.

    This rule is *not* assumed to help; it is only tested as an explicit experiment.
    """
    if "GrLivArea" not in train.columns:
        return pd.Series(True, index=train.index)
    return ~(pd.to_numeric(train["GrLivArea"], errors="coerce") > 4000)


def cross_validate_pipeline(
    train: pd.DataFrame,
    spec: ExperimentSpec,
    n_splits: int = 5,
    random_state: int = 42,
    quick: bool = False,
) -> Dict[str, object]:
    # Đánh giá bằng K-fold: mỗi fold chỉ fit tiền xử lý trên phần train của fold đó.
    data = train.copy().reset_index(drop=True)
    keep = paper_outlier_mask(data) if spec.remove_outliers else pd.Series(True, index=data.index)
    n_removed = int((~keep).sum())

    x = make_feature_frame(data, spec.feature_profile)
    y_log = np.log1p(data[TARGET].astype(float).to_numpy())

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    fold_scores: List[float] = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(x), start=1):
        # Nếu thử loại ngoại lệ, chỉ loại ở phần train; validation luôn giữ nguyên.
        fold_train_idx = tr_idx[keep.iloc[tr_idx].to_numpy()]
        pipe = make_sklearn_pipeline(x.iloc[fold_train_idx], spec.model_name, random_state=random_state + fold, quick=quick)
        pipe.fit(x.iloc[fold_train_idx], y_log[fold_train_idx])
        pred = pipe.predict(x.iloc[va_idx])
        fold_scores.append(kaggle_log_rmse(y_log[va_idx], pred))

    return {
        "experiment": spec.name,
        "feature_profile": spec.feature_profile,
        "model": spec.model_name,
        "remove_outliers": spec.remove_outliers,
        "rows_used": int(keep.sum()),
        "rows_removed": n_removed,
        "cv_folds": int(n_splits),
        "log_rmse_mean": float(np.mean(fold_scores)),
        "log_rmse_std": float(np.std(fold_scores, ddof=1)) if len(fold_scores) > 1 else 0.0,
        "fold_scores": [float(s) for s in fold_scores],
    }


def fit_full_sklearn(
    train: pd.DataFrame,
    test: pd.DataFrame,
    feature_profile: str,
    model_name: str,
    remove_outliers: bool,
    random_state: int = 42,
    quick: bool = False,
):
    fit_train = train.copy()
    if remove_outliers:
        fit_train = fit_train.loc[paper_outlier_mask(fit_train)].reset_index(drop=True)

    x_train = make_feature_frame(fit_train, feature_profile)
    x_test = make_feature_frame(test, feature_profile)
    y_log = np.log1p(fit_train[TARGET].astype(float).to_numpy())

    pipe = make_sklearn_pipeline(x_train, model_name, random_state=random_state, quick=quick)
    pipe.fit(x_train, y_log)
    pred_log = pipe.predict(x_test)
    pred_price = np.expm1(pred_log).clip(min=0)
    return pipe, pred_price


def save_feature_importance(
    train: pd.DataFrame,
    output_csv: str | Path,
    random_state: int = 42,
    quick: bool = False,
) -> pd.DataFrame:
    """Fit ExtraTrees trên đặc trưng đã biến đổi và lưu mức độ quan trọng của từng cột."""
    x = make_feature_frame(train, "engineered")
    y_log = np.log1p(train[TARGET].astype(float).to_numpy())
    pipe = make_sklearn_pipeline(x, "extra_trees", random_state=random_state, quick=quick)
    pipe.fit(x, y_log)

    pre = pipe.named_steps["preprocess"]
    model = pipe.named_steps["model"]
    names = pre.get_feature_names_out()
    imp = pd.DataFrame({"feature": names, "importance": model.feature_importances_})
    imp = imp.sort_values("importance", ascending=False).reset_index(drop=True)
    Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
    imp.to_csv(output_csv, index=False)
    return imp


def save_pipeline(pipe: Pipeline, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, path)


def write_metadata(path: str | Path, payload: Dict[str, object]) -> None:
    # Lưu thông tin cấu hình và kết quả để có thể kiểm tra lại lần chạy sau.
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
