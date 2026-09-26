import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge

from house_price_pipeline import (
    add_domain_features,
    kaggle_log_rmse,
    make_feature_frame,
    make_preprocessor,
)
from pytorch_mlp import TorchMLPRegressor


def _tiny_df():
    # Dữ liệu giả nhỏ để các bài test chạy nhanh, không phụ thuộc toàn bộ train.csv.
    return pd.DataFrame(
        {
            "Id": [1, 2, 3, 4, 5, 6],
            "MSSubClass": [20, 20, 60, 60, 20, 70],
            "MoSold": [1, 2, 3, 4, 5, 6],
            "Neighborhood": ["A", "A", "B", "B", "C", None],
            "OverallQual": [5, 6, 7, 8, 5, 7],
            "LotArea": [8000, 9000, 10000, 11000, 7000, 9500],
            "TotalBsmtSF": [800, 900, 1000, 1100, 750, np.nan],
            "1stFlrSF": [900, 1000, 1050, 1200, 800, 1000],
            "2ndFlrSF": [0, 0, 700, 800, 0, 500],
            "GrLivArea": [900, 1000, 1750, 2000, 800, 1500],
            "GarageCars": [1, 2, 2, 2, 1, 2],
            "GarageArea": [300, 450, 500, 520, 280, 480],
            "Fireplaces": [0, 1, 1, 2, 0, 1],
            "FullBath": [1, 2, 2, 2, 1, 2],
            "HalfBath": [0, 0, 1, 1, 0, 1],
            "BsmtFullBath": [0, 1, 0, 1, 0, 0],
            "BsmtHalfBath": [0, 0, 0, 0, 0, 0],
            "YearBuilt": [1980, 1990, 2000, 2005, 1975, 1998],
            "YearRemodAdd": [1980, 2000, 2000, 2010, 1980, 2005],
            "YrSold": [2008, 2008, 2009, 2010, 2007, 2009],
            "SalePrice": [120000, 150000, 210000, 260000, 100000, 190000],
        }
    )


def test_domain_features_are_created():
    # Kiểm tra các cột dẫn xuất quan trọng được tạo và tuổi nhà không âm.
    x = add_domain_features(_tiny_df().drop(columns=["SalePrice"]))
    expected = {
        "TotalSF",
        "TotalBathrooms",
        "HouseAge",
        "RemodAge",
        "OverallQual_x_GrLivArea",
        "MoSold_sin",
        "MoSold_cos",
        "Log1p_GrLivArea",
    }
    assert expected.issubset(x.columns)
    assert (x["HouseAge"] >= 0).all()


def test_preprocessor_handles_missing_and_unseen_category():
    # Đảm bảo tiền xử lý chịu được giá trị thiếu và category chưa từng gặp.
    df = _tiny_df()
    x = make_feature_frame(df, "engineered")
    pre = make_preprocessor(x)
    tr = pre.fit_transform(x.iloc[:5])
    te = pre.transform(x.iloc[[5]].assign(Neighborhood="NEVER_SEEN"))
    assert tr.shape[1] == te.shape[1]
    assert np.isfinite(tr).all()
    assert np.isfinite(te).all()


def test_log_rmse_zero_for_identical_predictions():
    # Dự đoán giống nhãn thật phải có sai số bằng 0.
    y = np.log1p(np.array([100000.0, 200000.0]))
    assert kaggle_log_rmse(y, y) == 0.0


def test_ridge_pipeline_smoke():
    # Smoke test cho pipeline Ridge: fit, predict và kiểm tra shape/kết quả hữu hạn.
    df = _tiny_df()
    x = make_feature_frame(df, "engineered")
    y = np.log1p(df["SalePrice"].to_numpy())
    pipe = Pipeline([("pre", make_preprocessor(x)), ("model", Ridge(alpha=1.0))])
    pipe.fit(x, y)
    pred = pipe.predict(x)
    assert pred.shape == y.shape
    assert np.isfinite(pred).all()


def test_torch_mlp_smoke():
    # Smoke test ngắn cho MLP PyTorch trên dữ liệu số ngẫu nhiên.
    rng = np.random.default_rng(42)
    X = rng.normal(size=(80, 12)).astype(np.float32)
    y = (10.0 + 0.2 * X[:, 0] - 0.1 * X[:, 1] + rng.normal(0, 0.03, size=80)).astype(np.float32)
    model = TorchMLPRegressor(
        hidden1=24,
        hidden2=12,
        hidden3=6,
        max_epochs=12,
        patience=4,
        batch_size=16,
        random_state=42,
    )
    model.fit(X, y)
    pred = model.predict(X[:7])
    assert pred.shape == (7,)
    assert np.isfinite(pred).all()
