"""Shared plots for the House Prices notebooks."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def _finish(fig, output_path: str | Path, title: str) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    try:
        from IPython import get_ipython
        from IPython.display import display

        shell = get_ipython()
        if shell is not None and getattr(shell, "kernel", None) is not None:
            display(fig)
    except ImportError:
        pass
    plt.close(fig)


def plot_missing_values(df: pd.DataFrame, output_path: str | Path) -> None:
    missing = df.isna().sum().sort_values(ascending=False)
    missing = missing[missing > 0].head(15).sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.3 * len(missing))))
    if missing.empty:
        ax.text(0.5, 0.5, "No missing values", ha="center", va="center")
        ax.set_axis_off()
    else:
        ax.barh(missing.index, missing.values, color="#e07a5f")
        ax.set_xlabel("Missing rows")
        ax.set_ylabel("Column")
    _finish(fig, output_path, "Columns with the most missing values")


def plot_price_distribution(
    train_prices: pd.Series,
    output_path: str | Path,
    predicted_prices: pd.Series | np.ndarray | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(train_prices, bins=35, alpha=0.65, label="Train SalePrice", color="#4c78a8")
    if predicted_prices is not None:
        ax.hist(predicted_prices, bins=35, alpha=0.65, label="Predicted SalePrice", color="#f58518")
    ax.set_xlabel("SalePrice")
    ax.set_ylabel("Number of houses")
    ax.legend()
    _finish(fig, output_path, "SalePrice distribution")


def plot_preprocessing_summary(
    numeric_count: int,
    categorical_count: int,
    transformed_count: int,
    output_path: str | Path,
) -> None:
    labels = ["Numeric inputs", "Categorical inputs", "Encoded columns"]
    values = [numeric_count, categorical_count, transformed_count]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, values, color=["#4c78a8", "#72b7b2", "#f2cf5b"])
    ax.bar_label(bars, padding=3)
    ax.set_ylabel("Number of columns")
    _finish(fig, output_path, "Feature dimensions after preprocessing")


def plot_feature_target_correlations(
    features: pd.DataFrame,
    target: pd.Series,
    output_path: str | Path,
    top_n: int = 12,
) -> None:
    numeric = features.select_dtypes(include=[np.number, "bool"])
    correlations = numeric.corrwith(pd.Series(target, index=features.index)).dropna()
    correlations = correlations.reindex(correlations.abs().sort_values(ascending=False).head(top_n).index)
    correlations = correlations.sort_values()
    fig, ax = plt.subplots(figsize=(9, max(4, 0.35 * len(correlations))))
    colors = ["#e07a5f" if value < 0 else "#4c78a8" for value in correlations]
    ax.barh(correlations.index, correlations.values, color=colors)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Pearson correlation with SalePrice")
    _finish(fig, output_path, "Strongest numeric feature associations (not causal)")


def plot_cv_results(results: pd.DataFrame, output_path: str | Path) -> None:
    ranked = results.sort_values("log_rmse_mean", ascending=True)
    errors = ranked.get("log_rmse_std", pd.Series(0.0, index=ranked.index)).fillna(0)
    fig, ax = plt.subplots(figsize=(10, max(4, 0.5 * len(ranked))))
    ax.barh(ranked["experiment"], ranked["log_rmse_mean"], xerr=errors, color="#4c78a8", capsize=3)
    ax.invert_yaxis()
    ax.set_xlabel("CV log-RMSE (lower is better; bars show ±1 std)")
    _finish(fig, output_path, "Cross-validation model comparison")


def plot_fold_scores(scores: list[float], output_path: str | Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    folds = np.arange(1, len(scores) + 1)
    ax.bar(folds, scores, color="#72b7b2")
    ax.axhline(np.mean(scores), color="#e07a5f", linestyle="--", label=f"Mean: {np.mean(scores):.4f}")
    ax.set_xticks(folds, [f"Fold {fold}" for fold in folds])
    ax.set_ylabel("log-RMSE (lower is better)")
    ax.legend()
    _finish(fig, output_path, title)


def plot_feature_importance(importance: pd.DataFrame, output_path: str | Path, top_n: int = 20) -> None:
    top = importance.nlargest(top_n, "importance").sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, max(5, 0.3 * len(top))))
    ax.barh(top["feature"], top["importance"], color="#f2a541")
    ax.set_xlabel("Importance")
    _finish(fig, output_path, f"Top {len(top)} transformed feature importances")
