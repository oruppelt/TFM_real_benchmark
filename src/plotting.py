"""
Reusable chart functions for the TabFM series.

All functions return (fig, ax) or (fig, axes) so callers can save or adjust.
Style: clean, publication-ready, no seaborn themes imposed by default.
"""

from __future__ import annotations

from typing import Any

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

PALETTE = {
    "tabpfn": "#E63946",
    "xgboost": "#457B9D",
    "lightgbm": "#2A9D8F",
    "catboost": "#F4A261",
    "glm": "#6C757D",
    "baseline": "#ADB5BD",
}

DEFAULT_FIGSIZE = (10, 5)


def _model_color(name: str) -> str:
    for key, color in PALETTE.items():
        if key.lower() in name.lower():
            return color
    return "#555555"


# ---------------------------------------------------------------------------
# Bar comparison chart
# ---------------------------------------------------------------------------

def plot_model_comparison(
    results: pd.DataFrame,
    metric: str,
    title: str = "",
    lower_is_better: bool = True,
    figsize: tuple = DEFAULT_FIGSIZE,
    highlight_best: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Horizontal bar chart comparing models on a single metric.

    Args:
        results: DataFrame with model names as index, metric columns
        metric: column name to plot
        lower_is_better: if True, shortest bar = best; annotated in title
    """
    df = results[[metric]].sort_values(metric, ascending=not lower_is_better)

    fig, ax = plt.subplots(figsize=figsize)
    colors = [_model_color(name) for name in df.index]

    bars = ax.barh(df.index, df[metric], color=colors, edgecolor="white", height=0.6)

    # Highlight best
    if highlight_best:
        best_idx = df[metric].idxmin() if lower_is_better else df[metric].idxmax()
        best_pos = list(df.index).index(best_idx)
        bars[best_pos].set_edgecolor("black")
        bars[best_pos].set_linewidth(1.5)

    # Value labels
    for bar in bars:
        w = bar.get_width()
        ax.text(w * 1.01, bar.get_y() + bar.get_height() / 2,
                f"{w:.4f}", va="center", ha="left", fontsize=9)

    ax.set_xlabel(metric)
    ax.set_title(title or f"{metric} by model {'(lower is better)' if lower_is_better else '(higher is better)'}")
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Multi-metric comparison table as chart
# ---------------------------------------------------------------------------

def plot_metrics_heatmap(
    results: pd.DataFrame,
    metrics: list[str],
    title: str = "Model × Metric comparison",
    figsize: tuple = (12, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Heatmap of normalized metric scores (0=worst, 1=best per metric).
    Useful for spotting which model wins on which axis.
    """
    df = results[metrics].copy()

    # Normalize each metric to [0, 1] — 1 = best
    normalized = pd.DataFrame(index=df.index, columns=metrics, dtype=float)
    for col in metrics:
        mn, mx = df[col].min(), df[col].max()
        if mx == mn:
            normalized[col] = 1.0
        else:
            # Assume lower is better; flip so 1 = best
            normalized[col] = (mx - df[col]) / (mx - mn)

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(normalized.values.astype(float), aspect="auto", cmap="RdYlGn", vmin=0, vmax=1)

    ax.set_xticks(range(len(metrics)))
    ax.set_xticklabels(metrics, rotation=30, ha="right")
    ax.set_yticks(range(len(df.index)))
    ax.set_yticklabels(df.index)

    for i in range(len(df.index)):
        for j in range(len(metrics)):
            val = df.iloc[i, j]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8,
                    color="black" if 0.3 < normalized.iloc[i, j] < 0.7 else "white")

    plt.colorbar(im, ax=ax, label="Normalized score (1=best)")
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# SHAP comparison side-by-side
# ---------------------------------------------------------------------------

def plot_shap_comparison(
    shap_values_a: Any,
    shap_values_b: Any,
    feature_names: list[str],
    label_a: str = "Model A",
    label_b: str = "Model B",
    top_n: int = 15,
    figsize: tuple = (14, 6),
) -> tuple[plt.Figure, list[plt.Axes]]:
    """
    Side-by-side mean |SHAP| bar charts for two models.
    Shows top_n features by mean(|SHAP|) in model A.
    """
    mean_abs_a = np.abs(np.array(shap_values_a)).mean(axis=0)
    mean_abs_b = np.abs(np.array(shap_values_b)).mean(axis=0)

    # Rank by model A importance
    top_idx = np.argsort(-mean_abs_a)[:top_n]
    names = [feature_names[i] for i in top_idx]
    vals_a = mean_abs_a[top_idx]
    vals_b = mean_abs_b[top_idx]

    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True)

    for ax, vals, label, color in [
        (axes[0], vals_a, label_a, _model_color(label_a)),
        (axes[1], vals_b, label_b, _model_color(label_b)),
    ]:
        ax.barh(names, vals, color=color, edgecolor="white")
        ax.set_xlabel("Mean |SHAP value|")
        ax.set_title(label)
        ax.spines[["top", "right"]].set_visible(False)
        ax.invert_yaxis()

    fig.suptitle(f"Feature importance — {label_a} vs {label_b} (top {top_n})")
    fig.tight_layout()
    return fig, list(axes)


# ---------------------------------------------------------------------------
# Calibration plot (for probabilistic forecasts / insurance)
# ---------------------------------------------------------------------------

def plot_calibration(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bins: int = 10,
    title: str = "Calibration plot",
    figsize: tuple = (6, 6),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Reliability diagram: actual mean vs predicted mean within quantile bins.
    A well-calibrated model follows the diagonal.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    order = np.argsort(y_pred)
    y_true_s = y_true[order]
    y_pred_s = y_pred[order]

    bins = np.array_split(np.arange(len(y_pred)), n_bins)
    actual_means  = [y_true_s[b].mean() for b in bins]
    predict_means = [y_pred_s[b].mean() for b in bins]

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(predict_means, actual_means, "o-", color="#E63946", label="Model", zorder=3)
    diag = [min(predict_means), max(predict_means)]
    ax.plot(diag, diag, "--", color="#ADB5BD", label="Perfect calibration")

    ax.set_xlabel("Predicted mean")
    ax.set_ylabel("Actual mean")
    ax.set_title(title)
    ax.legend()
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Time series forecast plot
# ---------------------------------------------------------------------------

def plot_forecast(
    dates: pd.Series,
    y_true: np.ndarray,
    forecasts: dict[str, np.ndarray],
    title: str = "Forecast comparison",
    figsize: tuple = (14, 5),
    history_days: int = 60,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plot actual series + multiple model forecasts.
    Shows `history_days` of training actuals for context.
    """
    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(dates, y_true, color="black", linewidth=1.5, label="Actual", zorder=5)

    for name, y_pred in forecasts.items():
        ax.plot(dates[-len(y_pred):], y_pred,
                color=_model_color(name), linewidth=1.2, linestyle="--", label=name)

    ax.set_title(title)
    ax.set_xlabel("Date")
    ax.set_ylabel("Sales")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax


# ---------------------------------------------------------------------------
# Deviance curve (insurance: predicted vs actual by decile)
# ---------------------------------------------------------------------------

def plot_deviance_curve(
    y_true: np.ndarray,
    predictions: dict[str, np.ndarray],
    exposure: np.ndarray | None = None,
    n_bins: int = 10,
    title: str = "Actual vs predicted by risk decile",
    figsize: tuple = (10, 5),
) -> tuple[plt.Figure, plt.Axes]:
    """
    Sort policies by predicted risk, bin into deciles, compare actual vs predicted rates.
    Standard actuarial validation chart.
    """
    fig, ax = plt.subplots(figsize=figsize)
    x = np.arange(1, n_bins + 1)

    for name, y_pred in predictions.items():
        order = np.argsort(y_pred)
        y_pred_s = y_pred[order]
        y_true_s = y_true[order]
        exp_s = exposure[order] if exposure is not None else np.ones_like(y_true)

        bins = np.array_split(np.arange(len(y_pred)), n_bins)
        pred_rates   = [y_pred_s[b].mean() for b in bins]
        actual_rates = [(y_true_s[b] / exp_s[b]).mean() for b in bins]

        ax.plot(x, pred_rates, "o--", color=_model_color(name), label=f"{name} (pred)")
        ax.plot(x, actual_rates, "s-", color=_model_color(name), alpha=0.5, label=f"{name} (actual)")

    ax.set_xlabel("Risk decile (1=lowest, 10=highest)")
    ax.set_ylabel("Mean claim rate")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    return fig, ax
