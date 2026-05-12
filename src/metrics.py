"""
Shared metrics for all experiments.

Usage:
    from src.metrics import tweedie_deviance, mase, rmsle

All functions follow the convention: metrics(y_true, y_pred, **kwargs) → float.
Lower is better for all deviance/error metrics.
Higher is better for Gini and AUC.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike


# ---------------------------------------------------------------------------
# Insurance metrics
# ---------------------------------------------------------------------------

def tweedie_deviance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    power: float = 1.5,
    sample_weight: ArrayLike | None = None,
) -> float:
    """
    Mean Tweedie deviance for power ∈ (0, 2).
      power=1 → Poisson deviance
      power=2 → Gamma deviance
      power=1.5 → compound Poisson-Gamma (pure premium)

    Formula: D(y, mu) = 2 * [ y^(2-p)/((1-p)(2-p)) - y*mu^(1-p)/(1-p) + mu^(2-p)/(2-p) ]
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    if np.any(y_pred <= 0):
        raise ValueError("y_pred must be strictly positive for Tweedie deviance")

    p = power
    if p == 1:
        # Poisson deviance — safe log avoids divide-by-zero when y_true=0
        safe_log = np.where(y_true > 0, np.log(np.where(y_true > 0, y_true, 1) / y_pred), 0)
        dev = 2 * (y_true * safe_log - (y_true - y_pred))
    elif p == 2:
        # Gamma deviance
        if np.any(y_true <= 0):
            raise ValueError("y_true must be strictly positive for Gamma deviance (power=2)")
        dev = 2 * (np.log(y_pred / y_true) + y_true / y_pred - 1)
    else:
        term1 = np.where(y_true > 0, y_true ** (2 - p) / ((1 - p) * (2 - p)), 0)
        term2 = y_true * y_pred ** (1 - p) / (1 - p)
        term3 = y_pred ** (2 - p) / (2 - p)
        dev = 2 * (term1 - term2 + term3)

    if sample_weight is not None:
        w = np.asarray(sample_weight, dtype=float)
        return float(np.average(dev, weights=w))
    return float(np.mean(dev))


def poisson_deviance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Mean Poisson deviance. Alias for tweedie_deviance(power=1)."""
    return tweedie_deviance(y_true, y_pred, power=1.0, sample_weight=sample_weight)


def gamma_deviance(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    """Mean Gamma deviance. Alias for tweedie_deviance(power=2)."""
    return tweedie_deviance(y_true, y_pred, power=2.0, sample_weight=sample_weight)


def gini_coefficient(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Normalized Gini coefficient (actuarial ordering metric).
    Ranges [-1, 1]; higher is better. Random model = 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    # Sort by predicted risk (descending)
    order = np.argsort(-y_pred)
    y_sorted = y_true[order]

    n = len(y_true)
    cum_actual = np.cumsum(y_sorted) / y_true.sum()
    lorenz_area = cum_actual.sum() / n

    # Gini = 2 * (area under Lorenz - 0.5)
    gini = 2 * lorenz_area - 1

    # Normalize by the perfect model Gini
    order_perfect = np.argsort(-y_true)
    y_perfect = y_true[order_perfect]
    cum_perfect = np.cumsum(y_perfect) / y_true.sum()
    perfect_gini = 2 * (cum_perfect.sum() / n) - 1

    return float(gini / perfect_gini) if perfect_gini != 0 else 0.0


def rmse(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    sq = (y_true - y_pred) ** 2
    if sample_weight is not None:
        return float(np.sqrt(np.average(sq, weights=sample_weight)))
    return float(np.sqrt(np.mean(sq)))


def mae(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    abs_err = np.abs(y_true - y_pred)
    if sample_weight is not None:
        return float(np.average(abs_err, weights=sample_weight))
    return float(np.mean(abs_err))


# ---------------------------------------------------------------------------
# Time series metrics
# ---------------------------------------------------------------------------

def mase(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: ArrayLike,
    seasonality: int = 7,
) -> float:
    """
    Mean Absolute Scaled Error.
    Scale = mean absolute seasonal naive error on the training series.

    Args:
        y_true: actual values in the forecast horizon
        y_pred: predicted values
        y_train: historical training values (used to compute scale)
        seasonality: period for seasonal naive baseline (7 = weekly)
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    y_train = np.asarray(y_train, dtype=float)

    naive_errors = np.abs(y_train[seasonality:] - y_train[:-seasonality])
    scale = naive_errors.mean()
    if scale == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)) / scale)


def smape(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """
    Symmetric Mean Absolute Percentage Error (%).
    sMAPE = 200 * mean( |y - yhat| / (|y| + |yhat|) )
    Returns 0 where both y and yhat are 0.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denom = np.abs(y_true) + np.abs(y_pred)
    safe_denom = np.where(denom == 0, 1.0, denom)
    err = np.where(denom == 0, 0.0, 2 * np.abs(y_true - y_pred) / safe_denom)
    return float(100 * np.mean(err))


def rmsle(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    """Root Mean Squared Log Error. Clips predictions to ≥0."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.clip(np.asarray(y_pred, dtype=float), 0, None)
    return float(np.sqrt(np.mean((np.log1p(y_true) - np.log1p(y_pred)) ** 2)))


def wql(
    y_true: ArrayLike,
    y_quantile_preds: dict[float, ArrayLike],
) -> float:
    """
    Weighted Quantile Loss (mean over quantiles).
    Measures calibration of probabilistic forecasts.

    Args:
        y_true: actual values, shape (n,)
        y_quantile_preds: dict mapping quantile level (e.g. 0.1) to predictions array

    Returns:
        Mean WQL across all quantiles (lower is better)
    """
    y_true = np.asarray(y_true, dtype=float)
    losses = []
    for q, y_q in y_quantile_preds.items():
        y_q = np.asarray(y_q, dtype=float)
        err = y_true - y_q
        loss = np.where(err >= 0, q * err, (q - 1) * err)
        losses.append(2 * np.mean(loss))
    return float(np.mean(losses))


def coverage(
    y_true: ArrayLike,
    y_lower: ArrayLike,
    y_upper: ArrayLike,
) -> float:
    """
    Empirical coverage of a prediction interval.
    Returns fraction of actuals that fall within [lower, upper].
    """
    y_true = np.asarray(y_true, dtype=float)
    y_lower = np.asarray(y_lower, dtype=float)
    y_upper = np.asarray(y_upper, dtype=float)
    return float(np.mean((y_true >= y_lower) & (y_true <= y_upper)))


# ---------------------------------------------------------------------------
# Classification metrics (for optional use)
# ---------------------------------------------------------------------------

def gini_auc(y_true: ArrayLike, y_score: ArrayLike) -> float:
    """Gini = 2 * AUC - 1. Faster than sklearn for large arrays."""
    y_true = np.asarray(y_true, dtype=float)
    y_score = np.asarray(y_score, dtype=float)
    order = np.argsort(-y_score)
    y_sorted = y_true[order]
    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        return 0.0
    tpr = np.cumsum(y_sorted) / n_pos
    fpr = np.cumsum(1 - y_sorted) / n_neg
    auc = np.trapz(tpr, fpr)
    return float(2 * auc - 1)


# ---------------------------------------------------------------------------
# Convenience: compute all insurance metrics at once
# ---------------------------------------------------------------------------

def insurance_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    sample_weight: ArrayLike | None = None,
) -> dict[str, float]:
    """Return dict of all insurance metrics. y_pred must be positive."""
    return {
        "poisson_deviance": poisson_deviance(y_true, y_pred, sample_weight),
        "tweedie_deviance_1.5": tweedie_deviance(y_true, y_pred, power=1.5, sample_weight=sample_weight),
        "rmse": rmse(y_true, y_pred, sample_weight),
        "mae": mae(y_true, y_pred, sample_weight),
        "gini": gini_coefficient(y_true, y_pred),
    }


def timeseries_metrics(
    y_true: ArrayLike,
    y_pred: ArrayLike,
    y_train: ArrayLike | None = None,
    seasonality: int = 7,
) -> dict[str, float]:
    """Return dict of all time series point-forecast metrics."""
    result = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "smape": smape(y_true, y_pred),
        "rmsle": rmsle(y_true, y_pred),
    }
    if y_train is not None:
        result["mase"] = mase(y_true, y_pred, y_train, seasonality=seasonality)
    return result
