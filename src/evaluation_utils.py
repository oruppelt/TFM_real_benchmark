"""
Shared evaluation utilities.

Provides:
  - run_model(): fit + predict + score any sklearn-compatible model
  - TimeSeriesCV: walk-forward cross-validation for tabular-reframed series
  - compare_models(): run multiple models and return a results DataFrame
"""

from __future__ import annotations

import time
import tracemalloc
from typing import Any, Callable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from src.metrics import insurance_metrics, timeseries_metrics


# ---------------------------------------------------------------------------
# Generic model runner
# ---------------------------------------------------------------------------

def run_model(
    model: BaseEstimator,
    X_train: pd.DataFrame | np.ndarray,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame | np.ndarray,
    y_test: pd.Series | np.ndarray,
    metric_fn: Callable,
    metric_kwargs: dict | None = None,
    sample_weight_train: np.ndarray | None = None,
    sample_weight_test: np.ndarray | None = None,
    model_name: str = "model",
) -> dict[str, Any]:
    """
    Fit a model, predict on test, compute metrics + timing + memory.

    Returns dict with:
        name, metrics, train_time_s, predict_time_s, peak_memory_mb
    """
    metric_kwargs = metric_kwargs or {}

    # --- train ---
    tracemalloc.start()
    t0 = time.perf_counter()
    fit_kwargs = {}
    if sample_weight_train is not None:
        fit_kwargs["sample_weight"] = sample_weight_train
    model.fit(X_train, y_train, **fit_kwargs)
    train_time = time.perf_counter() - t0

    # --- predict ---
    t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    predict_time = time.perf_counter() - t0

    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    # --- metrics ---
    if sample_weight_test is not None:
        metric_kwargs["sample_weight"] = sample_weight_test
    metrics = metric_fn(y_test, y_pred, **metric_kwargs)

    return {
        "name": model_name,
        "metrics": metrics,
        "train_time_s": round(train_time, 3),
        "predict_time_s": round(predict_time, 4),
        "peak_memory_mb": round(peak / 1024**2, 1),
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Insurance evaluation
# ---------------------------------------------------------------------------

def evaluate_insurance_model(
    model: BaseEstimator,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    exposure_train: np.ndarray | None = None,
    exposure_test: np.ndarray | None = None,
    model_name: str = "model",
    predict_rate: bool = False,
) -> dict[str, Any]:
    """
    Fit and evaluate an insurance model.

    If predict_rate=True, trains on y/exposure and post-multiplies predictions by exposure.
    Otherwise trains on raw counts.

    Returns the standard run_model result dict.
    """
    if predict_rate and exposure_train is not None:
        y_train_fit = y_train.values / exposure_train
    else:
        y_train_fit = y_train.values

    result = run_model(
        model=model,
        X_train=X_train,
        y_train=y_train_fit,
        X_test=X_test,
        y_test=y_test.values,
        metric_fn=lambda yt, yp, **kw: insurance_metrics(
            yt, yp * (exposure_test if (predict_rate and exposure_test is not None) else 1), kw.get("sample_weight")
        ),
        sample_weight_train=exposure_train,
        model_name=model_name,
    )
    return result


# ---------------------------------------------------------------------------
# Time series walk-forward CV
# ---------------------------------------------------------------------------

class TimeSeriesCV:
    """
    Walk-forward cross-validation for tabular-reframed time series.

    Splits on date: train on [start, cutoff], validate on (cutoff, cutoff + horizon].
    Rolls forward by `step` days for each fold.

    Example:
        cv = TimeSeriesCV(n_splits=3, horizon=16, step=30)
        for fold, (X_tr, y_tr, X_val, y_val) in enumerate(cv.split(df)):
            ...
    """

    def __init__(
        self,
        n_splits: int = 3,
        horizon: int = 16,
        step: int = 30,
        min_train_days: int = 365,
        date_col: str = "date",
    ):
        self.n_splits = n_splits
        self.horizon = horizon
        self.step = step
        self.min_train_days = min_train_days
        self.date_col = date_col

    def split(self, df: pd.DataFrame, target_col: str = "sales_log") -> list[tuple]:
        """Yield (X_train, y_train, X_val, y_val, cutoff_date) for each fold."""
        dates = df[self.date_col].sort_values().unique()
        feature_cols = [c for c in df.columns if c not in {self.date_col, "sales", "sales_log", "id"}]

        # Work backwards from the last date
        last_date = pd.Timestamp(dates[-1])
        folds = []
        for i in range(self.n_splits):
            val_end   = last_date - pd.Timedelta(days=i * self.step)
            val_start = val_end - pd.Timedelta(days=self.horizon - 1)
            train_end = val_start - pd.Timedelta(days=1)

            train_mask = df[self.date_col] <= train_end
            val_mask   = (df[self.date_col] >= val_start) & (df[self.date_col] <= val_end)

            if train_mask.sum() < self.min_train_days:
                continue

            train_df = df[train_mask].dropna(subset=[c for c in feature_cols if "lag_" in c])
            val_df   = df[val_mask].dropna(subset=[c for c in feature_cols if "lag_" in c])

            folds.append((
                train_df[feature_cols], train_df[target_col],
                val_df[feature_cols],   val_df[target_col],
                train_end,
            ))
        return list(reversed(folds))  # chronological order


def evaluate_timeseries_model(
    model: BaseEstimator,
    df: pd.DataFrame,
    cv: TimeSeriesCV | None = None,
    model_name: str = "model",
    target_col: str = "sales_log",
    y_raw_col: str = "sales",
) -> pd.DataFrame:
    """
    Run walk-forward CV for a time series model and return per-fold metrics.

    Metrics are computed on expm1(pred) vs actual raw sales (not log scale).
    """
    if cv is None:
        cv = TimeSeriesCV()

    rows = []
    for fold_idx, (X_tr, y_tr, X_val, y_val, cutoff) in enumerate(cv.split(df, target_col)):
        t0 = time.perf_counter()
        model.fit(X_tr, y_tr)
        train_time = time.perf_counter() - t0

        y_pred_log = model.predict(X_val)
        predict_time = time.perf_counter() - t0 - train_time

        # Back-transform from log space
        y_pred = np.expm1(y_pred_log)
        y_true = np.expm1(y_val.values)

        metrics = timeseries_metrics(y_true, y_pred)
        metrics.update({
            "model": model_name,
            "fold": fold_idx,
            "cutoff": cutoff,
            "train_time_s": round(train_time, 3),
            "predict_time_s": round(predict_time, 4),
            "n_train": len(X_tr),
            "n_val": len(X_val),
        })
        rows.append(metrics)
        print(f"  Fold {fold_idx} | cutoff={cutoff.date()} | RMSE={metrics['rmse']:.2f} | sMAPE={metrics['smape']:.1f}%")

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Multi-model comparison
# ---------------------------------------------------------------------------

def compare_models(
    models: dict[str, BaseEstimator],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    metric_fn: Callable,
    **run_kwargs,
) -> pd.DataFrame:
    """
    Run multiple models and return a tidy comparison DataFrame.

    Args:
        models: dict of {name: fitted-or-unfitted model}
        metric_fn: function(y_true, y_pred) → dict of metrics
        **run_kwargs: passed to run_model (e.g. sample_weight_train)

    Returns: DataFrame with one row per model, columns = metrics + timing
    """
    rows = []
    for name, model in models.items():
        print(f"Running {name}...")
        result = run_model(
            model=model,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test,
            metric_fn=metric_fn,
            model_name=name,
            **run_kwargs,
        )
        row = {"model": name, **result["metrics"],
               "train_time_s": result["train_time_s"],
               "predict_time_s": result["predict_time_s"],
               "peak_memory_mb": result["peak_memory_mb"]}
        rows.append(row)
        print(f"  {name}: {result['metrics']}")

    return pd.DataFrame(rows).set_index("model")
