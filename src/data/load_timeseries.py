"""
Time series loader — Corporación Favorita Store Sales (Kaggle).

Dataset: 1,782 series (54 stores × 33 product families), daily, 2013-01-01 to 2017-08-15.
Exogenous: oil prices, holidays/events, store metadata, transactions.

Target: sales (continuous, ~31% zeros; log1p transform is standard)

Tabular reframing strategy (for LightGBM / TabPFN):
  - Each row = one (date, store, family) forecast point
  - Features = lag sales, rolling means, EWM, calendar, exogenous
  - Forecast horizon is 16 days (matching competition structure), so
    minimum lag is 16 to avoid leakage
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data" / "store-sales-time-series-forecasting"
PROC_DIR = Path(__file__).parent.parent / "data" / "timeseries"

# Competition train ends 2017-08-15; we hold out the last 16 days as validation
TRAIN_END = pd.Timestamp("2017-07-30")
VAL_END   = pd.Timestamp("2017-08-15")
# Drop early 2013 rows that have no lag history (burn-in period)
BURNIN_END = pd.Timestamp("2013-12-31")

FORECAST_HORIZON = 16  # days — minimum lag to avoid leakage


# ---------------------------------------------------------------------------
# Loading raw files
# ---------------------------------------------------------------------------

def _load_raw() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_DIR / "train.csv", parse_dates=["date"])
    stores = pd.read_csv(DATA_DIR / "stores.csv")
    oil = pd.read_csv(DATA_DIR / "oil.csv", parse_dates=["date"])
    holidays = pd.read_csv(DATA_DIR / "holidays_events.csv", parse_dates=["date"])
    transactions = pd.read_csv(DATA_DIR / "transactions.csv", parse_dates=["date"])
    return train, stores, oil, holidays, transactions


def load_raw_frame() -> pd.DataFrame:
    """
    Merge all source files into a single flat frame. Minimal transforms only:
    - Oil NaNs → forward/backward fill
    - Holidays merged on date (national only to avoid duplicates)
    - Store metadata joined on store_nbr
    - Transactions joined on (date, store_nbr)
    Returns: ~3M row DataFrame, one row per (date, store_nbr, family)
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    cache = PROC_DIR / "raw_merged.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    print("Building merged frame from raw CSVs...")
    train, stores, oil, holidays, transactions = _load_raw()

    # Oil: fill gaps (weekends/holidays have no price)
    oil = oil.set_index("date").reindex(
        pd.date_range(oil.date.min(), oil.date.max(), freq="D")
    ).rename_axis("date").reset_index()
    oil["dcoilwtico"] = oil["dcoilwtico"].ffill().bfill()

    # Holidays: keep national only to avoid fan-out on merge
    nat_holidays = (
        holidays[holidays["locale"] == "National"]
        .drop_duplicates("date")[["date", "type", "description", "transferred"]]
        .rename(columns={"type": "holiday_type", "description": "holiday_desc"})
    )

    df = (
        train
        .merge(stores.rename(columns={"type": "store_type"}), on="store_nbr", how="left")
        .merge(transactions, on=["date", "store_nbr"], how="left")
        .merge(oil, on="date", how="left")
        .merge(nat_holidays, on="date", how="left")
    )

    df["holiday_type"] = df["holiday_type"].fillna("None").astype("category")
    df["holiday_desc"] = df["holiday_desc"].fillna("None").astype("category")
    df["transferred"] = df["transferred"].fillna(False)
    df["store_type"] = df["store_type"].astype("category")
    df["family"] = df["family"].astype("category")
    df["city"] = df["city"].astype("category")
    df["state"] = df["state"].astype("category")

    df = df.sort_values(["store_nbr", "family", "date"]).reset_index(drop=True)
    df.to_parquet(cache, index=False)
    print(f"Saved to {cache} ({len(df):,} rows)")
    return df


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------

def _add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["day_of_week"] = df["date"].dt.dayofweek
    df["day_of_month"] = df["date"].dt.day
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    df["month"] = df["date"].dt.month
    df["year"] = df["date"].dt.year
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    df["is_month_start"] = df["date"].dt.is_month_start.astype(int)
    df["is_month_end"] = df["date"].dt.is_month_end.astype(int)
    return df


def _add_lag_features(df: pd.DataFrame, lags: list[int]) -> pd.DataFrame:
    """Add per-series lag features. Minimum lag = FORECAST_HORIZON to avoid leakage."""
    lags = [l for l in lags if l >= FORECAST_HORIZON]
    grp = df.groupby(["store_nbr", "family"])["sales_log"]
    for lag in lags:
        df[f"lag_{lag}"] = grp.transform(lambda x: x.shift(lag))
    return df


def _add_rolling_features(df: pd.DataFrame, windows: list[int], shift: int = FORECAST_HORIZON) -> pd.DataFrame:
    """Rolling means on log sales, shifted by `shift` days."""
    grp = df.groupby(["store_nbr", "family"])["sales_log"]
    for w in windows:
        df[f"roll_mean_{w}"] = grp.transform(
            lambda x: x.shift(shift).rolling(w, min_periods=max(1, w // 2)).mean()
        )
    return df


def _add_ewm_features(df: pd.DataFrame, alphas: list[float], lags: list[int]) -> pd.DataFrame:
    """EWM features on log sales, shifted by lag days."""
    grp = df.groupby(["store_nbr", "family"])["sales_log"]
    for alpha in alphas:
        for lag in lags:
            col = f"ewm_a{str(alpha).replace('.','')}_l{lag}"
            df[col] = grp.transform(lambda x, a=alpha, l=lag: x.shift(l).ewm(alpha=a, min_periods=1).mean())
    return df


def build_tabular_frame(
    lags: list[int] | None = None,
    rolling_windows: list[int] | None = None,
    ewm_alphas: list[float] | None = None,
    ewm_lags: list[int] | None = None,
    force_rebuild: bool = False,
) -> pd.DataFrame:
    """
    Full tabular reframing of the store sales series.
    Each row = one (date, store_nbr, family) point with all features computed.

    Default feature set mirrors the reference notebook but uses log1p(sales)
    as the basis for lag/rolling/ewm features.

    Returns a DataFrame ready for train/val split via get_train_val_split().
    """
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    cache = PROC_DIR / "tabular_frame.parquet"
    if cache.exists() and not force_rebuild:
        return pd.read_parquet(cache)

    lags = lags or [16, 17, 18, 21, 28, 30, 90, 180, 364]
    rolling_windows = rolling_windows or [7, 14, 28, 56]
    ewm_alphas = ewm_alphas or [0.95, 0.8, 0.5]
    ewm_lags = ewm_lags or [FORECAST_HORIZON, 28]

    print("Building tabular frame (this takes ~2-3 min)...")
    df = load_raw_frame()

    df["sales_log"] = np.log1p(df["sales"])

    df = _add_calendar_features(df)
    df = _add_lag_features(df, lags)
    df = _add_rolling_features(df, rolling_windows)
    df = _add_ewm_features(df, ewm_alphas, ewm_lags)

    # Drop burn-in rows that have no lag history
    df = df[df["date"] > BURNIN_END].reset_index(drop=True)

    df.to_parquet(cache, index=False)
    print(f"Saved tabular frame to {cache} ({len(df):,} rows, {df.shape[1]} columns)")
    return df


# ---------------------------------------------------------------------------
# Train / validation split
# ---------------------------------------------------------------------------

FEATURE_COLS_EXCLUDE = {"id", "date", "sales", "sales_log"}


def get_train_val_split(
    df: pd.DataFrame | None = None,
) -> dict:
    """
    Time-based split:
      train → date <= TRAIN_END (2017-07-30)
      val   → TRAIN_END < date <= VAL_END (2017-08-15)

    Mirrors the competition structure: val is the last 16 days.

    Returns dict with keys:
        train, val            — full DataFrames
        X_train, X_val        — feature matrices (NaN rows dropped)
        y_train, y_val        — log1p(sales) targets
        y_raw_train, y_raw_val — raw sales (for metric evaluation)
        feature_cols          — list of feature column names used
    """
    if df is None:
        df = build_tabular_frame()

    train = df[(df["date"] > BURNIN_END) & (df["date"] <= TRAIN_END)].copy()
    val   = df[(df["date"] > TRAIN_END)  & (df["date"] <= VAL_END)].copy()

    feature_cols = [c for c in df.columns if c not in FEATURE_COLS_EXCLUDE]

    # Drop rows with NaN lag features (series start)
    lag_cols = [c for c in feature_cols if c.startswith("lag_")]
    train = train.dropna(subset=lag_cols).reset_index(drop=True)
    val   = val.dropna(subset=lag_cols).reset_index(drop=True)

    return {
        "train": train,
        "val": val,
        "X_train": train[feature_cols],
        "X_val": val[feature_cols],
        "y_train": train["sales_log"],
        "y_val": val["sales_log"],
        "y_raw_train": train["sales"],
        "y_raw_val": val["sales"],
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# Naive seasonal baseline
# ---------------------------------------------------------------------------

def naive_seasonal_baseline(df: pd.DataFrame | None = None) -> dict:
    """
    Naive seasonal forecast: predict last year's value for the same day.
    Uses raw sales (not log-transformed).

    Returns dict with actual and predicted arrays for the validation period,
    plus per-series breakdown for MASE computation.
    """
    if df is None:
        df = load_raw_frame()

    # Build a (date, store_nbr, family) → sales lookup
    lookup = df.set_index(["date", "store_nbr", "family"])["sales"]

    val_mask = (df["date"] > TRAIN_END) & (df["date"] <= VAL_END)
    val = df[val_mask].copy()

    def _last_year_sales(row):
        last_year_date = row["date"] - pd.DateOffset(years=1)
        try:
            return lookup.loc[(last_year_date, row["store_nbr"], row["family"])]
        except KeyError:
            return np.nan

    print("Computing naive seasonal baseline (last year's value)...")
    val["pred"] = val.apply(_last_year_sales, axis=1)
    val["pred"] = val["pred"].fillna(val["sales"].mean())  # fallback for missing history

    return {
        "y_true": val["sales"].values,
        "y_pred": val["pred"].values,
        "val_df": val,
    }


# ---------------------------------------------------------------------------
# Quick dataset summary
# ---------------------------------------------------------------------------

def describe_dataset(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = load_raw_frame()
    print(f"Shape: {df.shape}")
    print(f"Date range: {df.date.min().date()} to {df.date.max().date()}")
    print(f"Stores: {df.store_nbr.nunique()}, Families: {df.family.nunique()}")
    print(f"Series: {df.groupby(['store_nbr','family']).ngroups}")
    print(f"Sales zeros: {(df.sales == 0).mean():.1%}")
    print(f"Missing oil prices: {df.dcoilwtico.isna().sum()}")
    print(f"Missing transactions: {df.transactions.isna().sum()}")


if __name__ == "__main__":
    df = load_raw_frame()
    describe_dataset(df)
