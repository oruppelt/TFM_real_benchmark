"""
Insurance dataset loader — freMTPL2 (French Motor Third-Party Liability).

Combines frequency (freMTPL2freq, ~678K policies) and severity (freMTPL2sev,
one row per claim) into a single modeling frame.

Targets constructed:
    claim_flag    — 1 if any claims, 0 otherwise (hurdle stage 1)
    claim_freq    — ClaimNb (count target for Poisson frequency model)
    severity      — ClaimAmount / ClaimNb (avg cost per claim, NaN where ClaimNb=0)
    pure_premium  — ClaimAmount / Exposure (direct Tweedie target)

Split architecture:
    holdout     — 20% of full dataset, stratified on claim_flag, indices saved to disk
    development — remaining 80%; all CV/tuning runs on this set only
    cv_folds    — 5 stratified folds on development set, indices saved to disk

The same holdout indices and CV fold indices are used for every model, every
feature set, every approach. This is what makes fold-level comparisons valid.
"""

from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import StratifiedKFold, train_test_split

DATA_DIR = Path(__file__).parent.parent.parent / "data" / "insurance"

# Raw cache paths
_FREQ_RAW = DATA_DIR / "fremtpl2_freq_raw.parquet"
_SEV_RAW = DATA_DIR / "fremtpl2_sev_raw.parquet"

# Processed / split paths
PROCESSED_PATH = DATA_DIR / "processed.parquet"
HOLDOUT_IDX_PATH = DATA_DIR / "holdout_idx.npy"
CV_FOLDS_PATH = DATA_DIR / "cv_folds.pkl"

# Column groups
CATEGORICAL_COLS = ["Area", "VehBrand", "VehGas", "Region"]
NUMERIC_COLS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]
EXPOSURE_COL = "Exposure"
TARGET_COLS = ["claim_flag", "claim_freq", "severity", "pure_premium"]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _load_raw_frequency(force: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if _FREQ_RAW.exists() and not force:
        return pd.read_parquet(_FREQ_RAW)
    print("Downloading freMTPL2freq (OpenML id=41214)...")
    bunch = fetch_openml(data_id=41214, as_frame=True, parser="auto")
    df = bunch.frame
    df["ClaimNb"] = df["ClaimNb"].astype(int)
    df["Exposure"] = df["Exposure"].astype(float)
    df.to_parquet(_FREQ_RAW, index=False)
    print(f"  Saved {len(df):,} rows → {_FREQ_RAW}")
    return df


def _load_raw_severity(force: bool = False) -> pd.DataFrame:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if _SEV_RAW.exists() and not force:
        return pd.read_parquet(_SEV_RAW)
    print("Downloading freMTPL2sev (OpenML id=41215)...")
    bunch = fetch_openml(data_id=41215, as_frame=True, parser="auto")
    df = bunch.frame
    df.to_parquet(_SEV_RAW, index=False)
    print(f"  Saved {len(df):,} rows → {_SEV_RAW}")
    return df


# ---------------------------------------------------------------------------
# Target construction
# ---------------------------------------------------------------------------


def _build_targets(freq_df: pd.DataFrame, sev_df: pd.DataFrame) -> pd.DataFrame:
    """
    Join severity onto frequency and construct all four modeling targets.

    Severity join logic:
        freMTPL2sev has one row per individual claim. A policy with ClaimNb=2
        has two rows in the severity table. We aggregate: sum(ClaimAmount) per
        IDpol to get total claim cost per policy, then left-join onto the
        frequency table. Policies with no claims receive ClaimAmount_total=0.

    Target definitions:
        claim_flag    = 1 if ClaimNb > 0 else 0
                        Stratification variable for splits; also hurdle stage 1 target.

        claim_freq    = ClaimNb (raw count)
                        Poisson frequency model target. GLM uses log(Exposure) as
                        offset; GBMs use Exposure as sample_weight.

        severity      = ClaimAmount_total / ClaimNb (average cost per claim)
                        NaN where ClaimNb = 0. Only used on the claims-only subset
                        (~3-4% of policies). This is the natural turf for TabPFN
                        (small n, no exposure complexity).

        pure_premium  = ClaimAmount_total / Exposure (expected annualised loss)
                        Direct Tweedie modeling target. Normalising by Exposure
                        makes policies with different exposure durations comparable.
    """
    sev_agg = (
        sev_df.groupby("IDpol")["ClaimAmount"]
        .sum()
        .reset_index()
        .rename(columns={"ClaimAmount": "ClaimAmount_total"})
    )

    df = freq_df.merge(sev_agg, on="IDpol", how="left")
    df["ClaimAmount_total"] = df["ClaimAmount_total"].fillna(0.0)

    # Clip extreme values — policies with >4 claims are likely data anomalies
    df["ClaimNb"] = df["ClaimNb"].clip(upper=4)
    # Exposure must be strictly positive for log-offset; cap at 1 year
    df["Exposure"] = df["Exposure"].clip(lower=1e-6, upper=1.0)

    df["claim_flag"] = (df["ClaimNb"] > 0).astype(int)
    df["claim_freq"] = df["ClaimNb"]
    df["severity"] = np.where(
        # Severity only valid where ClaimNb > 0 AND ClaimAmount_total > 0.
        # Some claims in freMTPL2sev have ClaimAmount=0 (settled at zero cost).
        # Gamma GLM requires strictly positive response, so these are excluded.
        (df["ClaimNb"] > 0) & (df["ClaimAmount_total"] > 0),
        df["ClaimAmount_total"] / df["ClaimNb"],
        np.nan,
    )
    df["pure_premium"] = df["ClaimAmount_total"] / df["Exposure"]

    return df


# ---------------------------------------------------------------------------
# Full processing pipeline
# ---------------------------------------------------------------------------


def load_processed(force: bool = False) -> pd.DataFrame:
    """
    Download, join, and process the full freMTPL2 dataset.
    Caches to processed.parquet after first run.

    Returns a DataFrame with all feature columns + four target columns.
    IDpol is retained for traceability but excluded from feature lists.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    if PROCESSED_PATH.exists() and not force:
        return pd.read_parquet(PROCESSED_PATH)

    freq = _load_raw_frequency(force)
    sev = _load_raw_severity(force)
    df = _build_targets(freq, sev)

    for col in CATEGORICAL_COLS:
        df[col] = df[col].astype("category")

    df.to_parquet(PROCESSED_PATH, index=False)
    print(f"Processed dataset saved → {PROCESSED_PATH} ({len(df):,} rows)")
    return df


# ---------------------------------------------------------------------------
# Split creation
# ---------------------------------------------------------------------------


def create_splits(
    df: pd.DataFrame | None = None,
    test_size: float = 0.20,
    n_cv_folds: int = 5,
    random_state: int = 42,
    force: bool = False,
) -> tuple[np.ndarray, list[tuple[np.ndarray, np.ndarray]]]:
    """
    Create and persist holdout split + CV fold indices.

    Both are stratified on claim_flag so every split has ~3.5-4% claim rate.

    Persisted files:
        data/insurance/holdout_idx.npy   — indices into the full df (integer positions)
        data/insurance/cv_folds.pkl      — list of (train_idx, val_idx) tuples,
                                           indices relative to the DEVELOPMENT set

    Returns:
        holdout_idx   — 1-D int array of holdout row positions in full df
        cv_folds      — list of 5 (train_idx, val_idx) pairs for the dev set
    """
    if not force and HOLDOUT_IDX_PATH.exists() and CV_FOLDS_PATH.exists():
        holdout_idx = np.load(HOLDOUT_IDX_PATH)
        with open(CV_FOLDS_PATH, "rb") as fh:
            cv_folds = pickle.load(fh)
        return holdout_idx, cv_folds

    if df is None:
        df = load_processed()

    claim_flag = df["claim_flag"].values
    all_idx = np.arange(len(df))

    dev_idx, holdout_idx = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=random_state,
        stratify=claim_flag,
    )

    claim_flag_dev = claim_flag[dev_idx]
    skf = StratifiedKFold(n_splits=n_cv_folds, shuffle=True, random_state=random_state)
    # Fold indices are 0-based within dev_idx (i.e., relative to the dev set)
    cv_folds = list(skf.split(dev_idx, claim_flag_dev))

    np.save(HOLDOUT_IDX_PATH, holdout_idx)
    with open(CV_FOLDS_PATH, "wb") as fh:
        pickle.dump(cv_folds, fh)

    print(
        f"Splits created — holdout: {len(holdout_idx):,} "
        f"({len(holdout_idx) / len(df):.1%}) | "
        f"dev: {len(dev_idx):,} | "
        f"CV: {n_cv_folds}×{len(dev_idx) // n_cv_folds:,}"
    )
    return holdout_idx, cv_folds


# ---------------------------------------------------------------------------
# Accessor: development / holdout DataFrames
# ---------------------------------------------------------------------------


def get_dev_holdout(
    df: pd.DataFrame | None = None,
    force_splits: bool = False,
) -> dict:
    """
    Return development and holdout DataFrames plus all targets and exposure.

    Usage:
        splits = get_dev_holdout()
        X_dev            = splits["X_dev"]
        y_pure_premium   = splits["y_dev"]["pure_premium"]
        exposure_dev     = splits["exposure_dev"]
        cv_folds         = splits["cv_folds"]   # use these for every experiment

    Returned keys:
        df_dev, df_holdout          — full DataFrames (features + targets)
        X_dev, X_holdout            — feature-only (category dtypes preserved)
        y_dev, y_holdout            — DataFrames with all four target columns
        exposure_dev, exposure_holdout — Exposure Series
        holdout_idx                 — int array (positions in full df)
        cv_folds                    — list of (train_idx, val_idx) for dev set
        feature_cols                — ordered list of feature column names
    """
    if df is None:
        df = load_processed()

    holdout_idx, cv_folds = create_splits(df, force=force_splits)
    dev_idx = np.setdiff1d(np.arange(len(df)), holdout_idx)

    df_dev = df.iloc[dev_idx].reset_index(drop=True)
    df_holdout = df.iloc[holdout_idx].reset_index(drop=True)

    feature_cols = CATEGORICAL_COLS + NUMERIC_COLS

    return {
        "df_dev": df_dev,
        "df_holdout": df_holdout,
        "X_dev": df_dev[feature_cols],
        "X_holdout": df_holdout[feature_cols],
        "y_dev": df_dev[TARGET_COLS],
        "y_holdout": df_holdout[TARGET_COLS],
        "exposure_dev": df_dev[EXPOSURE_COL],
        "exposure_holdout": df_holdout[EXPOSURE_COL],
        "holdout_idx": holdout_idx,
        "cv_folds": cv_folds,
        "feature_cols": feature_cols,
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def describe_dataset(df: pd.DataFrame | None = None) -> None:
    if df is None:
        df = load_processed()
    n = len(df)
    n_claims = df["claim_flag"].sum()
    print(f"Shape:           {df.shape}")
    print(f"Claim rate:      {n_claims / n:.2%} ({n_claims:,} policies with claims)")
    print(f"Mean exposure:   {df['Exposure'].mean():.3f}")
    print(f"\nClaimNb distribution:\n{df['claim_freq'].value_counts().sort_index()}")
    pp = df.loc[df["pure_premium"] > 0, "pure_premium"]
    print(
        f"\nPure premium (non-zero): mean={pp.mean():.1f}, p50={pp.median():.1f}, p95={pp.quantile(0.95):.1f}"
    )
    sev = df["severity"].dropna()
    print(
        f"Severity (claims>0):     mean={sev.mean():.1f}, p50={sev.median():.1f}, p95={sev.quantile(0.95):.1f}"
    )
    print(f"\nMissing: {df.isnull().sum()[df.isnull().sum() > 0].to_dict()}")


if __name__ == "__main__":
    df = load_processed()
    describe_dataset(df)
    splits = get_dev_holdout(df)
    print(f"\nDev: {len(splits['df_dev']):,} | Holdout: {len(splits['df_holdout']):,}")
    print(f"CV folds: {len(splits['cv_folds'])}")
