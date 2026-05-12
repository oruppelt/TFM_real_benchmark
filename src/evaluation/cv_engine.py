"""
Cross-validation orchestration for the insurance experiment.

Implements three levels of evaluation:
    Level 1 — 5-fold out-of-fold CV on the development set
    Level 2 — Holdout evaluation after retraining on full dev set
    Level 3 — Auction analysis (see auction.py)

Key design decisions:
    - The same CV fold indices (from load_insurance.create_splits) are used
      for every model and every feature set. This makes fold-level comparisons
      statistically valid.
    - Feature pipelines are cloned and re-fit per fold to prevent any leakage
      from validation fold information into the encoding (critical for target
      encoding in EngineeredFeaturePipeline).
    - TabPFN recalibration uses Option B (nested inner-fold OOF):
        For each outer training fold, run inner K-fold CV on TabPFN.
        Collect OOF predictions on the inner training data.
        Fit recalibrator on those OOF preds.
        Apply to outer validation fold.
        This is more data-efficient than a calibration split (Option A) and
        avoids wasting any of TabPFN's limited training budget.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from tqdm import tqdm

from src.metrics import tweedie_deviance, poisson_deviance, gamma_deviance, gini_coefficient, rmse, mae

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "cv"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

ApproachLabel = str   # "freqsev" | "tweedie" | "hurdle"


# ---------------------------------------------------------------------------
# Core CV runner
# ---------------------------------------------------------------------------

def run_cv(
    model_factory,
    feature_pipeline_factory,
    X_dev: pd.DataFrame,
    y_dev: pd.DataFrame,          # DataFrame with all target columns
    exposure_dev: pd.Series,
    cv_folds: list[tuple[np.ndarray, np.ndarray]],
    approach: ApproachLabel,
    model_name: str,
    features_label: str,
    tabpfn_n_inner_folds: int = 3,
    tabpfn_recalibrate: bool = True,
    save: bool = True,
) -> dict:
    """
    Run 5-fold outer CV, returning OOF predictions and per-fold metrics.

    For each fold:
        1. Split dev set into train/val using saved fold indices.
        2. Clone and fit feature pipeline on train fold only.
        3. Fit model on transformed train fold.
        4. Predict on transformed val fold.
        5. Evaluate and store.

    For TabPFN models (detected via hasattr(model, 'n_train_max')):
        - Subsample the training fold to model.n_train_max.
        - Apply Option B recalibration using nested inner-fold OOF predictions.

    Args:
        model_factory:             callable() → new unfitted model instance
        feature_pipeline_factory:  callable() → new unfitted pipeline instance
        X_dev:                     feature DataFrame for development set
        y_dev:                     target DataFrame (claim_flag, claim_freq, severity, pure_premium)
        exposure_dev:              Exposure Series for development set
        cv_folds:                  from load_insurance.create_splits()
        approach:                  "freqsev", "tweedie", or "hurdle"
        model_name:                string label for saving results
        features_label:            "raw" or "engineered"
        tabpfn_n_inner_folds:      inner folds for TabPFN recalibration (Option B)
        save:                      whether to save OOF predictions and metrics to disk

    Returns:
        dict with keys:
            oof_predictions    — full OOF predictions array (len = len(X_dev))
            fold_metrics       — list of 5 metric dicts
            mean_metrics       — mean across folds
            std_metrics        — std across folds
    """
    oof_preds = np.full(len(X_dev), np.nan)
    fold_metrics = []
    exp_arr = exposure_dev.values

    for fold_idx, (train_idx, val_idx) in enumerate(tqdm(cv_folds, desc=f"{model_name}/{features_label}/{approach}")):
        # --- Data ---
        X_tr_raw = X_dev.iloc[train_idx]
        X_val_raw = X_dev.iloc[val_idx]
        exp_tr = exp_arr[train_idx]
        exp_val = exp_arr[val_idx]

        # Select targets for this approach
        y_tr, y_val = _select_targets(y_dev, exposure_dev, train_idx, val_idx, approach)

        # --- Feature pipeline (clone + fit on train fold only!) ---
        pipe = feature_pipeline_factory()
        # Engineered pipeline needs the target for target encoding
        target_for_encoding = _target_for_encoding(y_dev, train_idx, approach)
        if hasattr(pipe, "_target_encoders"):
            X_tr = pipe.fit_transform(X_tr_raw, y=target_for_encoding)
        else:
            X_tr = pipe.fit_transform(X_tr_raw)
        X_val = pipe.transform(X_val_raw)

        # --- Model fit ---
        model = model_factory()
        is_tabpfn = hasattr(model, "n_train_max")

        if is_tabpfn and tabpfn_recalibrate:
            preds_val, recal_info = _fit_predict_tabpfn_with_recalibration(
                model=model,
                pipe=pipe,
                X_tr=X_tr,
                y_tr=y_tr,
                exp_tr=exp_tr,
                X_val=X_val,
                exp_val=exp_val,
                n_inner_folds=tabpfn_n_inner_folds,
            )
        else:
            model.fit(X_tr, y_tr, exposure=exp_tr)
            preds_val = model.predict(X_val, exposure=exp_val)
            recal_info = {}

        oof_preds[val_idx] = preds_val

        # --- Metrics ---
        y_val_pp = y_dev["pure_premium"].values[val_idx]
        metrics = _evaluate_predictions(y_val_pp, preds_val, exp_val)
        metrics["fold"] = fold_idx
        metrics.update(recal_info)
        fold_metrics.append(metrics)

        print(
            f"  Fold {fold_idx}: "
            f"tweedie={metrics['tweedie_dev_1.5']:.4f} "
            f"gini={metrics['gini']:.4f}"
        )

    # Aggregate
    if not fold_metrics:
        raise ValueError("No CV folds completed successfully")
    metric_keys = [k for k in fold_metrics[0] if k not in ("fold",) and isinstance(fold_metrics[0][k], float)]
    mean_m = {k: float(np.mean([f[k] for f in fold_metrics])) for k in metric_keys}
    std_m  = {k: float(np.std([f[k] for f in fold_metrics]))  for k in metric_keys}

    result = {
        "oof_predictions": oof_preds,
        "fold_metrics":    fold_metrics,
        "mean_metrics":    mean_m,
        "std_metrics":     std_m,
    }

    if save:
        _save_cv_results(result, model_name, features_label, approach)

    return result


# ---------------------------------------------------------------------------
# TabPFN recalibration — Option B (nested inner-fold OOF)
# ---------------------------------------------------------------------------

def _fit_predict_tabpfn_with_recalibration(
    model,
    pipe,
    X_tr: np.ndarray,
    y_tr: np.ndarray,
    exp_tr: np.ndarray,
    X_val: np.ndarray,
    exp_val: np.ndarray,
    n_inner_folds: int = 3,
) -> tuple[np.ndarray, dict]:
    """
    Fit TabPFN with Option B recalibration.

    Option B — nested inner-fold OOF approach (documented choice, see module docstring):
        1. Run inner K-fold CV on the outer training data.
           Each inner fold: fit TabPFN on inner train, predict inner val.
        2. Collect OOF predictions covering all outer training rows.
        3. Fit TweedieRecalibrator on (y_train_outer, oof_preds_inner).
        4. Fit final TabPFN on all outer training data.
        5. Apply recalibrator to final TabPFN's predictions on outer val.

    This ensures:
        - The recalibrator sees no validation fold data (no leakage).
        - All outer training data is used for both TabPFN (step 4) and
          recalibrator (step 3), maximising data efficiency.

    Returns:
        (calibrated_val_preds, info_dict)
    """
    from src.models.tabpfn_wrapper import TweedieRecalibrator, IsotonicRecalibrator

    inner_cv = KFold(n_splits=n_inner_folds, shuffle=True, random_state=42)
    inner_oof = np.full(len(X_tr), np.nan)

    # Step 1-2: inner-fold OOF predictions
    for i_tr, i_val in inner_cv.split(X_tr):
        inner_model = model.clone()
        inner_model.fit(X_tr[i_tr], y_tr[i_tr], exposure=exp_tr[i_tr])
        inner_oof[i_val] = inner_model.predict(X_tr[i_val], exposure=exp_tr[i_val])

    # Drop NaN rows (shouldn't happen, but guard)
    valid = ~np.isnan(inner_oof)

    # Step 3: fit recalibrators on inner OOF
    tweedie_recal = TweedieRecalibrator()
    isotonic_recal = IsotonicRecalibrator()
    tweedie_recal.fit(y_tr[valid], inner_oof[valid])
    isotonic_recal.fit(y_tr[valid], inner_oof[valid])

    # Step 4: fit final TabPFN on all outer training data
    model.fit(X_tr, y_tr, exposure=exp_tr)

    # Step 5: predict on outer val + recalibrate
    raw_preds = model.predict(X_val, exposure=exp_val)
    tweedie_preds = tweedie_recal.predict(raw_preds)
    isotonic_preds = isotonic_recal.predict(raw_preds)

    info = {
        "n_clipped_raw": int(model._n_clipped),
        "recal_method": "tweedie_glm",  # default; isotonic saved separately
    }

    # Return Tweedie-recalibrated as the primary prediction
    return tweedie_preds, info


# ---------------------------------------------------------------------------
# Hurdle model CV runner
# ---------------------------------------------------------------------------

def run_cv_hurdle(
    stage1_factory,
    stage2_factory,
    feature_pipeline_factory,
    X_dev: pd.DataFrame,
    y_dev: pd.DataFrame,
    exposure_dev: pd.Series,
    cv_folds: list[tuple[np.ndarray, np.ndarray]],
    model_name: str,
    features_label: str,
    tabpfn_n_inner_folds: int = 3,
    save: bool = True,
) -> dict:
    """
    Full 2-stage hurdle model CV.

    Stage 1 — Classifier: P(claim > 0)
        Fitted on the full training fold. Target = claim_flag (0/1).

    Stage 2 — Regressor: E[pure_premium | claim > 0]
        Fitted on the claims-only subset of the training fold.
        Target = pure_premium (ClaimAmount / Exposure) restricted to rows where
        ClaimNb > 0 AND ClaimAmount > 0 (i.e. severity is not NaN).
        Using pure_premium (already exposure-normalised) as the stage 2 target
        keeps the combination simple:

            final_pred = P(claim > 0) × E[pure_premium | claim > 0]

        This is mathematically exact under the hurdle factorisation:
            E[pp] = P(pp > 0) × E[pp | pp > 0]

        For GLMs (GammaGLM): exposure offset is included in stage 2.
        For TabPFN: exposure passed as a feature (no native offset support).

    Args:
        stage1_factory:           callable() → unfitted classifier (TabPFNWrapper task='classification',
                                  or any model with fit(X, y) / predict(X) → probabilities)
        stage2_factory:           callable() → unfitted regressor (GammaGLM, TabPFNWrapper task='regression')
        feature_pipeline_factory: callable() → unfitted feature pipeline
        model_name:               label for saving results (e.g. 'TabPFN_hurdle')
        features_label:           pipeline label (e.g. 'gbm')

    Returns same dict structure as run_cv:
        oof_predictions, fold_metrics, mean_metrics, std_metrics
    """
    oof_preds = np.full(len(X_dev), np.nan)
    fold_metrics = []
    exp_arr = exposure_dev.values

    for fold_idx, (train_idx, val_idx) in enumerate(
        tqdm(cv_folds, desc=f"{model_name}/{features_label}/hurdle")
    ):
        X_tr_raw = X_dev.iloc[train_idx]
        X_val_raw = X_dev.iloc[val_idx]
        exp_tr = exp_arr[train_idx]
        exp_val = exp_arr[val_idx]

        y_tr_flag = y_dev["claim_flag"].values[train_idx]
        y_val_flag = y_dev["claim_flag"].values[val_idx]
        y_tr_pp   = y_dev["pure_premium"].values[train_idx]
        y_val_pp  = y_dev["pure_premium"].values[val_idx]

        # Claims-only mask for stage 2 (severity is NaN where ClaimNb=0 or ClaimAmount=0)
        sev_mask_tr = ~np.isnan(y_dev["severity"].values[train_idx])

        # --- Feature pipeline (fit on train fold only) ---
        pipe = feature_pipeline_factory()
        target_for_enc = y_dev["pure_premium"].iloc[train_idx]
        if hasattr(pipe, "_target_encoders"):
            X_tr = pipe.fit_transform(X_tr_raw, y=target_for_enc)
        else:
            X_tr = pipe.fit_transform(X_tr_raw)
        X_val = pipe.transform(X_val_raw)

        # ---- Stage 1: classify claim occurrence ----
        # No recalibration for classification — Tweedie recalibration is regression-only.
        # TabPFN classification is used directly; its internal ensembling handles calibration.
        s1 = stage1_factory()
        s1.fit(X_tr, y_tr_flag, exposure=exp_tr)
        s1_preds_val = s1.predict(X_val, exposure=exp_val)

        # Clip stage 1 to valid probability range
        s1_preds_val = np.clip(s1_preds_val, 1e-6, 1.0 - 1e-6)

        # ---- Stage 2: regress pure_premium on claims-only ----
        X_tr_claims = X_tr[sev_mask_tr]
        y_tr_pp_claims = y_tr_pp[sev_mask_tr]
        exp_tr_claims = exp_tr[sev_mask_tr]

        # Clip extreme pure_premium values before fitting stage 2.
        # pure_premium = ClaimAmount / Exposure — tiny exposures (min 1e-6) can produce
        # astronomically large values that destabilise GammaGLM's IRLS weights.
        pp_cap = np.percentile(y_tr_pp_claims, 99)
        y_tr_pp_claims = np.clip(y_tr_pp_claims, 1e-10, pp_cap)

        s2 = stage2_factory()
        is_tabpfn_s2 = hasattr(s2, "n_train_max")

        if is_tabpfn_s2 and tabpfn_n_inner_folds >= 2:
            # TabPFN stage 2 with recalibration — needs at least 2 inner folds
            s2_preds_val, _ = _fit_predict_tabpfn_with_recalibration(
                model=s2, pipe=pipe,
                X_tr=X_tr_claims, y_tr=y_tr_pp_claims,
                exp_tr=exp_tr_claims, X_val=X_val, exp_val=exp_val,
                n_inner_folds=tabpfn_n_inner_folds,
            )
        else:
            # GLM stage 2, or TabPFN without recalibration (n_inner_folds < 2)
            s2.fit(X_tr_claims, y_tr_pp_claims, exposure=exp_tr_claims)
            s2_preds_val = s2.predict(X_val, exposure=exp_val)

        # ---- Combine ----
        # E[pp] = P(claim > 0) × E[pp | claim > 0]
        combined_preds = np.clip(s1_preds_val * s2_preds_val, 1e-10, None)
        oof_preds[val_idx] = combined_preds

        # ---- Metrics ----
        metrics = _evaluate_predictions(y_val_pp, combined_preds, exp_val)
        metrics["fold"] = fold_idx
        metrics["n_claims_tr"] = int(sev_mask_tr.sum())
        metrics["mean_s1_prob"] = float(s1_preds_val.mean())
        fold_metrics.append(metrics)

        print(
            f"  Fold {fold_idx}: "
            f"tweedie={metrics['tweedie_dev_1.5']:.4f} "
            f"gini={metrics['gini']:.4f} "
            f"claims_in_train={sev_mask_tr.sum():,}"
        )

    if not fold_metrics:
        raise ValueError("No CV folds completed successfully")
    metric_keys = [
        k for k in fold_metrics[0]
        if k not in ("fold",) and isinstance(fold_metrics[0][k], float)
    ]
    mean_m = {k: float(np.mean([f[k] for f in fold_metrics])) for k in metric_keys}
    std_m  = {k: float(np.std([f[k]  for f in fold_metrics])) for k in metric_keys}

    result = {
        "oof_predictions": oof_preds,
        "fold_metrics":    fold_metrics,
        "mean_metrics":    mean_m,
        "std_metrics":     std_m,
    }

    if save:
        _save_cv_results(result, model_name, features_label, "hurdle")

    return result


# ---------------------------------------------------------------------------
# Holdout evaluation
# ---------------------------------------------------------------------------

def holdout_evaluation(
    model_configs: list[dict],
    X_dev: pd.DataFrame,
    y_dev: pd.DataFrame,
    exposure_dev: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.DataFrame,
    exposure_holdout: pd.Series,
    tag: str = "baselines",
) -> pd.DataFrame:
    """
    Retrain all models on full dev set and evaluate on holdout.

    Args:
        model_configs: list of dicts, each with keys:
            model_factory, pipeline_factory, approach, model_name, features_label
        tag: filename tag so multiple runs don't overwrite each other.
             e.g. "baselines" → metrics_baselines.parquet
                  "tabpfn"    → metrics_tabpfn.parquet
             post2_analysis.ipynb merges all metrics_*.parquet files.

    Returns:
        DataFrame with one row per model config, columns = metrics
    """
    from pathlib import Path
    save_dir = Path(__file__).parent.parent.parent / "results" / "holdout"
    save_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    all_preds = {}

    for cfg in model_configs:
        name = f"{cfg['model_name']}_{cfg['features_label']}_{cfg['approach']}"
        print(f"Holdout eval: {name}")

        pipe = cfg["pipeline_factory"]()
        target_for_enc = _target_for_encoding(y_dev, np.arange(len(y_dev)), cfg["approach"])
        if hasattr(pipe, "_target_encoders"):
            X_dev_fe = pipe.fit_transform(X_dev, y=target_for_enc)
        else:
            X_dev_fe = pipe.fit_transform(X_dev)
        X_hold_fe = pipe.transform(X_holdout)

        y_tr, _ = _select_targets(y_dev, exposure_dev, np.arange(len(y_dev)), np.array([0]), cfg["approach"])

        model = cfg["model_factory"]()
        is_tabpfn = hasattr(model, "n_train_max")

        if is_tabpfn:
            preds, _ = _fit_predict_tabpfn_with_recalibration(
                model=model,
                pipe=pipe,
                X_tr=X_dev_fe,
                y_tr=y_tr,
                exp_tr=exposure_dev.values,
                X_val=X_hold_fe,
                exp_val=exposure_holdout.values,
            )
        else:
            model.fit(X_dev_fe, y_tr, exposure=exposure_dev.values)
            preds = model.predict(X_hold_fe, exposure=exposure_holdout.values)

        y_hold_pp = y_holdout["pure_premium"].values
        metrics = _evaluate_predictions(y_hold_pp, preds, exposure_holdout.values)
        metrics["model"] = name
        rows.append(metrics)
        all_preds[name] = preds

    results_df = pd.DataFrame(rows).set_index("model")
    results_df.to_parquet(save_dir / f"metrics_{tag}.parquet")
    pd.DataFrame(all_preds).to_parquet(save_dir / f"predictions_{tag}.parquet")
    return results_df


# ---------------------------------------------------------------------------
# Hurdle holdout evaluation
# ---------------------------------------------------------------------------

def holdout_evaluation_hurdle(
    stage1_factory,
    stage2_factory,
    feature_pipeline_factory,
    model_name: str,
    features_label: str,
    X_dev: pd.DataFrame,
    y_dev: pd.DataFrame,
    exposure_dev: pd.Series,
    X_holdout: pd.DataFrame,
    y_holdout: pd.DataFrame,
    exposure_holdout: pd.Series,
) -> tuple[np.ndarray, dict]:
    """
    Refit a 2-stage hurdle model on full dev set, predict on holdout.

    Returns (holdout_preds, metrics_dict) — does NOT save to disk.
    Caller merges into the predictions parquet alongside other models.
    """
    exp_dev_arr  = exposure_dev.values
    exp_hold_arr = exposure_holdout.values

    pipe = feature_pipeline_factory()
    if hasattr(pipe, "_target_encoders"):
        X_dev_fe  = pipe.fit_transform(X_dev, y=y_dev["pure_premium"])
    else:
        X_dev_fe  = pipe.fit_transform(X_dev)
    X_hold_fe = pipe.transform(X_holdout)

    y_dev_flag      = y_dev["claim_flag"].values
    y_dev_pp        = y_dev["pure_premium"].values
    sev_mask        = ~np.isnan(y_dev["severity"].values)

    # Stage 1 — classifier on full dev
    s1 = stage1_factory()
    s1.fit(X_dev_fe, y_dev_flag, exposure=exp_dev_arr)
    s1_preds = np.clip(s1.predict(X_hold_fe, exposure=exp_hold_arr), 1e-6, 1 - 1e-6)

    # Stage 2 — regressor on claims-only subset
    y_dev_pp_claims = np.clip(y_dev_pp[sev_mask], 1e-10, np.percentile(y_dev_pp[sev_mask], 99))
    s2 = stage2_factory()
    s2.fit(X_dev_fe[sev_mask], y_dev_pp_claims, exposure=exp_dev_arr[sev_mask])
    s2_preds = s2.predict(X_hold_fe, exposure=exp_hold_arr)

    combined = np.clip(s1_preds * s2_preds, 1e-10, None)
    metrics  = _evaluate_predictions(y_holdout["pure_premium"].values, combined, exp_hold_arr)
    metrics["model"] = f"{model_name}_{features_label}_hurdle"

    return combined, metrics


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_targets(
    y_dev: pd.DataFrame,
    exposure_dev: pd.Series,
    train_idx: np.ndarray,
    val_idx: np.ndarray,
    approach: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (y_train, y_val) appropriate for the modeling approach."""
    if approach == "tweedie":
        y = y_dev["pure_premium"].values
    elif approach == "freqsev":
        y = y_dev["claim_freq"].values
    elif approach == "hurdle":
        y = y_dev["claim_flag"].values
    else:
        raise ValueError(f"Unknown approach: {approach}")
    return y[train_idx], y[val_idx]


def _target_for_encoding(
    y_dev: pd.DataFrame,
    train_idx: np.ndarray,
    approach: str,
) -> pd.Series:
    """Return the Series to use for target encoding in the pipeline."""
    if approach in ("tweedie", "freqsev"):
        return y_dev["pure_premium"].iloc[train_idx]
    return y_dev["claim_flag"].iloc[train_idx]


def _evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    exposure: np.ndarray,
) -> dict[str, float]:
    """Compute the full insurance metric suite on pure premium predictions."""
    return {
        "tweedie_dev_1.5": tweedie_deviance(y_true, y_pred, power=1.5, sample_weight=exposure),
        "poisson_dev":     poisson_deviance(y_true, y_pred, sample_weight=exposure),
        "gini":            gini_coefficient(y_true, y_pred),
        "rmse":            rmse(y_true, y_pred, sample_weight=exposure),
        "mae":             mae(y_true, y_pred, sample_weight=exposure),
    }


def _save_cv_results(result: dict, model_name: str, features_label: str, approach: str) -> None:
    stem = f"{model_name}_{features_label}_{approach}"
    oof_df = pd.DataFrame({"oof_pred": result["oof_predictions"]})
    oof_df.to_parquet(RESULTS_DIR / f"{stem}_oof.parquet")
    with open(RESULTS_DIR / f"{stem}_fold_metrics.json", "w") as f:
        json.dump({"folds": result["fold_metrics"], "mean": result["mean_metrics"], "std": result["std_metrics"]}, f, indent=2)
