"""
Competitive auction analysis — translating metric differences into business value.

Concept: simulate a competitive market where two models price the same portfolio.
For each policy, the insurer offering the lower premium wins the business but
inherits the actual losses. A model that underprices risky policies wins those
policies and loses money (adverse selection). A model that overprices safe
policies loses volume.

The model with better risk discrimination writes a more profitable book of
business — even if it doesn't "win" more policies by count.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "holdout"


def auction_analysis(
    y_true: np.ndarray,
    exposure: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    name_a: str,
    name_b: str,
) -> dict:
    """
    Head-to-head auction between two models on the holdout set.

    For each policy i:
        premium_a[i] = preds_a[i] * exposure[i]   # model A's charged premium
        premium_b[i] = preds_b[i] * exposure[i]   # model B's charged premium
        actual_loss[i] = y_true[i] * exposure[i]  # actual annualised claim cost

    Model A wins policy i if it quotes lower (premium_a[i] < premium_b[i]).

    Key outputs:
        1. Volume split — how many policies each model wins
        2. Loss ratio — total_losses / total_premium for each model's portfolio
        3. Total profit — total_premium - total_losses
        4. Adverse selection breakdown — average actual risk of each model's won policies

    Note on interpretation:
        Loss ratio < 1.0 means the model is profitable on the business it wins.
        A model can win 40% of policies but have a loss ratio of 0.6 (profitable).
        A model winning 60% of policies with loss ratio 1.1 is unprofitable.
        The auction winner is the model with better absolute profit, not more policies.
    """
    premium_a = preds_a * exposure
    premium_b = preds_b * exposure
    actual_loss = y_true * exposure

    a_wins = premium_a < premium_b
    b_wins = ~a_wins

    def portfolio_stats(wins_mask: np.ndarray, premium: np.ndarray, label: str) -> dict:
        total_premium = premium[wins_mask].sum()
        total_losses = actual_loss[wins_mask].sum()
        n_policies = wins_mask.sum()
        loss_ratio = total_losses / total_premium if total_premium > 0 else np.nan
        avg_risk = actual_loss[wins_mask].mean() if n_policies > 0 else np.nan
        return {
            f"{label}_n_policies": int(n_policies),
            f"{label}_pct_policies": float(n_policies / len(y_true)),
            f"{label}_total_premium": float(total_premium),
            f"{label}_total_losses": float(total_losses),
            f"{label}_loss_ratio": float(loss_ratio),
            f"{label}_profit": float(total_premium - total_losses),
            f"{label}_avg_actual_risk": float(avg_risk),
        }

    result = {
        "model_a": name_a,
        "model_b": name_b,
        "n_policies": len(y_true),
    }
    result.update(portfolio_stats(a_wins, premium_a, "a"))
    result.update(portfolio_stats(b_wins, premium_b, "b"))

    return result


def auction_by_risk_segment(
    y_true: np.ndarray,
    exposure: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    name_a: str,
    name_b: str,
    reference_preds: np.ndarray | None = None,
    n_deciles: int = 10,
) -> pd.DataFrame:
    """
    Break down the auction by risk decile.

    Policies are sorted into deciles by reference_preds (default: preds_a).
    Within each decile, we run the same auction logic.

    Shows WHERE each model wins/loses:
        - Does TabPFN underperform in the high-risk tail?
        - Does it win disproportionately in the low-risk majority?

    This reveals adverse selection patterns that aggregate metrics hide.

    Returns a DataFrame with one row per decile and columns for both models.
    """
    if reference_preds is None:
        reference_preds = preds_a

    decile_labels = pd.qcut(reference_preds, q=n_deciles, labels=False, duplicates='drop')

    rows = []
    for d in range(n_deciles):
        mask = decile_labels == d
        if mask.sum() == 0:
            continue
        seg = auction_analysis(
            y_true=y_true[mask],
            exposure=exposure[mask],
            preds_a=preds_a[mask],
            preds_b=preds_b[mask],
            name_a=name_a,
            name_b=name_b,
        )
        seg["decile"] = d + 1
        rows.append(seg)

    return pd.DataFrame(rows).set_index("decile")


def run_all_auctions(
    holdout_results: pd.DataFrame,
    y_holdout: np.ndarray,
    exposure_holdout: np.ndarray,
    predictions_path: Path | None = None,
) -> dict:
    """
    Run the four key auction matchups defined in the spec.

    Matchups:
        1. TabPFN (best config) vs XGBoost (tuned, engineered) — the headline
        2. TabPFN (best config) vs GLM Tweedie                 — TFM vs industry standard
        3. XGBoost (raw) vs XGBoost (engineered)               — value of feature engineering
        4. TabPFN (raw) vs TabPFN (engineered)                 — does engineering help TFMs?

    Args:
        holdout_results: DataFrame with model names as index (from cv_engine.holdout_evaluation)
        y_holdout:       pure_premium ground truth on holdout
        exposure_holdout: exposure on holdout
        predictions_path: path to holdout predictions parquet; defaults to results/holdout/predictions.parquet
    """
    if predictions_path is None:
        # Use merged file if available, fall back to baselines-only
        merged = RESULTS_DIR / "predictions_all.parquet"
        predictions_path = merged if merged.exists() else RESULTS_DIR / "predictions_baselines.parquet"

    preds_df = pd.read_parquet(predictions_path)
    print(f"Auction using predictions from: {predictions_path.name}")
    print(f"Available models: {list(preds_df.columns)}")

    def _find_col(pattern: str) -> str | None:
        """Find the first column matching a pattern string (case-insensitive)."""
        matches = [c for c in preds_df.columns if pattern.lower() in c.lower()]
        return matches[0] if matches else None

    # Best TabPFN = hurdle (best Gini) > Tweedie 10K fallback
    tabpfn_hurdle = _find_col("tabpfn_hurdle_cls_gamma") or _find_col("tabpfn_hurdle_full")
    tabpfn_tweedie = _find_col("tabpfn_10k_gbm") or _find_col("tabpfn_10k_raw")
    tabpfn_best = tabpfn_hurdle or tabpfn_tweedie

    lgbm_gbm_col = _find_col("lightgbm_gbm")
    lgbm_hurdle_col = _find_col("lightgbm_hurdle")
    xgb_gbm_col = _find_col("xgboost_gbm")
    xgb_raw_col = _find_col("xgboost_raw")
    xgb_hurdle_col = _find_col("xgboost_hurdle")
    glm_col = _find_col("glm_tweedie_engineered") or _find_col("glm_tweedie")
    tabpfn_raw = _find_col("tabpfn_10k_raw")
    tabpfn_gbm = _find_col("tabpfn_10k_gbm")

    matchups = []
    # 1. Headline: best TabPFN vs best GBM (Tweedie)
    if tabpfn_best and lgbm_gbm_col:
        matchups.append((tabpfn_best, lgbm_gbm_col, "TabPFN_best", "LightGBM_gbm"))
    # 2. TabPFN hurdle vs LightGBM hurdle — apples-to-apples hurdle comparison
    if tabpfn_hurdle and lgbm_hurdle_col:
        matchups.append((tabpfn_hurdle, lgbm_hurdle_col, "TabPFN_hurdle", "LightGBM_hurdle"))
    # 3. TabPFN vs GLM — TFM vs industry standard
    if tabpfn_best and glm_col:
        matchups.append((tabpfn_best, glm_col, "TabPFN_best", "GLM_Tweedie_eng"))
    # 4. Feature engineering value for GBMs
    if xgb_gbm_col and xgb_raw_col:
        matchups.append((xgb_gbm_col, xgb_raw_col, "XGBoost_gbm", "XGBoost_raw"))
    # 5. Hurdle vs direct Tweedie for TabPFN
    if tabpfn_hurdle and tabpfn_tweedie:
        matchups.append((tabpfn_hurdle, tabpfn_tweedie, "TabPFN_hurdle", "TabPFN_10K_tweedie"))
    # 6. GBM hurdle vs GBM Tweedie — does the hurdle factorisation help GBMs too?
    if lgbm_hurdle_col and lgbm_gbm_col:
        matchups.append((lgbm_hurdle_col, lgbm_gbm_col, "LightGBM_hurdle", "LightGBM_gbm"))

    results = {}
    for col_a, col_b, name_a, name_b in matchups:
        key = f"{name_a}_vs_{name_b}"
        results[key] = {
            "aggregate": auction_analysis(
                y_true=y_holdout,
                exposure=exposure_holdout,
                preds_a=preds_df[col_a].values,
                preds_b=preds_df[col_b].values,
                name_a=name_a,
                name_b=name_b,
            ),
            "by_decile": auction_by_risk_segment(
                y_true=y_holdout,
                exposure=exposure_holdout,
                preds_a=preds_df[col_a].values,
                preds_b=preds_df[col_b].values,
                name_a=name_a,
                name_b=name_b,
            ).to_dict(),
        }
        print(
            f"{name_a} vs {name_b}: "
            f"A wins {results[key]['aggregate']['a_pct_policies']:.1%} of policies, "
            f"loss ratio {results[key]['aggregate']['a_loss_ratio']:.3f} vs "
            f"{results[key]['aggregate']['b_loss_ratio']:.3f}"
        )

    with open(RESULTS_DIR / "auction_results.json", "w") as f:
        json.dump(results, f, indent=2, default=str)

    return results
