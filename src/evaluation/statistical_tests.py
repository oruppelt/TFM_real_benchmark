"""
Statistical comparison of CV fold-level metrics.

With 5 folds, statistical power is limited — we acknowledge this.
CV provides supporting evidence and consistency checks; holdout provides
the headline point estimate.

Tests used:
    Paired t-test              — standard, assumes normality (fine for fold metrics)
    Wilcoxon signed-rank test  — non-parametric alternative, better for n=5
    We report both; the Wilcoxon is more appropriate for small n.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS_DIR = Path(__file__).parent.parent.parent / "results" / "cv"


def compare_models_cv(
    fold_metrics_a: list[dict],
    fold_metrics_b: list[dict],
    metric: str = "tweedie_dev_1.5",
    name_a: str = "model_a",
    name_b: str = "model_b",
) -> dict:
    """
    Paired statistical comparison of two models across CV folds.

    Args:
        fold_metrics_a: list of per-fold metric dicts for model A
        fold_metrics_b: list of per-fold metric dicts for model B
        metric:         which metric to compare (lower is better)
        name_a, name_b: labels for output

    Returns dict with:
        mean_a, mean_b     — mean metric across folds
        mean_diff          — mean_a - mean_b (negative = A is better)
        ci_95              — (lower, upper) 95% CI on the difference
        t_stat, t_pvalue   — paired t-test
        w_stat, w_pvalue   — Wilcoxon signed-rank test
        better_model       — which model has lower mean metric
        a_wins_n_folds     — how many folds model A has the lower metric
        interpretation     — plain-English summary
    """
    vals_a = np.array([f[metric] for f in fold_metrics_a])
    vals_b = np.array([f[metric] for f in fold_metrics_b])
    diffs  = vals_a - vals_b

    n = len(diffs)
    mean_diff = float(np.mean(diffs))
    se_diff   = float(np.std(diffs, ddof=1) / np.sqrt(n))
    ci_95 = (
        float(mean_diff - 1.96 * se_diff),
        float(mean_diff + 1.96 * se_diff),
    )

    t_stat, t_pval = stats.ttest_rel(vals_a, vals_b)
    try:
        w_stat, w_pval = stats.wilcoxon(diffs, zero_method="wilcox", alternative="two-sided")
    except ValueError:
        # Wilcoxon requires at least one non-zero difference
        w_stat, w_pval = float("nan"), float("nan")

    better = name_a if np.mean(vals_a) < np.mean(vals_b) else name_b
    a_wins = int(np.sum(vals_a < vals_b))

    # Plain-English summary
    sig = t_pval < 0.05
    interp = (
        f"{better} has lower mean {metric} "
        f"({np.mean(vals_a):.4f} vs {np.mean(vals_b):.4f}, "
        f"diff={mean_diff:+.4f}). "
        f"Paired t-test: {'significant' if sig else 'not significant'} "
        f"(p={t_pval:.3f}). "
        f"Note: with n={n} folds, statistical power is limited."
    )

    return {
        "metric":        metric,
        "name_a":        name_a,
        "name_b":        name_b,
        "mean_a":        float(np.mean(vals_a)),
        "mean_b":        float(np.mean(vals_b)),
        "std_a":         float(np.std(vals_a)),
        "std_b":         float(np.std(vals_b)),
        "mean_diff":     mean_diff,
        "ci_95":         ci_95,
        "t_stat":        float(t_stat),
        "t_pvalue":      float(t_pval),
        "w_stat":        float(w_stat),
        "w_pvalue":      float(w_pval),
        "better_model":  better,
        "a_wins_n_folds": a_wins,
        "interpretation": interp,
    }


def rank_models_by_metric(
    results: dict[str, list[dict]],
    metric: str = "tweedie_dev_1.5",
) -> pd.DataFrame:
    """
    Rank all models by a single metric across folds.

    Args:
        results: dict of {model_name: list_of_fold_metric_dicts}
        metric:  column to rank on

    Returns DataFrame with model as index, columns = mean, std, rank, fold_wins.
    """
    rows = []
    for name, folds in results.items():
        vals = [f[metric] for f in folds]
        rows.append({
            "model": name,
            "mean":  np.mean(vals),
            "std":   np.std(vals),
            "min":   np.min(vals),
            "max":   np.max(vals),
        })
    df = pd.DataFrame(rows).sort_values("mean")
    df["rank"] = range(1, len(df) + 1)
    return df.set_index("model")


def metric_disagreement_table(
    holdout_results: pd.DataFrame,
) -> pd.DataFrame:
    """
    Show how model rankings change depending on the metric used.

    For each metric column in holdout_results, rank models 1..N.
    Highlight where the same model ranks #1 on one metric but #3+ on another.

    This surfaces the "benchmarks lie" finding — accuracy on one metric
    does not imply accuracy on all metrics.

    Returns a DataFrame of ranks: models as rows, metrics as columns.
    """
    # Metrics where higher is better — rank descending so rank 1 = best
    HIGHER_IS_BETTER = {"gini"}

    metrics = [c for c in holdout_results.columns if c not in ("model",)]
    ranks = pd.DataFrame(index=holdout_results.index)
    for m in metrics:
        ascending = m not in HIGHER_IS_BETTER
        ranks[m] = holdout_results[m].rank(ascending=ascending).astype(int)

    # Flag models with high rank variance (disagreement across metrics)
    ranks["rank_std"] = ranks.std(axis=1).round(2)
    ranks["rank_range"] = ranks[metrics].max(axis=1) - ranks[metrics].min(axis=1)

    return ranks.sort_values("rank_std", ascending=False)


def run_all_comparisons(
    all_fold_metrics: dict[str, list[dict]],
    reference_model: str,
    metric: str = "tweedie_dev_1.5",
) -> list[dict]:
    """
    Compare all models against a reference model on a single metric.

    Args:
        all_fold_metrics: dict of {model_name: fold_metrics_list}
        reference_model:  name of the model to compare against (e.g., "XGBoost_engineered_tweedie")
        metric:           metric to use for comparison

    Returns list of comparison dicts, sorted by mean metric of the challenger.
    """
    ref = all_fold_metrics[reference_model]
    comparisons = []
    for name, folds in all_fold_metrics.items():
        if name == reference_model:
            continue
        cmp = compare_models_cv(folds, ref, metric=metric, name_a=name, name_b=reference_model)
        comparisons.append(cmp)

    comparisons.sort(key=lambda x: x["mean_a"])

    save_path = RESULTS_DIR / "statistical_comparison.json"
    with open(save_path, "w") as f:
        json.dump(comparisons, f, indent=2)
    print(f"Statistical comparisons saved to {save_path}")

    return comparisons
