"""Discovery + loading helpers for versioned TabPFN experiments.

Each TabPFN run (``post2_tabpfn.ipynb``) writes its artifacts under
``results/post2`` in a per-experiment subdirectory named after
``EXPERIMENT_ID`` — e.g. ``results/post2/cv/tabpfn_v2.6/``,
``results/post2/holdout/tabpfn_v3.0/``.  Baseline
artifacts (GLM / XGBoost / LightGBM, written by ``post2_baselines.ipynb``) are
version-independent and stay at the flat top level within ``results/post2``.

These helpers auto-discover *every* experiment subdirectory and load all of
them together, so ``post2_analysis.ipynb`` / ``post2_supplementary.ipynb`` can
compare TabPFN library versions side by side.  Models that come from an
experiment subdir are tagged ``<model>@<experiment_id>`` (e.g.
``TabPFN_10K_gbm_tweedie@tabpfn_v2.6``); baseline models keep their plain names.

Directory layout assumed::

    results/post2/cv/                 GLM_*/XGBoost_*/LightGBM_*  (baseline, flat)
    results/post2/cv/<exp>/           TabPFN_*                    (per experiment)
    results/post2/holdout/            metrics_baselines.parquet, predictions_baselines.parquet
    results/post2/holdout/<exp>/      metrics_tabpfn.parquet, predictions_tabpfn.parquet,
                                      predictions_all.parquet, auction_results.json
    results/post2/deployment/         benchmarks_baselines.parquet
    results/post2/deployment/<exp>/   benchmarks_tabpfn.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

# Separates a model name from its experiment id in tagged keys/columns.
EXP_SEP = "@"


# ── Tagging helpers ────────────────────────────────────────────────────────────

def tag(model: str, experiment: str) -> str:
    """Tag a model name with its experiment id: ``TabPFN_10K_gbm_tweedie@tabpfn_v2.6``."""
    return f"{model}{EXP_SEP}{experiment}"


def split_tag(name: str) -> tuple[str, str | None]:
    """Inverse of :func:`tag`. Returns ``(model, experiment_or_None)``."""
    if EXP_SEP in name:
        model, experiment = name.rsplit(EXP_SEP, 1)
        return model, experiment
    return name, None


# ── Discovery ──────────────────────────────────────────────────────────────────

def _results_root(project_root) -> Path:
    return Path(project_root) / "results" / "post2"


def discover_experiments(project_root, category: str = "cv") -> list[str]:
    """Return sorted experiment ids — the subdirectories under ``results/post2/<category>/``.

    Any directory (as opposed to a file) directly under ``results/post2/<category>/`` is
    treated as an experiment.  Use ``category="cv"`` (the default) for the
    canonical list; all categories should agree if every run completed.
    """
    base = _results_root(project_root) / category
    if not base.exists():
        return []
    return sorted(d.name for d in base.iterdir() if d.is_dir())


# ── CV fold metrics ────────────────────────────────────────────────────────────

def _load_fold_metrics_json(path: Path) -> dict:
    with open(path) as f:
        data = json.load(f)
    return {
        "fold_metrics": data["folds"],
        "mean_metrics": data["mean"],
        "std_metrics":  data["std"],
    }


def load_all_cv(project_root) -> dict[str, dict]:
    """Load every ``*_fold_metrics.json`` under ``results/post2/cv/``.

    Baseline results (flat ``results/post2/cv/*.json``) are keyed by their plain stem,
    e.g. ``LightGBM_gbm_tweedie``.  Experiment results (``results/post2/cv/<exp>/*.json``)
    are keyed ``<stem>@<exp>``, e.g. ``TabPFN_10K_gbm_tweedie@tabpfn_v2.6``.

    Returns ``dict[key] -> {'fold_metrics', 'mean_metrics', 'std_metrics'}``.
    """
    cv_root = _results_root(project_root) / "cv"
    out: dict[str, dict] = {}

    # Flat baseline results.
    for path in sorted(cv_root.glob("*_fold_metrics.json")):
        stem = path.name.replace("_fold_metrics.json", "")
        out[stem] = _load_fold_metrics_json(path)

    # Per-experiment results.
    for exp in discover_experiments(project_root, "cv"):
        for path in sorted((cv_root / exp).glob("*_fold_metrics.json")):
            stem = path.name.replace("_fold_metrics.json", "")
            out[tag(stem, exp)] = _load_fold_metrics_json(path)

    return out


def load_oof(project_root, stem: str, experiment: str | None = None) -> pd.Series:
    """Load OOF CV predictions for one result.

    ``stem``        e.g. ``'TabPFN_10K_raw_tweedie'``.
    ``experiment``  experiment id, or ``None`` to read the flat baseline directory.
    """
    cv_root = _results_root(project_root) / "cv"
    base = cv_root if experiment is None else cv_root / experiment
    return pd.read_parquet(base / f"{stem}_oof.parquet")["oof_pred"]


# ── Holdout metrics ────────────────────────────────────────────────────────────

def load_all_holdout_metrics(project_root) -> pd.DataFrame:
    """Concatenate all holdout metrics — baseline (flat) + every experiment subdir.

    Baseline rows keep their plain model name; experiment rows are re-indexed
    ``<model>@<exp>``.  Returns a DataFrame indexed by (tagged) model name.
    """
    holdout_root = _results_root(project_root) / "holdout"
    frames: list[pd.DataFrame] = []

    # Flat baseline metrics (metrics_baselines.parquet, and any other flat files).
    for path in sorted(holdout_root.glob("metrics_*.parquet")):
        frames.append(pd.read_parquet(path))

    # Per-experiment metrics.
    for exp in discover_experiments(project_root, "holdout"):
        for path in sorted((holdout_root / exp).glob("metrics_*.parquet")):
            df = pd.read_parquet(path)
            frames.append(df.rename(index=lambda m: tag(m, exp)))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)


# ── Holdout predictions ────────────────────────────────────────────────────────

def load_all_holdout_predictions(project_root) -> pd.DataFrame:
    """Build a single holdout-predictions frame.

    Columns = baseline model predictions (plain names) + every experiment's
    TabPFN predictions, tagged ``<model>@<exp>``.  Reads
    ``results/post2/holdout/predictions_baselines.parquet`` and each
    ``results/post2/holdout/<exp>/predictions_tabpfn.parquet``.
    """
    holdout_root = _results_root(project_root) / "holdout"
    cols: dict[str, np.ndarray] = {}

    # Flat baseline predictions.
    for path in sorted(holdout_root.glob("predictions_*.parquet")):
        df = pd.read_parquet(path)
        for c in df.columns:
            cols[c] = df[c].values

    # Per-experiment TabPFN predictions.
    for exp in discover_experiments(project_root, "holdout"):
        path = holdout_root / exp / "predictions_tabpfn.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            for c in df.columns:
                cols[tag(c, exp)] = df[c].values

    return pd.DataFrame(cols)


def write_predictions_all(project_root, preds_df: pd.DataFrame) -> Path:
    """Write the consolidated holdout predictions to ``results/post2/holdout/predictions_all.parquet``.

    ``run_all_auctions`` reads this file by default.  Returns the written path.
    """
    out_path = _results_root(project_root) / "holdout" / "predictions_all.parquet"
    preds_df.to_parquet(out_path)
    return out_path


# ── Deployment benchmarks ──────────────────────────────────────────────────────

def load_all_benchmarks(project_root) -> pd.DataFrame:
    """Concatenate deployment benchmarks — baseline (flat) + every experiment subdir.

    Baseline rows keep their plain model name; experiment rows are re-indexed
    ``<model>@<exp>``.
    """
    deploy_root = _results_root(project_root) / "deployment"
    frames: list[pd.DataFrame] = []

    baseline = deploy_root / "benchmarks_baselines.parquet"
    if baseline.exists():
        frames.append(pd.read_parquet(baseline))

    for exp in discover_experiments(project_root, "deployment"):
        path = deploy_root / exp / "benchmarks_tabpfn.parquet"
        if path.exists():
            df = pd.read_parquet(path)
            frames.append(df.rename(index=lambda m: tag(m, exp)))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames)
