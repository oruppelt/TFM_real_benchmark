"""Launch LightGBM Optuna tuning for Post 3 (§6.1) — the deferred multi-hour job.

Tunes once per objective on the `full` feature set (§6.1: feature set must be the
only variable in the ablation, so hyperparameters are frozen across feature sets).
Validation uses 3 rolling-origin folds over the validation period. Writes
`results/tuning/lgbm_{objective}_full.json`, which the model notebooks load via
their `TUNED_PARAMS_PATH` knob.

    python scripts/run_tuning.py                 # full panel, point + quantile objectives (~hours)
    python scripts/run_tuning.py --core-slice    # faster smoke on the 100-series slice
    python scripts/run_tuning.py --objectives point

Nothing else in the pipeline depends on this having run — the notebooks fall back
to documented defaults (config/default.yaml `lgbm.defaults`) when no tuned JSON
exists.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from src.demand.config import load_config, resolve_path
from src.demand.data.load import build_panel, load_raw_data
from src.demand.data.splits import build_core_slice, build_main_splits
from src.demand.data.supervised_frame import build_supervised_frame
from src.demand.tuning.optuna_lgbm import tune

FEATURE_SET = "full"   # §6.1 — tune on `full` only
N_FOLDS = 3            # §5.1 — 3 rolling-origin validation folds


def _chunks(items: list, n: int) -> list[list]:
    """Split a list into `n` contiguous, roughly equal chunks (drops empties)."""
    return [c for c in np.array_split(np.array(items, dtype=object), n) if len(c)]


def build_folds(cfg, panel, raw, splits, series_filter):
    """Return a callable producing [(train_frame, val_frame)] x N_FOLDS.

    Every fold trains on the same training-origin grid and validates on one
    contiguous slice of the validation origins.
    """
    train_frame = build_supervised_frame(
        panel, origins=list(splits.training_origins), horizon=cfg["horizon_days"],
        feature_set=FEATURE_SET, config=cfg, mode="train",
        holidays=raw.holidays, stores=raw.stores, series_filter=series_filter,
    )
    val_chunks = _chunks(list(splits.validation_origins), N_FOLDS)
    val_frames = [
        build_supervised_frame(
            panel, origins=list(chunk), horizon=cfg["horizon_days"],
            feature_set=FEATURE_SET, config=cfg, mode="train",
            holidays=raw.holidays, stores=raw.stores, series_filter=series_filter,
        )
        for chunk in val_chunks
    ]

    def make_folds():
        return [(train_frame, vf) for vf in val_frames if not vf.empty]

    return make_folds


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="default.yaml")
    parser.add_argument("--objectives", nargs="+",
                        default=["point", "quantile"])
    parser.add_argument("--core-slice", action="store_true",
                        help="tune on the 100-series core slice (fast smoke, not §6.1 spec)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    splits = build_main_splits(cfg)
    raw = load_raw_data(cfg)
    panel = build_panel(raw, include_test=False)
    series_filter = (
        build_core_slice(panel, raw.stores, cfg, write_csv=False)
        if args.core_slice else None
    )
    print(f"tuning scope: {'core slice' if args.core_slice else 'full panel'} | "
          f"objectives={args.objectives}")

    make_folds = build_folds(cfg, panel, raw, splits, series_filter)
    out_root = resolve_path(cfg, "results_dir") / "tuning"
    for obj in args.objectives:
        out_path = out_root / f"lgbm_{obj}_{FEATURE_SET}.json"
        print(f"\n=== tuning {obj} -> {out_path} ===")
        best = tune(cfg, obj, make_folds, out_path, feature_set=FEATURE_SET)
        print(f"best value={best.get('_best_value')}  "
              f"trials={best.get('_n_completed_trials')}")


if __name__ == "__main__":
    main()
