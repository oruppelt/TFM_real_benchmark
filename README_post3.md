# tabfm-real-world

Experiments for the Substack series "Tabular Foundation Models in the Real
World." Post 2 covers insurance pricing (existing code in `src/data`,
`src/features`, `src/models`). Post 3 covers retail demand forecasting on the
Favorita dataset — see [post3-prd-demand-forecasting.md](post3-prd-demand-forecasting.md).


## Post 2 — insurance pricing

The original Post 2 modules (`src/data/load_insurance.py`,
`src/features/insurance_features.py`, `src/models/{gbm,glm,tabpfn_wrapper}.py`,
etc.) are unchanged and continue to drive the notebooks under `notebooks/`.


The Post 3 non-TabPFN pipeline lives under `src/demand/`. TabPFN-TS and
TabPFN 2.5 adapters are intentionally absent (handled separately on GPU).

## Post 3 — reproducing the pipeline

### 1. Data

Place the Favorita CSVs under `data/favorita_demand/` (already present in this
repo). The loader expects: `train.csv`, `test.csv`, `stores.csv`, `oil.csv`,
`transactions.csv`, `holidays_events.csv`, `sample_submission.csv`.

### 2. Kaggle sanity check (run first — this is the gate, §11)

```bash
python -m src.demand.sanity_check.kaggle_pipeline --config default.yaml
```

Outputs:
- `results/sanity_check/submission.csv` — submit to Kaggle
- `results/sanity_check/metrics.json` — local RMSLE and row counts

Expected local RMSLE range: 0.45–0.55. A local RMSLE below 0.30 means
target leakage; above 0.70 means a feature or training bug. If the local
number is in range but the Kaggle public LB differs by more than 0.15,
investigate distribution drift between local holdout and Kaggle test.

### 3. Leakage unit tests

```bash
pytest tests/test_leakage.py
```

These must pass before any main experiment runs (§4.4).

### 4. Main experiment building blocks

Every module exposes a small, typed entry point. Typical usage:

```python
from src.demand.config import load_config
from src.demand.data.load import load_raw_data, build_panel
from src.demand.data.splits import build_main_splits, build_core_slice
from src.demand.data.supervised_frame import build_supervised_frame
from src.demand.models import lgbm_tweedie, lgbm_quantile
from src.demand.evaluation import metrics, newsvendor, statistical_tests, plots

cfg = load_config()
raw = load_raw_data(cfg)
panel = build_panel(raw, include_test=False)
splits = build_main_splits(cfg)

train_frame = build_supervised_frame(
    panel, origins=splits.training_origins, horizon=cfg["horizon_days"],
    feature_set="full", config=cfg, mode="train",
    holidays=raw.holidays, stores=raw.stores,
)
val_frame = build_supervised_frame(
    panel, origins=splits.validation_origins, horizon=cfg["horizon_days"],
    feature_set="full", config=cfg, mode="train",
    holidays=raw.holidays, stores=raw.stores,
)

artifact = lgbm_tweedie.train(train_frame, val_frame, "full", cfg)
```

### 5. Directory layout

```
config/
  default.yaml               # main experiment config
  sensitivity_*.yaml         # cost-ratio variants (§8.5)
src/demand/
  config.py                  # YAML loader w/ extends:
  data/
    load.py                  # raw CSVs → merged panel
    holidays.py              # transfer-resolved holiday calendar
    features.py              # 5 nested feature levels (§4.3)
    supervised_frame.py      # (series, origin, h) direct-multi-step rows
    splits.py                # main / drift / cold-start / core-slice splits
    leakage_tests.py         # runtime leakage assertions
  models/
    lgbm_base.py             # shared LightGBM train/predict plumbing
    lgbm_tweedie.py          # primary production baseline
    lgbm_gaussian.py         # point forecast, standard loss
    lgbm_quantile.py         # p10/p50/p90 triple
  tuning/optuna_lgbm.py      # 50-trial Optuna per objective on `full`
  evaluation/
    metrics.py               # MAE, RMSE, sMAPE, MASE, bias, pinball, coverage
    newsvendor.py            # inventory simulation (§8)
    statistical_tests.py     # paired Wilcoxon (§7.3)
    plots.py                 # ablation, calibration, cost, coldstart, rankings
  deployment/
    throughput.py            # timing + 2.5M-forecast SLA extrapolation
    coldstart.py             # 7/28/90-day history experiment (§5.3)
    drift.py                 # earthquake A/B variants (§5.2)
  interpret/shap_lgbm.py     # TreeSHAP — global, top-5 profiles, local
  sanity_check/kaggle_pipeline.py  # standalone Kaggle-split validator
results/
  sanity_check/              # submission.csv + metrics.json
  figures/                   # all plots as PNG
tests/test_leakage.py        # leakage unit tests (§4.4)
```

### 6. TabPFN pieces

TabPFN-TS and TabPFN 2.5 code paths are deliberately out of scope here; they
are expected to live in a sibling module and share the `build_features` /
`build_supervised_frame` contracts so the fairness regime in §6.0 holds.
