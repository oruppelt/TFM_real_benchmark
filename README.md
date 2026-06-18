# tabfm-real-world

Experiments for the Substack series "Tabular Foundation Models in the Real
World." Post 2 covers insurance pricing (existing code in `src/data`,
`src/features`, `src/models`). Post 3 covers retail demand forecasting on the
Favorita dataset.


## Post 2 — insurance pricing

Post 2 uses the freMTPL2 insurance dataset from OpenML:

- `freMTPL2freq` — OpenML id `41214`
- `freMTPL2sev` — OpenML id `41215`

The data is downloaded automatically by `src/data/load_insurance.py` and cached
under `data/insurance/` as raw parquet files, `processed.parquet`, the holdout
split, and CV folds.

### Setup

```bash
uv sync --extra dev
uv run jupyter lab
```

### Run Order

Run these notebooks from top to bottom:

1. `notebooks/insurance_profile.ipynb` — download/profile the data and create splits.
2. `notebooks/post2_baselines.ipynb` — run GLM, XGBoost, and LightGBM baselines.
3. `notebooks/post2_tabpfn.ipynb` — run TabPFN experiments.
4. `notebooks/recalibration_analysis.ipynb` — optional recalibration checks.
5. `notebooks/post2_analysis.ipynb` — combine results and build main figures.
6. `notebooks/post2_tabpfn_interpretability.ipynb` — TabPFN interpretation figures.
7. `notebooks/post2_supplementary.ipynb` — optional supplementary tables/figures.


## Post 3 — retail demand forecasting (Favorita)

Spec: [`post3-prd-demand-forecasting.md`](post3-prd-demand-forecasting.md).
The experiment benchmarks the production incumbent against a tabular foundation
model on the Kaggle "Store Sales — Time Series Forecasting" data:

- **LightGBM** — two objectives, both fit on a `log1p` target (demand is
  multiplicative): a **point** model (`objective=regression`, back-transformed
  with `expm1`, targets the conditional median → matches the MAE/WAPE headline)
  and a **quantile** model (pinball at p10/p50/p90, for the §8 newsvendor decision).
- **TabPFN-TS** — the foundation model, run separately on GPU (see below).

All implementation lives in `src/demand/`; config in `config/`. The non-TabPFN
pipeline is fully runnable on CPU; the TabPFN-TS leg runs on GPU. Every comparison
artifact carries an `approach`/`model_name` column, so TabPFN results drop in
later with no rework — until then they render as `_pending_` in the summary tables.

### 1. Setup

```bash
uv sync --extra dev
```

### 2. Get the data

The dataset is the Kaggle **Store Sales — Time Series Forecasting** competition
(Corporación Favorita, Ecuador). It is not redistributed in this repo — you must
download it from Kaggle (free account + one-time acceptance of the competition
rules required).

```bash
# Option A — Kaggle CLI (pip install kaggle; put kaggle.json in ~/.kaggle/)
kaggle competitions download -c store-sales-time-series-forecasting -p data/favorita_demand
cd data/favorita_demand && unzip -o store-sales-time-series-forecasting.zip && cd -
```

Or **Option B — manual**: download the zip from
<https://www.kaggle.com/competitions/store-sales-time-series-forecasting/data>
and unzip it into `data/favorita_demand/`.

Either way the loader ([`src/demand/data/load.py`](src/demand/data/load.py))
expects these six CSVs in `data/favorita_demand/`:

```
train.csv  test.csv  stores.csv  oil.csv  transactions.csv  holidays_events.csv
```

The data path is configurable via `paths.data_dir` in
[`config/default.yaml`](config/default.yaml).

### 3. Gate checks (run once)

```bash
uv run python -m pytest tests/ -q                          # §4.4 leakage / fairness
uv run python -m src.demand.sanity_check.kaggle_pipeline   # §11 data + feature gate
```

The sanity check trains a direct multi-step LightGBM (L2 on a `log1p` target —
the RMSLE-native setup) on the canonical Kaggle split and writes
`results/sanity_check/{submission.csv,metrics.json}`:

- The local holdout is built at `as_of=2017-07-30`, so the July 31-August 15
  validation window has the same missing future-lag pattern as the true Kaggle
  test window.
- The model uses the same direct supervised framing as the main benchmark:
  origin-frozen historical features plus `horizon_offset`. Its honest local RMSLE
  should be close to the public LB; a large gap means the Kaggle setup differs
  from the labeled benchmark protocol.

### 4. Notebook run order

```bash
uv run jupyter lab
```

| # | Notebook | Notes |
|---|----------|-------|
| 1 | `post3_data_analysis.ipynb` | EDA (§1) — optional |
| 2 | `post3_feature_build.ipynb` | five nested feature levels + `as_of` leakage guard (§4.3–4.4) — optional |
| 3 | `post3_lightgbm_engineered.ipynb` | **LightGBM models** (§5.1, §6.1). Set `QUICK=False`; run the main comparison first on the core slice as a shakedown, then on the full panel for the production-scale baseline. The five-level `feature_ablation` run is separate/diagnostic. |
| 4 | `post3_tabpfn_ts.ipynb` *(GPU, to build)* | **TabPFN-TS leg** — must forecast the same series × origins × horizons as the LightGBM run being compared. Use the core slice while debugging; use the full panel for deployment-scale claims if runtime allows. |
| 5 | `post3_newsvendor.ipynb` | inventory-cost simulation + cost-ratio sensitivity (§8) |
| 6 | `post3_deployment.ipynb` | throughput/SLA, cold-start, drift A/B (§10, §5.2, §5.3) |
| 7 | `post3_evaluation.ipynb` | **head-to-head**: metrics, paired Wilcoxon, ranking-flip (§7) — run last |
| — | `post3_interpret.ipynb` | LightGBM TreeSHAP (§9) — optional, not on the benchmark path |

Notebooks 2 and 5–7 are driver notebooks: a control-panel cell up top, then
load → run → evaluate → save against the `src/demand` modules. Edit them directly.
Dependencies: #7 reads #3's (and #4's) per-series MAE parquets; #5 reads their
quantile forecasts. Run #3 and #4 before #5/#7.

Recommended LightGBM sequence:

1. **Core main-comparison shakedown** — verifies the full point + quantile path
   on the matched comparison slice before spending time on full-panel runs.
   ```python
   EXPERIMENT_KIND = "main_comparison"
   RUN_NAME = "lgbm_main_comparison_core"
   MAIN_FEATURE_SETS = ["minimal", "full"]
   USE_CORE_SLICE = True
   TRAIN_QUANTILE_MODELS = True
   QUICK = False
   ```
2. **Full-panel main comparison** — the production-scale LightGBM incumbent.
   This is the run to use for full-business deployment/runtime claims.
   ```python
   EXPERIMENT_KIND = "main_comparison"
   RUN_NAME = "lgbm_main_comparison_full"
   MAIN_FEATURE_SETS = ["minimal", "full"]
   USE_CORE_SLICE = False
   TRAIN_QUANTILE_MODELS = True
   QUICK = False
   ```
3. **Optional feature ablation** — useful for the feature-engineering story, but
   not the first main run. Keep it on the core slice unless you deliberately want
   to pay for full-panel ablation.
   ```python
   EXPERIMENT_KIND = "feature_ablation"
   RUN_NAME = "lgbm_feature_ablation_core"
   FEATURE_SETS = ["minimal", "lag", "rolling", "promo", "full"]
   USE_CORE_SLICE = True
   TRAIN_QUANTILE_MODELS = False
   QUICK = False
   ```

Finish by refreshing the headline artifacts:

```bash
uv run python -m src.demand.reporting.summary_tables      # results/summary_tables.md
uv run python -m src.demand.reporting.narrative_bullets   # results/narrative_bullets.md (vs §13 priors)
```

### 5. Benchmarking against TabPFN-TS

The LightGBM-vs-TabPFN comparison lives in **#7 evaluation** (paired Wilcoxon on
per-series MAE) and **#5 newsvendor** (cost table). Implement
`src/demand/models/tabpfn_ts.py` + the `post3_tabpfn_ts.ipynb` driver on GPU per
§6.2 (do the §16 day-1 future-covariate spike first), then have it write results
in the shapes the harness already expects:

1. **`results/tabpfn_ts/per_series_mae_*.parquet`** with the same columns as
   LightGBM's (`series_id, origin, regime, mae, feature_set`). Then add it to
   `PER_SERIES_SOURCES` in #7's control panel:
   ```python
   PER_SERIES_SOURCES = {
       "lgbm_point": "lgbm_engineered",
       "tabpfn_ts":  "tabpfn_ts",      # <- activates the Wilcoxon pair + ranking-flip
   }
   ```
2. **Quantile costs appended to `results/newsvendor_costs.parquet`** with
   `approach="tabpfn_ts"`. Re-run #5 to fill the §8.4 table's TabPFN columns.

Then re-run the two report generators. **Fairness requirement:** TabPFN-TS must
consume the same `(series, origin, horizon)` rows and known-future covariates as
the LightGBM artifact it is compared against. A core-slice TabPFN run should be
paired with `lgbm_main_comparison_core`; a full-panel TabPFN run should be paired
with `lgbm_main_comparison_full`. Mixing scopes is not apples-to-apples.

### Optional: Makefile shortcuts

The steps above are also wrapped as `make` targets (`make tests`,
`make kaggle-sanity`, `make tune`, `make reports`, `make all`, `make clean`) —
purely a convenience cheat-sheet for the commands in this section.
