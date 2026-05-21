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
