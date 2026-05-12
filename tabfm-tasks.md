# TabFM Series — Task Backlog

## Epic: Tabular Foundation Models in the Real World — Substack Series

---

## 🏗️ Phase 0: Setup

### TABFM-001: Create project infrastructure
- [ ] Create GitHub repo (`tabfm-real-world` or similar)
- [ ] Set up `pyproject.toml` with pinned dependencies: tabpfn, xgboost, lightgbm, catboost, statsmodels, optuna, shap, scikit-learn, matplotlib, seaborn
- [ ] Create folder structure: `notebooks/`, `src/`, `data/`, `results/`, `figures/`
- [ ] Set up `.gitignore` (data files, model checkpoints, `.env`)
- [ ] Confirm GPU access (local M3 Pro won't cut it for TabPFN — need Colab Pro, Lambda, or similar)
- [ ] Test that `TabPFNClassifier()` and `TabPFNRegressor()` actually run on your setup
- [ ] Pin TabPFN version — check if 2.5 is available via pip or only HuggingFace

**Acceptance:** `pip install -e .` works, TabPFN runs a toy example, repo is pushed to GitHub.

---

### TABFM-002: Acquire and validate insurance dataset
- [ ] Download freMTPL2 via `fetch_openml(data_id=41214)` — or find the freq + sev split
- [ ] Verify columns: driver age, vehicle age, vehicle power, region, density, bonus-malus, exposure
- [ ] Verify target: ClaimNb (frequency), ClaimAmount (severity)
- [ ] Check row count (~670K), missing values, data types
- [ ] Create a `data/insurance/` folder with raw + processed versions
- [ ] Write a `src/load_insurance.py` loader with train/test split logic
- [ ] Sanity check: fit a quick GLM Poisson model — does it match known benchmarks from sklearn docs?

**Acceptance:** Dataset loads cleanly, basic GLM reproduces expected ballpark results.

---

### TABFM-003: Acquire and validate time series dataset
- [ ] Decide on dataset (energy demand, Store Sales Kaggle, or M5 subset)
- [ ] Download and inspect: series count, length, exogenous variables available
- [ ] Check for the properties you need: seasonality, trend, exogenous features, multiple series
- [ ] Create `data/timeseries/` folder
- [ ] Write a `src/load_timeseries.py` loader
- [ ] Sanity check: plot a few series, fit a naive seasonal baseline

**Acceptance:** Dataset loads, you understand its structure, naive baseline produces reasonable numbers.

---

### TABFM-004: Build shared evaluation utilities
- [ ] `src/metrics.py` — all metrics in one place:
  - Insurance: Tweedie deviance, Poisson deviance, Gamma deviance, RMSE, MAE, Gini
  - Time series: MASE, sMAPE, RMSE, WQL
  - Classification (if needed): AUC, precision@k, F-beta
- [ ] `src/evaluation.py` — cross-validation wrapper that runs any model and returns all metrics
- [ ] `src/plotting.py` — reusable chart functions (bar comparisons, SHAP side-by-side, calibration plots)
- [ ] Test all metrics on dummy data to make sure they're correct

**Acceptance:** `metrics.tweedie_deviance(y_true, y_pred, power=1.5)` returns a sensible number.

---

## 🏥 Phase 1: Insurance Experiments (Post 2)

### TABFM-005: Build insurance feature engineering pipeline
- [ ] **Raw features pipeline:** minimal preprocessing — handle categoricals, pass exposure as a feature
- [ ] **Engineered features pipeline:** the stuff that took you years to learn:
  - Bonus-malus binning / transformation
  - Vehicle power groupings
  - Age bands (nonlinear effects)
  - Density log-transformation
  - Region clustering or target encoding
  - Interaction features (age × vehicle power, density × region)
  - Exposure normalization
- [ ] Both pipelines output clean X_train, X_test, y_train, y_test
- [ ] Document each engineered feature with a one-line rationale (this becomes blog content)

**Acceptance:** Two feature sets ready, both pass through XGBoost without errors.

---

### TABFM-006: Run baseline models on insurance data
- [ ] **GLM Poisson (frequency):** statsmodels, with log(exposure) as offset
- [ ] **GLM Gamma (severity):** statsmodels
- [ ] **GLM Tweedie (pure premium):** statsmodels
- [ ] **XGBoost:** Optuna tuning (50-100 trials), `objective='reg:tweedie'`
- [ ] **LightGBM:** Optuna tuning, Tweedie objective
- [ ] **CatBoost:** Optuna tuning, Tweedie objective
- [ ] Run each on BOTH raw and engineered features
- [ ] Record all metrics from TABFM-004 for every combination
- [ ] Record training time and inference time (predict 100K rows)
- [ ] Save all results to `results/insurance_baselines.parquet`

**Acceptance:** 6 models × 2 feature sets × 6+ metrics = full results matrix populated.

---

### TABFM-007: Run TabPFN on insurance data
- [ ] **Critical first question:** freMTPL2 has ~670K rows. TabPFN 2.5 handles up to 50K. You MUST subsample. Try: 5K, 10K, 25K, 50K subsamples. This itself becomes a finding.
- [ ] Run TabPFN 2.5 on raw features (Cell A — the promise)
- [ ] Run TabPFN 2.5 on engineered features (Cell B — does engineering help?)
- [ ] **Exposure handling experiment:** try multiple approaches:
  - Exposure as a regular feature
  - Predict rate (target / exposure) instead of count
  - Sample weights (if TabPFN API supports it)
  - Document what works and what doesn't
- [ ] **Custom metric question:** TabPFN optimizes its own internal loss. Evaluate on Tweedie deviance anyway — document the gap vs XGBoost with native Tweedie objective
- [ ] **Workaround test:** TabPFN predictions → recalibrate with isotonic regression or thin Tweedie GLM layer. Does this close the gap?
- [ ] Record all metrics, training time, inference time
- [ ] Save to `results/insurance_tabpfn.parquet`

**Acceptance:** Feature engineering 2×2 matrix is fully populated. Exposure handling findings documented.

---

### TABFM-008: Run TabICL v2 on insurance data (optional but valuable)
- [ ] Same experiments as TABFM-007 but with TabICL v2
- [ ] Compare: does a fully open-source TFM perform similarly?
- [ ] Record all metrics
- [ ] Save to `results/insurance_tabicl.parquet`

**Acceptance:** Side-by-side comparison of TabPFN 2.5 vs TabICL v2 on insurance data.

---

### TABFM-009: Insurance interpretability experiments
- [ ] **TreeSHAP on XGBoost:** standard, fast — compute for all features
- [ ] **KernelSHAP on TabPFN:** does `shap.Explainer(tabpfn.predict, X_train)` work? Time it.
- [ ] Side-by-side: pick 3-4 key features (e.g., bonus-malus, driver age, vehicle power). Compare SHAP profiles from XGBoost vs TabPFN. Do they agree?
- [ ] **Partial dependence plots:** same features, both models
- [ ] Check if TabPFN attention weights are accessible via API — if yes, extract and compare to SHAP
- [ ] **The regulatory test:** for 5 individual policies, generate a "why did premium change" explanation from each model. Which is more convincing?
- [ ] Save all SHAP values and plots to `results/insurance_interpretability/`

**Acceptance:** Side-by-side SHAP comparison charts ready. Interpretability gap is quantified (even if qualitatively).

---

### TABFM-010: Insurance deployment benchmarks
- [ ] Inference latency: time `model.predict(X_test)` for 10K, 50K, 100K rows — all models
- [ ] Memory usage during inference (use `tracemalloc` or `psutil`)
- [ ] Model serialization size (pickle file size)
- [ ] **TabPFN distillation test:** use the distillation engine to convert to tree ensemble. Measure:
  - Accuracy retention (Tweedie deviance before vs after distillation)
  - Inference speed improvement
  - File size of distilled model
- [ ] Save to `results/insurance_deployment.parquet`

**Acceptance:** Deployment comparison table fully populated.

---

## 📈 Phase 2: Time Series Experiments (Post 3)

### TABFM-011: Set up time series baselines
- [ ] **Naive seasonal baseline** (last year's value)
- [ ] **ETS / ARIMA:** via statsforecast
- [ ] **Prophet:** standard config
- [ ] **LightGBM with lag features:** manual tabular reframing with:
  - Lag features (1, 7, 14, 28 days or appropriate for your data)
  - Calendar features (day of week, month, holiday flags)
  - Rolling statistics (7-day mean, 28-day mean)
- [ ] **XGBoost with same lag features**
- [ ] Run on both minimal features and rich/domain-engineered features
- [ ] Record MASE, sMAPE, RMSE, WQL for all
- [ ] Save to `results/timeseries_baselines.parquet`

**Acceptance:** All baselines produce forecasts. Metrics computed and saved.

---

### TABFM-012: Run TabPFN-TS on time series data
- [ ] **Default TabPFN-TS:** follow their setup (automatic lag/calendar/Fourier features)
- [ ] **TabPFN-TS with added exogenous features** (if the dataset has them)
- [ ] **Standard TabPFN 2.5 with YOUR manual tabular reframing** — this is the "domain expert reframing" test. Use the same lag features as LightGBM but feed to TabPFN as a standard tabular problem
- [ ] **The interesting comparison:** TabPFN-TS (automatic reframing) vs TabPFN 2.5 (your manual reframing). Does domain expertise in the reframing beat the automatic approach?
- [ ] **Probabilistic forecasting:** extract prediction intervals from TabPFN. Compare to quantile regression from LightGBM
- [ ] **Calibration:** are TabPFN's 90% prediction intervals actually covering 90% of the time?
- [ ] Record all metrics
- [ ] Save to `results/timeseries_tabpfn.parquet`

**Acceptance:** Automatic vs manual reframing comparison complete. Probabilistic calibration assessed.

---

### TABFM-013: Time series practical workflow tests
- [ ] **Scaling test:** run TabPFN-TS on 1, 10, 100, 1000 series. Plot runtime vs count.
- [ ] **Cold start test:** take a series with full history, forecast. Then take only 10/20/50 data points and forecast again. Compare degradation for TabPFN-TS vs LightGBM.
- [ ] **Distribution shift:** if possible, train on one period, test on a period with known shift (e.g., COVID period, promo period). Which model degrades more gracefully?
- [ ] Save to `results/timeseries_practical.parquet`

**Acceptance:** Scaling, cold start, and robustness findings documented.

---

### TABFM-014: Time series interpretability + deployment
- [ ] SHAP on the tabular-reframed representation — does it make temporal sense?
- [ ] Inference latency comparison across all models
- [ ] Compare: time-to-first-forecast (setup + fit + predict) for each approach

**Acceptance:** Interpretability and deployment comparison tables ready.

---

## ✍️ Phase 3: Writing

### TABFM-015: Write Post 1 — "Can Tabular FMs Survive the Real World?"
- [ ] Draft opening hook (the 60+ projects angle)
- [ ] Write the 6-stage lifecycle framework
- [ ] Describe both experiment designs
- [ ] Write "what I expect to find" (priors)
- [ ] Self-review: does it make someone want to read Posts 2-4?
- [ ] Get 1-2 peer reviews (ML friends, DataRobot colleagues)
- [ ] Final polish
- [ ] Publish on Substack
- [ ] Cross-post to LinkedIn (600-800 word version)

**Note:** Can be written in parallel with experiments (Phase 1/2). Doesn't need results.

---

### TABFM-016: Write Post 2 — "Insurance Pricing: An Actuary's Stress Test"
- [ ] Compile results from TABFM-006 through TABFM-010
- [ ] Create all 5 key visualizations (feature matrix, metric disagreement, SHAP comparison, deviance curves, latency)
- [ ] Draft each lifecycle stage section
- [ ] Write the business value translation
- [ ] Build the results summary table
- [ ] Write honest "what surprised me" closing
- [ ] Self-review, peer review, polish
- [ ] Publish, cross-post

---

### TABFM-017: Write Post 3 — "Time Series: Beyond the Hype"
- [ ] Compile results from TABFM-011 through TABFM-014
- [ ] Create all 5 key visualizations
- [ ] Draft each section
- [ ] Contrast findings with Post 2 (insurance)
- [ ] Self-review, peer review, polish
- [ ] Publish, cross-post

---

### TABFM-018: Write Post 4 — "The Practitioner's Verdict"
- [ ] Build the combined lifecycle scorecard from Posts 2+3
- [ ] Write the "gaps that matter" section based on actual findings
- [ ] Write "what I'd build" section — 6 concrete ideas
- [ ] Write "what this means for ML teams" — leadership perspective
- [ ] Write the personal reflection / interest signal closing
- [ ] Self-review, peer review, polish
- [ ] Publish, cross-post
- [ ] Tag Prior Labs, Fundamental, TabICL authors

---

## 🚀 Phase 4: Distribution

### TABFM-019: GitHub repo polish
- [ ] Clean notebooks — remove dead code, add markdown commentary
- [ ] Write proper README: what this is, how to reproduce, link to blog posts
- [ ] Add a `results/` summary with key findings
- [ ] Make sure it runs end-to-end from a clean clone
- [ ] Pin to your GitHub profile

---

### TABFM-020: Engagement and follow-up
- [ ] Respond to all comments on Substack and LinkedIn (first 48 hours are critical)
- [ ] If Prior Labs team engages — be gracious, offer to share raw results
- [ ] If anyone challenges methodology — respond thoughtfully, update if valid
- [ ] Track: views, subscribers gained, LinkedIn impressions, any DMs from target companies
- [ ] If traction is strong: consider Post 5 (capstone / open letter)

---

## 📋 Quick Reference: Dependencies

```
TABFM-001 → everything else
TABFM-002 → TABFM-005, 006, 007, 008, 009, 010
TABFM-003 → TABFM-011, 012, 013, 014
TABFM-004 → TABFM-006, 007, 011, 012
TABFM-005 → TABFM-006, 007
TABFM-006 + 007 + 009 + 010 → TABFM-016
TABFM-011 + 012 + 013 + 014 → TABFM-017
TABFM-015 → can start anytime after TABFM-001
TABFM-016 + 017 → TABFM-018
TABFM-018 → TABFM-019, 020
```

## 🗓️ Suggested Sprint Plan

**Sprint 1 (Week 1):** TABFM-001, 002, 003, 004, start 015
**Sprint 2 (Week 2-3):** TABFM-005, 006, 007, 008
**Sprint 3 (Week 3-4):** TABFM-009, 010, 011, 012
**Sprint 4 (Week 4-5):** TABFM-013, 014, finish 015, start 016
**Sprint 5 (Week 5-6):** TABFM-016, 017
**Sprint 6 (Week 6-7):** TABFM-018, 019, 020
