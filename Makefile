# Post 3 — Favorita demand forecasting pipeline (§12.4).
#
# `make all` reproduces the non-TabPFN pipeline from a clean tree. The TabPFN-TS
# leg (§6.2) runs separately on GPU and is intentionally not wired here. Heavy
# steps checkpoint to parquet, so a second `make all` on an unchanged tree is fast.
#
# PY lets you pin an interpreter, e.g. `make all PY="uv run python"`.
PY ?= python
NBCONVERT = $(PY) -m jupyter nbconvert --to notebook --execute --inplace \
            --ExecutePreprocessor.timeout=-1

NB_DIR = notebooks
RESULTS = results

.PHONY: all clean tests reports kaggle-sanity tune \
        eda feature-build lgbm newsvendor interpret evaluation deployment

## all: leakage tests -> sanity gate -> feature/model/eval/business/deploy notebooks -> reports
all: tests kaggle-sanity feature-build lgbm newsvendor interpret deployment evaluation reports
	@echo "==> Post 3 non-TabPFN pipeline complete. TabPFN-TS (§6.2) runs separately on GPU."

## tests: leakage / fairness unit tests (§4.4, §6.0.3)
tests:
	$(PY) -m pytest tests/ -q

## kaggle-sanity: §11 gate — writes results/sanity_check/{submission.csv,metrics.json}
kaggle-sanity:
	$(PY) -m src.demand.sanity_check.kaggle_pipeline --config default.yaml

## tune: deferred Optuna job (§6.1) — full panel, all objectives, hours of compute
tune:
	$(PY) scripts/run_tuning.py --config default.yaml

## reports: refresh results/summary_tables.md and results/narrative_bullets.md
reports:
	$(PY) -m src.demand.reporting.summary_tables --config default.yaml
	$(PY) -m src.demand.reporting.narrative_bullets --config default.yaml

# --- Individual notebook executions (each writes parquet/figures) -------------
eda:
	$(NBCONVERT) $(NB_DIR)/post3_data_analysis.ipynb
feature-build:
	$(NBCONVERT) $(NB_DIR)/post3_feature_build.ipynb
lgbm:
	$(NBCONVERT) $(NB_DIR)/post3_lightgbm_engineered.ipynb
newsvendor:
	$(NBCONVERT) $(NB_DIR)/post3_newsvendor.ipynb
interpret:
	$(NBCONVERT) $(NB_DIR)/post3_interpret.ipynb
evaluation:
	$(NBCONVERT) $(NB_DIR)/post3_evaluation.ipynb
deployment:
	$(NBCONVERT) $(NB_DIR)/post3_deployment.ipynb

## clean: remove generated Post 3 results + figures + data cache (keeps Post 2)
clean:
	rm -rf $(RESULTS)/lgbm_engineered $(RESULTS)/newsvendor $(RESULTS)/deployment \
	       $(RESULTS)/interpret $(RESULTS)/evaluation $(RESULTS)/feature_build \
	       $(RESULTS)/tuning $(RESULTS)/sanity_check \
	       $(RESULTS)/figures/post3_data_analysis \
	       $(RESULTS)/newsvendor_costs.parquet $(RESULTS)/newsvendor_sensitivity.parquet \
	       $(RESULTS)/coldstart.parquet $(RESULTS)/drift.parquet \
	       $(RESULTS)/throughput.parquet $(RESULTS)/runtime_log.parquet \
	       $(RESULTS)/wilcoxon_tests.parquet $(RESULTS)/shap_global_importance.parquet \
	       $(RESULTS)/summary_tables.md $(RESULTS)/narrative_bullets.md \
	       data/favorita_demand/_cache
	@echo "==> cleaned Post 3 artifacts"
