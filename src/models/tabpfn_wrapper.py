"""
TabPFN wrapper for insurance pricing.

Consequences for the experiment design:
    1. No sample_weight → can't use exposure weighting like GBMs.
       Two workarounds implemented (see exposure_strategy).
    2. No native Tweedie objective → TabPFN optimises its own internal loss.
       Expect a gap vs GBMs with native Tweedie objective.
       Post-hoc recalibration (Tweedie GLM layer) partially addresses this.
    3. Row limit → must subsample training data. Tested at 10K and 50K.

Exposure strategies:
    'feature'  — exposure is a regular input feature, target = ClaimNb (raw count)
                 Simple; lets TabPFN discover the exposure relationship from data.
                 Likely less accurate than the rate approach for varying exposures.

    'rate'     — target = ClaimNb / Exposure, no exposure feature
                 Predicts the claim rate directly. At inference, multiply by exposure.
                 Mirrors GLM structure; better calibrated for exposure-varying portfolios.

Recalibration (Option B — nested inner-fold approach):
    TabPFN optimises its own internal loss, which may not align with Tweedie deviance.
    To address this, we apply a thin post-hoc recalibration layer.

    Implementation (per outer CV fold):
        1. Within the outer training fold, run inner K-fold CV on TabPFN.
        2. Collect out-of-fold (OOF) predictions on the inner training data.
        3. Fit the recalibration model (Tweedie GLM or isotonic) on those OOF preds.
        4. Apply recalibration to outer validation fold predictions.

    Why Option B (nested inner-fold) over Option A (simple calibration split)?
        Option A (80/20 split within training fold) is simpler but wastes 20% of
        training data. With TabPFN already limited to 10-50K samples, losing 20%
        is significant. Option B uses all training data for both TabPFN training
        (inner folds) and recalibration fitting (OOF preds from inner folds).
        It's more statistically correct and more data-efficient, at the cost of
        running TabPFN n_inner_folds times per outer fold.

        For 10K subsample × 3 inner folds: 30K total TabPFN fits per outer fold.
        This is manageable locally; 50K × 3 may require GPU.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.isotonic import IsotonicRegression

ExposureStrategy = Literal["feature", "rate"]
RecalibrationMethod = Literal["tweedie_glm", "isotonic", "none"]


class TabPFNWrapper:
    """
    TabPFN (Regressor or Classifier) with insurance-specific adaptations.

    For classification (hurdle stage 1): uses TabPFNClassifier.
    For regression (frequency, severity, pure premium): uses TabPFNRegressor.
    """

    def __init__(
        self,
        task: Literal["regression", "classification"] = "regression",
        n_train_max: int = 10_000,
        exposure_strategy: ExposureStrategy = "rate",
        random_state: int = 42,
        predict_batch_size: int = 1000,
    ):
        """
        Args:
            task:               'regression' for frequency/severity/PP; 'classification' for hurdle.
            n_train_max:        Maximum training rows. Excess rows are subsampled (stratified).
            exposure_strategy:  How to handle exposure in regression tasks.
                                'rate'    → predict ClaimNb/Exposure, multiply back at inference
                                'feature' → pass Exposure as feature, predict raw ClaimNb
            random_state:       Seed for subsampling reproducibility.
            predict_batch_size: Batch size for inference. TabPFN loads the full test set into
                                GPU VRAM — large val folds (100K+ rows) cause OOM.
                                Default 1000 is safe on a 24GB GPU; increase if fast enough.
        """
        self.task = task
        self.n_train_max = n_train_max
        self.exposure_strategy = exposure_strategy
        self.random_state = random_state
        self.predict_batch_size = predict_batch_size
        self._model = None
        self._n_clipped = 0  # count of negative predictions clipped to 1e-10

    @property
    def name(self) -> str:
        return f"TabPFN_{self.task}_{self.n_train_max//1000}K_{self.exposure_strategy}"

    def _subsample(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """
        Stratified subsample when len(X) > n_train_max.

        Stratification: bin y into 5 quantile buckets to preserve the target
        distribution. For binary targets (hurdle), stratify directly on y.
        """
        if len(X) <= self.n_train_max:
            return X, y, exposure

        rng = np.random.default_rng(self.random_state)

        # Stratify on quantile bins of y
        if len(np.unique(y)) <= 2:
            strata = y.astype(int)
        else:
            bins = np.quantile(y, np.linspace(0, 1, 6))
            strata = np.digitize(y, bins[1:-1])

        # Sample proportionally from each stratum
        idx_sampled = []
        for s in np.unique(strata):
            s_idx = np.where(strata == s)[0]
            n_take = max(1, int(self.n_train_max * len(s_idx) / len(X)))
            idx_sampled.append(rng.choice(s_idx, size=min(n_take, len(s_idx)), replace=False))

        idx = np.concatenate(idx_sampled)
        # Trim or pad to exactly n_train_max
        if len(idx) > self.n_train_max:
            idx = rng.choice(idx, size=self.n_train_max, replace=False)

        exp_sub = exposure[idx] if exposure is not None else None
        return X[idx], y[idx], exp_sub

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> "TabPFNWrapper":
        """
        Fit TabPFN after subsampling and applying exposure strategy.

        Note: TabPFN v7.0.1 does not support sample_weight in fit().
        Exposure is handled via the exposure_strategy parameter instead.
        """
        from tabpfn import TabPFNRegressor, TabPFNClassifier

        # Apply exposure strategy before subsampling
        if self.task == "regression" and exposure is not None:
            if self.exposure_strategy == "rate":
                y_fit = y / np.clip(exposure, 1e-10, None)
                X_fit = X  # exposure NOT included as feature
            else:  # 'feature' — append exposure as last column
                y_fit = y
                X_fit = np.column_stack([X, exposure])
        else:
            y_fit = y
            X_fit = X

        X_sub, y_sub, _ = self._subsample(X_fit, y_fit)

        if self.task == "classification":
            self._model = TabPFNClassifier(random_state=self.random_state)
        else:
            self._model = TabPFNRegressor(random_state=self.random_state)

        self._model.fit(X_sub, y_sub)
        return self

    def predict(
        self,
        X: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Predict and apply inverse exposure transformation.

        For 'rate' strategy: prediction is the claim rate → multiply by exposure.
        For 'feature' strategy: prediction is raw count → return as-is.
        For classification: returns P(claim > 0).
        """
        if self.exposure_strategy == "feature" and exposure is not None:
            X_pred = np.column_stack([X, exposure])
        else:
            X_pred = X

        # Batch inference to avoid GPU OOM on large val/holdout sets.
        # TabPFN loads the full test set into VRAM — 100K+ rows exhausts 24GB.
        raw = self._predict_batched(X_pred, classify=(self.task == "classification"))

        if self.task == "classification":
            return raw

        # Count and clip negative/zero predictions — informative for TabPFN
        n_bad = int(np.sum(raw <= 0))
        self._n_clipped += n_bad
        pred = np.clip(raw, 1e-10, None)

        if self.exposure_strategy == "rate" and exposure is not None:
            return pred * exposure
        return pred

    def _predict_batched(self, X: np.ndarray, classify: bool = False) -> np.ndarray:
        """Run predict in batches of predict_batch_size to avoid GPU OOM."""
        n = len(X)
        if n <= self.predict_batch_size:
            if classify:
                return self._model.predict_proba(X)[:, 1]
            return self._model.predict(X)

        results = []
        for start in range(0, n, self.predict_batch_size):
            batch = X[start: start + self.predict_batch_size]
            if classify:
                results.append(self._model.predict_proba(batch)[:, 1])
            else:
                results.append(self._model.predict(batch))
        return np.concatenate(results)

    def predict_quantiles(
        self,
        X: np.ndarray,
        quantiles: list[float],
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Return quantile predictions from TabPFNRegressor.
        Shape: (n_samples, n_quantiles).
        """
        if self.task == "classification":
            raise ValueError("Quantile prediction only for regression.")

        X_pred = np.column_stack([X, exposure]) if (
            self.exposure_strategy == "feature" and exposure is not None
        ) else X

        # Batched quantile prediction to avoid GPU OOM
        n = len(X_pred)
        batches = []
        for start in range(0, n, self.predict_batch_size):
            batch = X_pred[start: start + self.predict_batch_size]
            result = self._model.predict(batch, output_type="quantiles", quantiles=quantiles)
            if hasattr(result, "shape"):
                batches.append(result)
            else:
                batches.append(np.stack(result, axis=1))
        q_preds = np.concatenate(batches, axis=0)

        if self.exposure_strategy == "rate" and exposure is not None:
            q_preds = q_preds * exposure[:, None]
        return q_preds

    def clone(self) -> "TabPFNWrapper":
        return TabPFNWrapper(
            task=self.task,
            n_train_max=self.n_train_max,
            exposure_strategy=self.exposure_strategy,
            random_state=self.random_state,
            predict_batch_size=self.predict_batch_size,
        )


# ---------------------------------------------------------------------------
# Recalibration layer (Option B — nested inner-fold OOF approach)
# ---------------------------------------------------------------------------

class TweedieRecalibrator:
    """
    Post-hoc Tweedie GLM recalibration for TabPFN predictions.

    Fits a regularized Tweedie GLM on a stabilized log TabPFN prediction:

        z = standardize(log(max(pred, floor)))
        E[y | z] = exp(intercept + beta * z)

    This keeps the original multiplicative recalibration idea while avoiding
    the numerical leverage of log(1e-10) when TabPFN emits non-positive values
    that the wrapper clips to its prediction floor.

    Intended to be fitted on OOF predictions from inner-fold CV (Option B).
    Applied to outer validation fold at inference time.

    Why a learned floor?
        Current TabPFN regression outputs can be non-positive before wrapper
        clipping. Using log(1e-10) for those clipped values creates extreme
        leverage points. A low positive OOF quantile treats those rows as very
        low predictions without letting an arbitrary numerical epsilon dominate
        the fit.
    """

    def __init__(
        self,
        var_power: float = 1.5,
        alpha: float = 1e-4,
        floor_quantile: float = 0.01,
    ):
        self.var_power = var_power
        self.alpha = alpha
        self.floor_quantile = floor_quantile
        self._model = None
        self.n_fit_samples_ = 0
        self.pred_floor_ = 1e-10
        self.log_pred_mean_ = 0.0
        self.log_pred_scale_ = 1.0

    def fit(
        self,
        y_true: np.ndarray,
        y_pred_tabpfn: np.ndarray,
    ) -> "TweedieRecalibrator":
        from sklearn.linear_model import TweedieRegressor

        y_true = np.asarray(y_true, dtype=float).ravel()
        y_pred_tabpfn = np.asarray(y_pred_tabpfn, dtype=float).ravel()
        valid = np.isfinite(y_true) & np.isfinite(y_pred_tabpfn) & (y_true >= 0)
        if not np.any(valid):
            raise ValueError("No valid rows available to fit TweedieRecalibrator")

        pred = y_pred_tabpfn[valid]
        y = y_true[valid]
        self.n_fit_samples_ = int(len(y))
        self.pred_floor_ = self._learn_floor(pred)
        log_pred = np.log(np.clip(pred, self.pred_floor_, None))
        self.log_pred_mean_ = float(np.mean(log_pred))
        scale = float(np.std(log_pred))
        self.log_pred_scale_ = scale if np.isfinite(scale) and scale > 0 else 1.0
        X = ((log_pred - self.log_pred_mean_) / self.log_pred_scale_).reshape(-1, 1)

        self._model = TweedieRegressor(
            power=self.var_power,
            link="log",
            alpha=self.alpha,
            max_iter=1000,
            tol=1e-6,
        )
        self._model.fit(X, y)
        return self

    def predict(self, y_pred_tabpfn: np.ndarray) -> np.ndarray:
        if self._model is None:
            raise ValueError("TweedieRecalibrator must be fitted before predict")

        y_pred_tabpfn = np.asarray(y_pred_tabpfn, dtype=float)
        pred = np.where(np.isfinite(y_pred_tabpfn), y_pred_tabpfn, self.pred_floor_)
        log_pred = np.log(np.clip(pred, self.pred_floor_, None))
        X = ((log_pred - self.log_pred_mean_) / self.log_pred_scale_).reshape(-1, 1)
        return np.clip(self._model.predict(X), 1e-10, None)

    def clone(self) -> "TweedieRecalibrator":
        return TweedieRecalibrator(
            var_power=self.var_power,
            alpha=self.alpha,
            floor_quantile=self.floor_quantile,
        )

    def _learn_floor(self, pred: np.ndarray) -> float:
        positive = pred[np.isfinite(pred) & (pred > 1e-8)]
        if len(positive) == 0:
            return 1e-3

        floor = float(np.quantile(positive, self.floor_quantile))
        if not np.isfinite(floor) or floor <= 0:
            return 1e-3
        return max(floor, 1e-3)


class IsotonicRecalibrator:
    """
    Post-hoc isotonic regression recalibration.

    More flexible than Tweedie GLM but risks overfitting with small OOF sets.
    Run both and compare; the blog post documents which closes more of the gap.
    """

    def __init__(self):
        self._model = IsotonicRegression(out_of_bounds="clip")

    def fit(self, y_true: np.ndarray, y_pred_tabpfn: np.ndarray) -> "IsotonicRecalibrator":
        self._model.fit(y_pred_tabpfn, y_true)
        return self

    def predict(self, y_pred_tabpfn: np.ndarray) -> np.ndarray:
        return np.clip(self._model.predict(y_pred_tabpfn), 1e-10, None)

    def clone(self) -> "IsotonicRecalibrator":
        return IsotonicRecalibrator()
