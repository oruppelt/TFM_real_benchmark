"""
GBM wrappers (XGBoost and LightGBM) for insurance pricing.

Each wrapper supports three modeling approaches:
    'frequency'  — Poisson objective, sample_weight=exposure
    'severity'   — Gamma objective, claims-only subset (no exposure)
    'tweedie'    — Tweedie objective, sample_weight=exposure
    'binary'     — binary logistic (hurdle stage 1)

Hyperparameter tuning via Optuna:
    - Tuning happens ONCE on the full development set (not per outer CV fold).
    - Inner 3-fold CV within dev set for tuning.
    - Best params saved to configs/{model}_best_params_{approach}.json
    - Outer 5-fold CV then uses those fixed params.

    This prevents outer-fold overfitting to hyperparams while keeping compute
    manageable. The alternative (tune per outer fold) would be 5× more expensive.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import optuna
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import KFold, StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

CONFIGS_DIR = Path(__file__).parent.parent.parent / "configs"
CONFIGS_DIR.mkdir(exist_ok=True)

ApproachType = Literal["frequency", "severity", "tweedie", "binary"]

# ---------------------------------------------------------------------------
# Optuna search spaces
# ---------------------------------------------------------------------------

XGB_SEARCH_SPACE = {
    "max_depth":        ("int",       3,    10),
    "learning_rate":    ("float_log", 0.01, 0.3),
    "n_estimators":     ("int",       100,  1000),
    "min_child_weight": ("int",       1,    100),
    "subsample":        ("float",     0.6,  1.0),
    "colsample_bytree": ("float",     0.6,  1.0),
    "reg_alpha":        ("float_log", 1e-8, 10.0),
    "reg_lambda":       ("float_log", 1e-8, 10.0),
}

LGBM_SEARCH_SPACE = {
    "num_leaves":        ("int",       20,   300),
    "learning_rate":     ("float_log", 0.01, 0.3),
    "n_estimators":      ("int",       100,  1000),
    "min_child_samples": ("int",       5,    100),
    "subsample":         ("float",     0.6,  1.0),
    "colsample_bytree":  ("float",     0.6,  1.0),
    "reg_alpha":         ("float_log", 1e-8, 10.0),
    "reg_lambda":        ("float_log", 1e-8, 10.0),
}


def _sample_params(trial: optuna.Trial, space: dict) -> dict:
    params = {}
    for name, spec in space.items():
        kind = spec[0]
        if kind == "int":
            params[name] = trial.suggest_int(name, spec[1], spec[2])
        elif kind == "float":
            params[name] = trial.suggest_float(name, spec[1], spec[2])
        elif kind == "float_log":
            params[name] = trial.suggest_float(name, spec[1], spec[2], log=True)
    return params


def _objective_for_approach(approach: ApproachType) -> tuple[str, str]:
    """Return (xgb_objective, lgbm_objective) for an approach."""
    mapping = {
        "frequency": ("count:poisson",   "poisson"),
        "severity":  ("reg:gamma",        "gamma"),
        "tweedie":   ("reg:tweedie",      "tweedie"),
        "binary":    ("binary:logistic",  "binary"),
    }
    return mapping[approach]


# ---------------------------------------------------------------------------
# XGBoost wrapper
# ---------------------------------------------------------------------------

class XGBoostModel:
    """
    XGBoost wrapper with Optuna tuning for insurance pricing.

    For frequency/tweedie: exposure is passed as sample_weight (not an offset).
    This is less theoretically rigorous than a GLM offset but is standard
    practice and performs well empirically.

    For severity: no exposure — each claim is a claim.
    For binary: no exposure — classification probability.
    """

    def __init__(
        self,
        approach: ApproachType = "tweedie",
        tweedie_power: float = 1.5,
        params: dict | None = None,
    ):
        self.approach = approach
        self.tweedie_power = tweedie_power
        self._params = params or {}
        self._model: xgb.XGBModel | None = None

    @property
    def name(self) -> str:
        return f"XGBoost_{self.approach}"

    def _build_model(self, params: dict | None = None) -> xgb.XGBModel:
        p = {**self._params, **(params or {})}
        xgb_obj, _ = _objective_for_approach(self.approach)
        kwargs = dict(
            objective=xgb_obj,
            tree_method="hist",
            random_state=42,
            n_jobs=-1,
            verbosity=0,
            **p,
        )
        if self.approach == "tweedie":
            kwargs["tweedie_variance_power"] = self.tweedie_power
        if self.approach == "binary":
            return xgb.XGBClassifier(**kwargs)
        return xgb.XGBRegressor(**kwargs)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> "XGBoostModel":
        sample_weight = exposure if self.approach in ("frequency", "tweedie") else None
        self._model = self._build_model()
        self._model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(
        self,
        X: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.approach == "binary":
            return self._model.predict_proba(X)[:, 1]
        return np.clip(self._model.predict(X), 1e-10, None)

    def tune(
        self,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
        exposure_dev: np.ndarray | None = None,
        n_trials: int = 50,
        n_inner_folds: int = 3,
        features_label: str = "raw",
        force: bool = False,
    ) -> dict:
        """
        Optuna hyperparameter tuning on the full development set.

        Uses n_inner_folds-fold inner CV within the development set.
        Best params saved to configs/xgb_best_params_{approach}_{features}.json.
        """
        cache_path = CONFIGS_DIR / f"xgb_best_params_{self.approach}_{features_label}.json"
        if cache_path.exists() and not force:
            with open(cache_path) as f:
                self._params = json.load(f)
            print(f"Loaded cached XGB params from {cache_path}")
            return self._params

        from src.metrics import poisson_deviance, gamma_deviance, tweedie_deviance

        metric_fn = {
            "frequency": lambda yt, yp, sw: poisson_deviance(yt, yp, sample_weight=sw),
            "severity":  lambda yt, yp, sw: gamma_deviance(yt, yp, sample_weight=sw),
            "tweedie":   lambda yt, yp, sw: tweedie_deviance(yt, yp, power=self.tweedie_power, sample_weight=sw),
            "binary":    lambda yt, yp, sw: -float(np.mean((yt == (yp > 0.5)).astype(float))),
        }[self.approach]

        if self.approach == "binary":
            cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=0)
            folds = list(cv.split(X_dev, y_dev))
        else:
            cv = KFold(n_splits=n_inner_folds, shuffle=True, random_state=0)
            folds = list(cv.split(X_dev))

        def objective(trial: optuna.Trial) -> float:
            params = _sample_params(trial, XGB_SEARCH_SPACE)
            scores = []
            for tr_idx, val_idx in folds:
                X_tr, X_val = X_dev[tr_idx], X_dev[val_idx]
                y_tr, y_val = y_dev[tr_idx], y_dev[val_idx]
                exp_tr = exposure_dev[tr_idx] if exposure_dev is not None else None
                exp_val = exposure_dev[val_idx] if exposure_dev is not None else None
                m = self.clone()
                m._params = params
                m.fit(X_tr, y_tr, exposure=exp_tr)
                preds = m.predict(X_val, exposure=exp_val)
                scores.append(metric_fn(y_val, preds, exp_val))
            return float(np.mean(scores))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self._params = study.best_params
        with open(cache_path, "w") as f:
            json.dump(self._params, f, indent=2)
        print(f"Best XGB params ({self.approach}, {features_label}): {self._params}")
        return self._params

    def clone(self) -> "XGBoostModel":
        return XGBoostModel(
            approach=self.approach,
            tweedie_power=self.tweedie_power,
            params=dict(self._params),
        )


# ---------------------------------------------------------------------------
# LightGBM wrapper
# ---------------------------------------------------------------------------

class LightGBMModel:
    """
    LightGBM wrapper with Optuna tuning for insurance pricing.

    LightGBM natively handles categorical features via categorical_feature
    parameter — more efficient than ordinal encoding for this use case.
    However, we still use the same preprocessed features as XGBoost for
    a fair comparison. Native categorical support is a LightGBM advantage
    that shows up in the raw pipeline more than the engineered one.
    """

    def __init__(
        self,
        approach: ApproachType = "tweedie",
        tweedie_power: float = 1.5,
        params: dict | None = None,
    ):
        self.approach = approach
        self.tweedie_power = tweedie_power
        self._params = params or {}
        self._model: lgb.LGBMModel | None = None

    @property
    def name(self) -> str:
        return f"LightGBM_{self.approach}"

    def _build_model(self, params: dict | None = None) -> lgb.LGBMModel:
        p = {**self._params, **(params or {})}
        _, lgbm_obj = _objective_for_approach(self.approach)
        kwargs = dict(
            objective=lgbm_obj,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
            **p,
        )
        if self.approach == "tweedie":
            kwargs["tweedie_variance_power"] = self.tweedie_power
        if self.approach == "binary":
            return lgb.LGBMClassifier(**kwargs)
        return lgb.LGBMRegressor(**kwargs)

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> "LightGBMModel":
        sample_weight = exposure if self.approach in ("frequency", "tweedie") else None
        self._model = self._build_model()
        self._model.fit(X, y, sample_weight=sample_weight)
        return self

    def predict(
        self,
        X: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        if self.approach == "binary":
            return self._model.predict_proba(X)[:, 1]
        return np.clip(self._model.predict(X), 1e-10, None)

    def tune(
        self,
        X_dev: np.ndarray,
        y_dev: np.ndarray,
        exposure_dev: np.ndarray | None = None,
        n_trials: int = 50,
        n_inner_folds: int = 3,
        features_label: str = "raw",
        force: bool = False,
    ) -> dict:
        """Same tuning protocol as XGBoostModel.tune()."""
        cache_path = CONFIGS_DIR / f"lgbm_best_params_{self.approach}_{features_label}.json"
        if cache_path.exists() and not force:
            with open(cache_path) as f:
                self._params = json.load(f)
            print(f"Loaded cached LGBM params from {cache_path}")
            return self._params

        from src.metrics import poisson_deviance, gamma_deviance, tweedie_deviance

        metric_fn = {
            "frequency": lambda yt, yp, sw: poisson_deviance(yt, yp, sample_weight=sw),
            "severity":  lambda yt, yp, sw: gamma_deviance(yt, yp, sample_weight=sw),
            "tweedie":   lambda yt, yp, sw: tweedie_deviance(yt, yp, power=self.tweedie_power, sample_weight=sw),
            "binary":    lambda yt, yp, sw: -float(np.mean((yt == (yp > 0.5)).astype(float))),
        }[self.approach]

        if self.approach == "binary":
            cv = StratifiedKFold(n_splits=n_inner_folds, shuffle=True, random_state=0)
            folds = list(cv.split(X_dev, y_dev))
        else:
            cv = KFold(n_splits=n_inner_folds, shuffle=True, random_state=0)
            folds = list(cv.split(X_dev))

        def objective(trial: optuna.Trial) -> float:
            params = _sample_params(trial, LGBM_SEARCH_SPACE)
            scores = []
            for tr_idx, val_idx in folds:
                X_tr, X_val = X_dev[tr_idx], X_dev[val_idx]
                y_tr, y_val = y_dev[tr_idx], y_dev[val_idx]
                exp_tr = exposure_dev[tr_idx] if exposure_dev is not None else None
                exp_val = exposure_dev[val_idx] if exposure_dev is not None else None
                m = self.clone()
                m._params = params
                m.fit(X_tr, y_tr, exposure=exp_tr)
                preds = m.predict(X_val, exposure=exp_val)
                scores.append(metric_fn(y_val, preds, exp_val))
            return float(np.mean(scores))

        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

        self._params = study.best_params
        with open(cache_path, "w") as f:
            json.dump(self._params, f, indent=2)
        print(f"Best LGBM params ({self.approach}, {features_label}): {self._params}")
        return self._params

    def clone(self) -> "LightGBMModel":
        return LightGBMModel(
            approach=self.approach,
            tweedie_power=self.tweedie_power,
            params=dict(self._params),
        )
