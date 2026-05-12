"""
GLM wrappers for insurance pricing.

Three models, all using statsmodels GLM with log link:
    PoissonGLM   — frequency model (Approach A stage 1, Approach B alt)
    GammaGLM     — severity model (Approach A stage 2, Approach C stage 2)
    TweedieGLM   — direct pure premium model (Approach B main)

Exposure handling differs by model:
    PoissonGLM  — log-offset (target is raw counts, scales with exposure)
    TweedieGLM  — no offset (target is pure_premium, already a rate)
    GammaGLM    — no offset (target is severity per claim, no exposure dependence)

All wrappers follow the shared interface:
    fit(X, y, exposure=None)  -> self
    predict(X, exposure=None) -> np.ndarray
    name                      -> str
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf


class PoissonGLM:
    """
    GLM with Poisson family and log link.

    Frequency model: target = ClaimNb (count).
    Exposure is handled as a log-offset (the proper actuarial way).
    At predict time, returns the predicted RATE (claims per unit exposure).
    Multiply by exposure to get expected claim count.

    Why offset, not sample_weight?
        An offset directly models log(E[y]) = X@beta + log(exposure).
        This encodes the assumption that a policy observed for 6 months
        should have half the expected claims of a 12-month policy — structurally,
        not just as a weighting heuristic.
    """

    name = "GLM_Poisson"

    def __init__(self, add_intercept: bool = True):
        self.add_intercept = add_intercept
        self._model = None
        self._result = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> "PoissonGLM":
        X_sm = sm.add_constant(X) if self.add_intercept else X
        offset = np.log(exposure) if exposure is not None else None
        glm = sm.GLM(
            y,
            X_sm,
            family=sm.families.Poisson(link=sm.families.links.Log()),
            offset=offset,
        )
        self._result = glm.fit(method="irls", maxiter=100, disp=False)
        return self

    def predict(
        self,
        X: np.ndarray,
        exposure: np.ndarray | None = None,
    ) -> np.ndarray:
        X_sm = sm.add_constant(X) if self.add_intercept else X
        offset = np.log(exposure) if exposure is not None else None
        # predict() with offset returns predicted COUNTS; divide by exposure for rate
        pred = self._result.predict(X_sm, offset=offset)
        if exposure is not None:
            return np.clip(pred / exposure, 1e-10, None)
        return np.clip(pred, 1e-10, None)

    def clone(self) -> "PoissonGLM":
        return PoissonGLM(add_intercept=self.add_intercept)


class GammaGLM:
    """
    GLM with Gamma family and log link.

    Severity model: target = ClaimAmount / ClaimNb (average cost per claim).
    Only fit on policies where ClaimNb > 0 — pass the claims-only subset.
    No exposure offset — each claim is a claim regardless of policy duration.
    """

    name = "GLM_Gamma"

    def __init__(self, add_intercept: bool = True):
        self.add_intercept = add_intercept
        self._result = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray | None = None,  # ignored for severity
    ) -> "GammaGLM":
        X_sm = sm.add_constant(X) if self.add_intercept else X
        glm = sm.GLM(
            y,
            X_sm,
            family=sm.families.Gamma(link=sm.families.links.Log()),
        )
        self._result = glm.fit(method="irls", maxiter=100, disp=False)
        return self

    def predict(
        self,
        X: np.ndarray,
        exposure: np.ndarray | None = None,  # ignored
    ) -> np.ndarray:
        X_sm = sm.add_constant(X) if self.add_intercept else X
        return np.clip(self._result.predict(X_sm), 1e-10, None)

    def clone(self) -> "GammaGLM":
        return GammaGLM(add_intercept=self.add_intercept)


class TweedieGLM:
    """
    GLM with Tweedie family (var_power=1.5) and log link.

    Direct pure premium model: target = ClaimAmount / Exposure.
    No exposure offset — pure_premium is already a rate (normalized by exposure),
    so it should NOT scale with exposure. Unlike PoissonGLM where the target is
    raw claim counts, adding log(exposure) here would bias the fitted coefficients.

    power=1.5 is the standard for compound Poisson-Gamma pure premium.
    Predictions are in the same units as the target (cost per unit of exposure).
    """

    name = "GLM_Tweedie"

    def __init__(self, var_power: float = 1.5, add_intercept: bool = True):
        self.var_power = var_power
        self.add_intercept = add_intercept
        self._result = None

    def fit(
        self,
        X: np.ndarray,
        y: np.ndarray,
        exposure: np.ndarray | None = None,  # ignored — target is already a rate
    ) -> "TweedieGLM":
        X_sm = sm.add_constant(X) if self.add_intercept else X
        glm = sm.GLM(
            y,
            X_sm,
            family=sm.families.Tweedie(
                var_power=self.var_power,
                link=sm.families.links.Log(),
            ),
        )
        self._result = glm.fit(method="irls", maxiter=200, disp=False)
        return self

    def predict(
        self,
        X: np.ndarray,
        exposure: np.ndarray | None = None,  # ignored — returns rate directly
    ) -> np.ndarray:
        X_sm = sm.add_constant(X) if self.add_intercept else X
        return np.clip(self._result.predict(X_sm), 1e-10, None)

    def clone(self) -> "TweedieGLM":
        return TweedieGLM(var_power=self.var_power, add_intercept=self.add_intercept)
