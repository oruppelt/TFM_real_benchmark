"""
Feature engineering pipelines for the insurance experiment.

Three pipelines with a shared interface:
    RawFeaturePipeline       — minimal preprocessing, no domain knowledge
    EngineeredFeaturePipeline — actuarial transformations and interactions (GLM-oriented)
    GBMFeaturePipeline       — GBM-optimised: raw numerics + TE + business features

All expose:
    fit_transform(df_train) -> np.ndarray
    transform(df)           -> np.ndarray
    feature_names_          -> list[str]   (set after fit)
    clone()                 -> new unfitted instance

CRITICAL on target encoding leakage:
    Target encoding must be fit ONLY on the training portion of each CV fold.
    Both pipelines call fit_transform on train and transform on val/holdout.
    The CV engine (cv_engine.py) is responsible for calling .clone() per fold
    so nothing from one fold leaks into another.

    We use sklearn's TargetEncoder(cv=5) which internally uses 5-fold
    cross-fitting when computing the in-sample encoding during fit_transform.
    This prevents within-train-fold leakage too.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, TargetEncoder
from sklearn.compose import ColumnTransformer


# ---------------------------------------------------------------------------
# Pipeline A: Raw Features
# ---------------------------------------------------------------------------


class RawFeaturePipeline:
    """
    Minimal preprocessing — no domain knowledge applied.

    Transforms:
        Categoricals (Area, VehBrand, VehGas, Region) → ordinal integers
        Numerics (VehPower, VehAge, DrivAge, BonusMalus, Density) → as-is
        Exposure is NOT transformed here — it is passed separately to model fit/predict.

    Rationale: this is the "out of the box" experience. No actuary needed.
    """

    CATEGORICAL_COLS = ["Area", "VehBrand", "VehGas", "Region"]
    # Exposure is NOT in the pipeline — passed separately to model fit/predict
    # (as offset for GLMs, sample_weight for GBMs, or handled by TabPFNWrapper)
    NUMERIC_COLS = ["VehPower", "VehAge", "DrivAge", "BonusMalus", "Density"]

    def __init__(self):
        self._transformer: ColumnTransformer | None = None
        self.feature_names_: list[str] = []

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        self._transformer = ColumnTransformer(
            transformers=[
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                    ),
                    self.CATEGORICAL_COLS,
                ),
                ("num", "passthrough", self.NUMERIC_COLS),
            ],
            remainder="drop",
        )
        result = self._transformer.fit_transform(X)
        self.feature_names_ = self.CATEGORICAL_COLS + self.NUMERIC_COLS
        return result.astype(np.float32)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if self._transformer is None:
            raise RuntimeError("Call fit_transform before transform.")
        return self._transformer.transform(X).astype(np.float32)

    def clone(self) -> "RawFeaturePipeline":
        return RawFeaturePipeline()


# ---------------------------------------------------------------------------
# Pipeline B: Domain-Engineered Features
# ---------------------------------------------------------------------------


class EngineeredFeaturePipeline:
    """
    Actuarial feature engineering — every transformation has a documented rationale.

    Transformations applied:
        BonusMalus   → bins [50-60, 60-80, 80-100, 100-150, 150+]
                        Rationale: regulatory risk tiers, non-linear relationship
                        with claims; 50 = maximum bonus, >100 = malus territory.

        VehPower     → groups [4-6, 7-9, 10-12, 13+]
                        Rationale: claim risk flattens above power 10; grouping
                        reduces noise from high-power exotic vehicles.

        DrivAge      → bands [18-25, 26-35, 36-45, 46-55, 56-65, 65+]
                        Rationale: classic U-shaped actuarial age curve; young
                        and elderly drivers have elevated risk.

        Density      → log1p(Density)
                        Rationale: 100× range (1 to 27,000), extremely right-skewed.
                        Log transformation makes the feature interpretable as
                        "urban vs rural" rather than a raw population count.

        VehAge       → capped at 20, bins [0-1, 2-5, 6-10, 11-15, 16+]
                        Rationale: new-car effect (0-1 years) vs old-car effect
                        (depreciated vehicles with higher repair costs).

        VehGas       → binary Diesel=1, Regular=0
                        Rationale: already binary; Diesel historically associated
                        with slightly higher severity.

        Area         → ordinal A=1..F=6 (already density-ordered in source data)
                        Rationale: ordinal preserves the density gradient.

        Region       → TargetEncoder(cv=5)
                        Rationale: 22 levels — too many for one-hot, too few for
                        simple ordinal. Target encoding captures the regional
                        claim rate signal without expanding dimensionality.
                        cv=5 prevents within-sample leakage during fit_transform.

        VehBrand     → TargetEncoder(cv=5)
                        Rationale: same as Region. 11 brands, claim rates vary
                        significantly (commercial vs premium vs economy).

        DrivAge × VehPower  → interaction of age band × power group
                        Rationale: young driver + high-power car is the highest-
                        risk combination; the interaction captures this synergy
                        that additive models miss.

        BonusMalus × DrivAge → interaction of BM bin × age band
                        Rationale: a BM of 80 (slight malus) means different things
                        for a 20-year-old (still building record) vs a 50-year-old
                        (established driver with recent claims).

        Exposure is NOT in these features — passed separately to model fit/predict.
    """

    # Use -inf as left edge so all values are captured (BonusMalus starts at 50, but
    # using -inf is safe and prevents NaN from pd.cut for any edge-case values)
    _BM_BINS = [-np.inf, 60, 80, 100, 150, np.inf]
    _BM_LABELS = [0, 1, 2, 3, 4]

    _POW_BINS = [-np.inf, 6, 9, 12, np.inf]
    _POW_LABELS = [0, 1, 2, 3]

    _AGE_BINS = [-np.inf, 25, 35, 45, 55, 65, np.inf]
    _AGE_LABELS = [0, 1, 2, 3, 4, 5]

    _VAGE_BINS = [-np.inf, 1, 5, 10, 15, np.inf]
    _VAGE_LABELS = [0, 1, 2, 3, 4]

    _AREA_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}

    TARGET_ENCODE_COLS = ["Region", "VehBrand"]

    def __init__(self):
        self._target_encoders: dict[str, TargetEncoder] = {}
        self.feature_names_: list[str] = []
        self._fitted = False

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all deterministic transforms (no target info needed)."""
        df = X.copy()

        # BonusMalus bins
        df["bm_bin"] = pd.cut(
            df["BonusMalus"].astype(float), bins=self._BM_BINS, labels=self._BM_LABELS, right=True
        ).astype(float)

        # VehPower groups
        df["pow_group"] = pd.cut(
            df["VehPower"].astype(float), bins=self._POW_BINS, labels=self._POW_LABELS, right=True
        ).astype(float)

        # DrivAge bands
        df["age_band"] = pd.cut(
            df["DrivAge"].astype(float), bins=self._AGE_BINS, labels=self._AGE_LABELS, right=True
        ).astype(float)

        # Density log transform
        df["log_density"] = np.log1p(df["Density"].astype(float))

        # VehAge bins (cap at 20 first)
        df["vehage_capped"] = df["VehAge"].astype(float).clip(upper=20)
        df["vehage_bin"] = pd.cut(
            df["vehage_capped"], bins=self._VAGE_BINS, labels=self._VAGE_LABELS, right=True
        ).astype(float)

        # VehGas binary
        df["is_diesel"] = (df["VehGas"].astype(str).str.strip() == "Diesel").astype(float)

        # Area ordinal
        df["area_ordinal"] = df["Area"].astype(str).str.strip().map(self._AREA_MAP).astype(float)

        # Interactions
        df["age_x_power"] = df["age_band"] * df["pow_group"]
        df["bm_x_age"] = df["bm_bin"] * df["age_band"]

        return df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        """
        Fit target encoders on training data, apply all transforms.

        y should be the pure_premium target (or claim_freq for frequency-only work).
        If y is None, target encoding falls back to ordinal encoding with a warning.
        """
        df = self._engineer(X)

        # Target encoding — fit on training data only
        self._target_encoders = {}
        for col in self.TARGET_ENCODE_COLS:
            enc = TargetEncoder(cv=5, smooth="auto")
            if y is not None:
                enc.fit(df[[col]], y)
                df[f"{col}_te"] = enc.transform(df[[col]]).ravel()
            else:
                import warnings

                warnings.warn(
                    f"No target provided for TargetEncoder on {col}. "
                    "Falling back to ordinal encoding. Pass y to fit_transform for proper encoding."
                )
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                df[f"{col}_te"] = oe.fit_transform(df[[col]]).ravel()
                enc = oe  # store whatever we used
            self._target_encoders[col] = enc

        self._fitted = True
        self.feature_names_ = self._get_feature_names()
        return df[self.feature_names_].values.astype(np.float32)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")
        df = self._engineer(X)
        for col, enc in self._target_encoders.items():
            df[f"{col}_te"] = enc.transform(df[[col]]).ravel()
        return df[self.feature_names_].values.astype(np.float32)

    def _get_feature_names(self) -> list[str]:
        # Exposure is excluded — passed separately to model fit/predict
        return [
            "bm_bin",
            "pow_group",
            "age_band",
            "log_density",
            "vehage_bin",
            "is_diesel",
            "area_ordinal",
            "Region_te",
            "VehBrand_te",
            "age_x_power",
            "bm_x_age",
        ]

    def clone(self) -> "EngineeredFeaturePipeline":
        return EngineeredFeaturePipeline()


# ---------------------------------------------------------------------------
# Pipeline C: GBM-Optimised Features
# ---------------------------------------------------------------------------


class GBMFeaturePipeline:
    """
    GBM-optimised feature engineering — raw numerics preserved, TE for high-
    cardinality categoricals, plus genuine business features that add signal
    beyond what a GBM can discover from plain splits alone.

    Design principle: never bin or discretise continuous variables (binning
    destroys split resolution for trees). Only add features that encode
    domain knowledge trees cannot easily discover at typical depth/regularization.

    Raw continuous (kept as-is):
        VehPower, VehAge, DrivAge, BonusMalus
        Rationale: trees split at optimal thresholds automatically.

    Compressed continuous:
        log_density = log1p(Density)
        Rationale: Density spans 1–27,000 (27,000× range). Extreme right-skew
        means regularization/depth is dominated by a handful of dense-city rows.
        Log compression makes the feature proportional to urban/rural gradient
        rather than a raw population count.

    Ordinal categorical:
        area_ordinal — Area A=1..F=6 (density-ordered)
        Rationale: ordinal preserves the density gradient (A=rural, F=urban).

    Binary:
        is_diesel — Diesel=1, Regular=0

    Target encoded (cv=5):
        Region_te, VehBrand_te
        Rationale: 22 and 11 levels. Unlike the EngineeredPipeline, no binning
        accompanies the TE here — just the signal the GBM cannot derive from
        ordinal integers.

    Business features — genuine signal that trees miss at typical depth:

        bm_excess = BonusMalus - 50
            Range: 0–300+. Zero means maximum no-claims bonus (perfect driver).
            More interpretable than raw BM; encodes "how much bad history exists".
            A tree on raw BonusMalus could learn this shift, but centering at zero
            makes the feature's intercept interpretable and aids regularization.

        is_malus = (BonusMalus > 100)
            Hard actuarial threshold: BM >100 means the driver is currently in
            the malus zone — they've had recent at-fault claims. This regime
            change is well-known to actuaries and worth making explicit.

        young_driver = (DrivAge < 26)
            Universal industry threshold. Under-26 is the consistently highest-
            risk bracket globally (inexperience + risk tolerance). Making it
            explicit helps regularized/shallow trees that might not split here.

        senior_driver = (DrivAge > 69)
            Second peak of the U-shaped actuarial age risk curve. Cognitive and
            reaction-time deterioration elevates risk after 70.

        young_x_power = young_driver × VehPower  (continuous)
            The single most dangerous combination in motor insurance. A 22-year-
            old with VehPower=15 is a qualitatively different risk from either
            young age or high power alone. Trees need 2 sequential splits to
            approximate this synergy; an explicit feature captures it in 1 split.

        vehicle_value_proxy = VehPower / (VehAge + 1)
            Approximates vehicle value in the absence of a price column. Newer,
            more powerful vehicles cost more to repair or replace. The +1 avoids
            division by zero for new vehicles (VehAge=0).

    Exposure is NOT in these features — passed separately to model fit/predict.
    """

    _AREA_MAP = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6}
    TARGET_ENCODE_COLS = ["Region", "VehBrand"]

    def __init__(self):
        self._target_encoders: dict[str, TargetEncoder] = {}
        self.feature_names_: list[str] = []
        self._fitted = False

    def _engineer(self, X: pd.DataFrame) -> pd.DataFrame:
        """Apply all deterministic transforms (no target info needed)."""
        df = X.copy()

        # Log-compress density
        df["log_density"] = np.log1p(df["Density"].astype(float))

        # Area ordinal (density-ordered A=1..F=6)
        df["area_ordinal"] = df["Area"].astype(str).str.strip().map(self._AREA_MAP).astype(float)

        # Binary gas
        df["is_diesel"] = (df["VehGas"].astype(str).str.strip() == "Diesel").astype(float)

        # --- Novel business features ---

        # BonusMalus excess over minimum (0 = perfect record)
        df["bm_excess"] = df["BonusMalus"].astype(float) - 50.0

        # Malus zone flag (BM > 100 = active malus territory)
        df["is_malus"] = (df["BonusMalus"].astype(float) > 100).astype(float)

        # Young driver flag (< 26 = highest-risk age bracket)
        df["young_driver"] = (df["DrivAge"].astype(float) < 26).astype(float)

        # Senior driver flag (> 69 = second risk peak)
        df["senior_driver"] = (df["DrivAge"].astype(float) > 69).astype(float)

        # Young × power interaction (continuous: young=1 if <26, else 0)
        df["young_x_power"] = df["young_driver"] * df["VehPower"].astype(float)

        # Vehicle value proxy (power / (age + 1))
        df["vehicle_value_proxy"] = (
            df["VehPower"].astype(float) / (df["VehAge"].astype(float) + 1.0)
        )

        return df

    def fit_transform(self, X: pd.DataFrame, y: pd.Series | None = None) -> np.ndarray:
        """
        Fit target encoders on training data, apply all transforms.

        y should be the pure_premium target. If None, falls back to ordinal
        encoding with a warning (same contract as EngineeredFeaturePipeline).
        """
        df = self._engineer(X)

        self._target_encoders = {}
        for col in self.TARGET_ENCODE_COLS:
            enc = TargetEncoder(cv=5, smooth="auto")
            if y is not None:
                enc.fit(df[[col]], y)
                df[f"{col}_te"] = enc.transform(df[[col]]).ravel()
            else:
                import warnings

                warnings.warn(
                    f"No target provided for TargetEncoder on {col}. "
                    "Falling back to ordinal encoding."
                )
                oe = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
                df[f"{col}_te"] = oe.fit_transform(df[[col]]).ravel()
                enc = oe
            self._target_encoders[col] = enc

        self._fitted = True
        self.feature_names_ = self._get_feature_names()
        return df[self.feature_names_].values.astype(np.float32)

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Call fit_transform before transform.")
        df = self._engineer(X)
        for col, enc in self._target_encoders.items():
            df[f"{col}_te"] = enc.transform(df[[col]]).ravel()
        return df[self.feature_names_].values.astype(np.float32)

    def _get_feature_names(self) -> list[str]:
        return [
            # Raw numerics
            "VehPower",
            "VehAge",
            "DrivAge",
            "BonusMalus",
            # Compressed / encoded
            "log_density",
            "area_ordinal",
            "is_diesel",
            "Region_te",
            "VehBrand_te",
            # Business features
            "bm_excess",
            "is_malus",
            "young_driver",
            "senior_driver",
            "young_x_power",
            "vehicle_value_proxy",
        ]

    def clone(self) -> "GBMFeaturePipeline":
        return GBMFeaturePipeline()
