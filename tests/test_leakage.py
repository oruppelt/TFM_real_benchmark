"""Leakage-prevention unit tests (§4.4, §6.0.3).

Build a tiny synthetic panel, run build_features / build_supervised_frame, and
assert both temporal-leakage and future-covariate symmetry properties.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.demand.data.features import FEATURE_SETS, build_features
from src.demand.data.leakage_tests import (
    assert_future_covariate_channel, assert_no_future_sales_leakage,
    assert_symmetric_future_covariates,
)
from src.demand.data.load import RawData, build_panel
from src.demand.data.supervised_frame import build_supervised_frame


@pytest.fixture
def tiny_panel() -> pd.DataFrame:
    dates = pd.date_range("2016-01-01", "2016-03-31", freq="D")
    stores = [1, 2]
    families = ["PRODUCE", "AUTOMOTIVE"]
    rows = []
    rng = np.random.default_rng(0)
    for s in stores:
        for f in families:
            for d in dates:
                rows.append({
                    "date": d, "store_nbr": s, "family": f,
                    "sales": float(rng.integers(0, 100)),
                    "onpromotion": int(rng.integers(0, 5)),
                    "oil_price": 50.0 + float(rng.normal(0, 2)),
                    "city": "Quito" if s == 1 else "Guayaquil",
                    "state": "Pichincha" if s == 1 else "Guayas",
                    "type": "A", "cluster": 1 if s == 1 else 2,
                    "transactions": float(rng.integers(500, 2000)),
                    "id": None,
                })
    return pd.DataFrame(rows)


@pytest.fixture
def stores_df() -> pd.DataFrame:
    return pd.DataFrame({
        "store_nbr": [1, 2],
        "city": ["Quito", "Guayaquil"],
        "state": ["Pichincha", "Guayas"],
        "type": ["A", "B"],
        "cluster": [1, 2],
    })


@pytest.fixture
def holidays_df() -> pd.DataFrame:
    return pd.DataFrame({
        "date": pd.to_datetime(["2016-02-09", "2016-03-25"]),
        "type": ["Holiday", "Holiday"],
        "locale": ["National", "National"],
        "locale_name": ["Ecuador", "Ecuador"],
        "description": ["Carnaval", "Viernes Santo"],
        "transferred": [False, False],
    })


@pytest.mark.parametrize("feature_set", FEATURE_SETS)
def test_no_future_sales_leakage(tiny_panel, holidays_df, stores_df, feature_set):
    origin = pd.Timestamp("2016-02-29")
    assert_no_future_sales_leakage(
        tiny_panel, origin=origin, feature_set=feature_set,
        config={"horizon_days": 28},
        holidays=holidays_df, stores=stores_df,
    )


def test_supervised_frame_shape(tiny_panel, holidays_df, stores_df):
    origin = pd.Timestamp("2016-02-29")
    frame = build_supervised_frame(
        tiny_panel, origins=[origin], horizon=28, feature_set="full",
        config={"horizon_days": 28}, mode="train",
        holidays=holidays_df, stores=stores_df,
    )
    # 2 stores × 2 families × 28 horizons (for the one origin).
    assert len(frame) <= 2 * 2 * 28
    assert {"origin", "date", "horizon_offset", "store_nbr", "family", "series_id"}.issubset(
        frame.columns
    )
    assert frame["horizon_offset"].between(1, 28).all()


def test_symmetric_future_covariates(tiny_panel, holidays_df, stores_df):
    """Build the supervised frame twice at different feature sets — known-future
    columns that exist in both must have identical values per row key."""
    origin = pd.Timestamp("2016-02-29")
    cfg = {"horizon_days": 28}
    frame_min = build_supervised_frame(
        tiny_panel, origins=[origin], horizon=28, feature_set="minimal",
        config=cfg, mode="forecast",
        holidays=holidays_df, stores=stores_df,
    )
    frame_full = build_supervised_frame(
        tiny_panel, origins=[origin], horizon=28, feature_set="full",
        config=cfg, mode="forecast",
        holidays=holidays_df, stores=stores_df,
    )
    assert_symmetric_future_covariates(frame_min, frame_full)


def test_future_perturbation_with_mock_model(tiny_panel, holidays_df, stores_df):
    """Use a simple deterministic 'model' that depends on onpromotion to confirm
    the leakage_tests harness correctly detects response/non-response."""
    origin = pd.Timestamp("2016-02-29")
    frame = build_supervised_frame(
        tiny_panel, origins=[origin], horizon=28, feature_set="full",
        config={"horizon_days": 28}, mode="forecast",
        holidays=holidays_df, stores=stores_df,
    )

    def fake_predict(df: pd.DataFrame) -> pd.Series:
        # Row-wise: each row's prediction depends only on that row's onpromotion.
        return df["onpromotion"].astype(float) * 10.0

    assert_future_covariate_channel(frame, fake_predict, origin=origin,
                                    horizon_to_perturb=14)


def test_supervised_frame_freezes_historical_features_per_origin(tiny_panel, holidays_df, stores_df):
    origin = pd.Timestamp("2016-02-29")
    frame = build_supervised_frame(
        tiny_panel, origins=[origin], horizon=28, feature_set="full",
        config={"horizon_days": 28}, mode="forecast",
        holidays=holidays_df, stores=stores_df,
    )
    historical_cols = [
        "sales_lag_1", "sales_lag_7", "sales_lag_14", "sales_lag_28",
        "sales_ma_7", "sales_ma_28", "sales_std_7", "sales_std_28",
        "promo_lift", "family_share_of_store_sales",
        "transactions_lag_1", "transactions_lag_7",
    ]
    for _, group in frame.groupby(["store_nbr", "family"], sort=False):
        first = group.iloc[0]
        for col in historical_cols:
            expected = first[col] if pd.notna(first[col]) else "__NA__"
            assert group[col].fillna("__NA__").eq(expected).all(), col


def test_supervised_frame_uses_origin_oil_not_future_actuals():
    dates = pd.date_range("2016-01-01", "2016-03-31", freq="D")
    origin = pd.Timestamp("2016-02-29")
    rows = []
    for i, date in enumerate(dates):
        rows.append({
            "date": date, "store_nbr": 1, "family": "PRODUCE",
            "sales": float(i), "onpromotion": 0, "oil_price": float(100 + i),
            "city": "Quito", "state": "Pichincha", "type": "A", "cluster": 1,
            "transactions": 1000.0 + i, "id": None,
        })
    panel = pd.DataFrame(rows)

    frame = build_supervised_frame(
        panel, origins=[origin], horizon=7, feature_set="full",
        config={"horizon_days": 28}, mode="forecast",
    )

    origin_oil = panel.loc[panel["date"] == origin, "oil_price"].iloc[0]
    assert frame["oil_price"].eq(origin_oil).all()
    assert frame.loc[frame["horizon_offset"] == 7, "oil_ma_7"].iloc[0] == origin_oil


def test_family_share_uses_store_daily_total():
    dates = pd.date_range("2016-01-01", "2016-02-15", freq="D")
    rows = []
    for date in dates:
        for family, sales in [("PRODUCE", 10.0), ("GROCERY I", 30.0)]:
            rows.append({
                "date": date, "store_nbr": 1, "family": family,
                "sales": sales, "onpromotion": 0, "oil_price": 50.0,
                "city": "Quito", "state": "Pichincha", "type": "A", "cluster": 1,
                "transactions": 1000.0, "id": None,
            })
    panel = pd.DataFrame(rows)
    origin = pd.Timestamp("2016-01-31")

    frame = build_supervised_frame(
        panel, origins=[origin], horizon=1, feature_set="full",
        config={"horizon_days": 28}, mode="forecast",
    )

    shares = frame.set_index("family")["family_share_of_store_sales"]
    assert shares.loc["PRODUCE"] == pytest.approx(0.25)
    assert shares.loc["GROCERY I"] == pytest.approx(0.75)


def test_build_panel_zero_fills_train_gaps():
    train = pd.DataFrame({
        "id": [1, 2, 3],
        "date": pd.to_datetime(["2016-01-01", "2016-01-01", "2016-01-02"]),
        "store_nbr": [1, 1, 1],
        "family": ["PRODUCE", "GROCERY I", "PRODUCE"],
        "sales": [10.0, 20.0, 11.0],
        "onpromotion": [0, 0, 1],
    })
    raw = RawData(
        train=train,
        test=pd.DataFrame(columns=["id", "date", "store_nbr", "family", "onpromotion"]),
        stores=pd.DataFrame({
            "store_nbr": [1], "city": ["Quito"], "state": ["Pichincha"],
            "type": ["A"], "cluster": [1],
        }),
        oil=pd.DataFrame({
            "date": pd.to_datetime(["2016-01-01", "2016-01-02"]),
            "dcoilwtico": [50.0, 51.0],
        }),
        holidays=pd.DataFrame(),
        transactions=pd.DataFrame({
            "date": pd.to_datetime(["2016-01-01", "2016-01-02"]),
            "store_nbr": [1, 1],
            "transactions": [1000.0, 1100.0],
        }),
    )

    panel = build_panel(raw, include_test=False)
    missing = panel[
        (panel["date"] == pd.Timestamp("2016-01-02"))
        & (panel["store_nbr"] == 1)
        & (panel["family"] == "GROCERY I")
    ]
    assert len(missing) == 1
    assert missing["sales"].iloc[0] == 0.0
    assert missing["onpromotion"].iloc[0] == 0
