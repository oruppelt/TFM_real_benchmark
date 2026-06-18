from __future__ import annotations

import pandas as pd
import pytest

from src.demand.sanity_check.kaggle_pipeline import (
    KAGGLE_HORIZON_DAYS,
    _coerce_submission_ids,
    _direct_train_origins,
)


def test_coerce_submission_ids_accepts_float_dtype_integer_values():
    ids = pd.Series([3000888.0, 3000889.0, 3000890.0])

    coerced = _coerce_submission_ids(ids)

    assert coerced.dtype == "int32"
    assert coerced.tolist() == [3000888, 3000889, 3000890]


def test_coerce_submission_ids_rejects_fractional_values():
    ids = pd.Series([3000888.0, 3000889.5])

    with pytest.raises(ValueError, match="integer-valued"):
        _coerce_submission_ids(ids)


def test_direct_train_origins_have_labeled_target_windows():
    as_of = pd.Timestamp("2017-08-15")

    origins = _direct_train_origins(as_of)

    assert origins
    assert origins[-1] + pd.Timedelta(days=KAGGLE_HORIZON_DAYS) <= as_of
    assert origins[-1] + pd.Timedelta(days=KAGGLE_HORIZON_DAYS) == as_of
