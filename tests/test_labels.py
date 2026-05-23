"""Tests for add_multi_horizon_labels with min_return_threshold."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.labels import add_multi_horizon_labels


def _make_close(values: list[float]) -> pd.DataFrame:
    return pd.DataFrame({"close": values})


def test_default_threshold_zero_any_positive_counts():
    df = _make_close([100.0, 100.001, 100.0, 100.0])
    result = add_multi_horizon_labels(df, {"1d": 1})
    # +0.001% return -> up=1 with default threshold of 0
    assert result["up_1d"].iloc[0] == 1


def test_nonzero_threshold_subthreshold_return_is_zero():
    # +0.001% = 0.00001 < 0.002 threshold -> up=0
    df = _make_close([100.0, 100.001, 100.0])
    result = add_multi_horizon_labels(df, {"1d": 1}, min_return_threshold=0.002)
    assert result["up_1d"].iloc[0] == 0


def test_nonzero_threshold_superthreshold_return_is_one():
    # +0.3% = 0.003 > 0.002 threshold -> up=1
    df = _make_close([100.0, 100.3, 100.0])
    result = add_multi_horizon_labels(df, {"1d": 1}, min_return_threshold=0.002)
    assert result["up_1d"].iloc[0] == 1


def test_threshold_boundary_below_is_zero():
    # 0.1% return < 0.2% threshold -> up=0 (clearly below, avoids float precision issues)
    df = _make_close([100.0, 100.1, 100.0])
    result = add_multi_horizon_labels(df, {"1d": 1}, min_return_threshold=0.002)
    assert result["up_1d"].iloc[0] == 0


def test_negative_return_always_zero():
    df = _make_close([100.0, 99.0, 100.0])
    result = add_multi_horizon_labels(df, {"1d": 1}, min_return_threshold=0.0)
    assert result["up_1d"].iloc[0] == 0
    result2 = add_multi_horizon_labels(df, {"1d": 1}, min_return_threshold=0.002)
    assert result2["up_1d"].iloc[0] == 0


def test_ret_column_preserved():
    df = _make_close([100.0, 101.0, 100.0])
    result = add_multi_horizon_labels(df, {"1d": 1}, min_return_threshold=0.002)
    assert "ret_1d" in result.columns
    assert abs(result["ret_1d"].iloc[0] - 0.01) < 1e-9


def test_multiple_horizons():
    prices = [100.0, 101.0, 102.0, 103.0]
    df = _make_close(prices)
    result = add_multi_horizon_labels(df, {"1d": 1, "2d": 2}, min_return_threshold=0.005)
    assert "up_1d" in result.columns
    assert "up_2d" in result.columns
