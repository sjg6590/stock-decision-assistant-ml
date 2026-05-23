"""Tests that vol_z and gap are clipped to configurable limits."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.features import GAP_CLIP, VOL_Z_CLIP, build_features


def _make_frame(n: int = 50) -> pd.DataFrame:
    """Minimal OHLCV frame with datetime column."""
    dates = pd.date_range("2023-01-01", periods=n, freq="B")
    close = 100.0 + np.arange(n) * 0.1
    return pd.DataFrame(
        {
            "datetime": dates,
            "open": close * 1.001,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.ones(n) * 1_000_000,
        }
    )


def test_vol_z_clipped_to_default():
    df = _make_frame(50)
    # Inject a single extreme volume spike on the last bar
    df.loc[df.index[-1], "volume"] = 1e12
    result = build_features(df)
    assert result["vol_z"].max() <= VOL_Z_CLIP
    assert result["vol_z"].min() >= -VOL_Z_CLIP


def test_gap_clipped_to_default():
    df = _make_frame(50)
    # Inject a 50% overnight gap
    df.loc[df.index[-1], "open"] = df.loc[df.index[-2], "close"] * 1.50
    result = build_features(df)
    assert result["gap"].max() <= GAP_CLIP
    assert result["gap"].min() >= -GAP_CLIP


def test_feature_clip_config_override():
    df = _make_frame(50)
    df.loc[df.index[-1], "volume"] = 1e12
    result = build_features(df, feature_clip={"vol_z": 2.0, "gap": 0.05})
    assert result["vol_z"].max() <= 2.0
    assert result["gap"].max() <= 0.05


def test_no_clipping_when_values_in_range():
    df = _make_frame(50)
    result = build_features(df)
    # With normal data no row should be at the clip boundary
    assert result["vol_z"].abs().max() < VOL_Z_CLIP


def test_vol_z_clip_constant_is_positive():
    assert VOL_Z_CLIP > 0
    assert GAP_CLIP > 0
