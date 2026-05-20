from __future__ import annotations

import numpy as np
import pandas as pd

from ml.features import build_features, feature_columns, model_features_stale, resolve_model_features

BB_WARMUP = 20


def _sample_df(n: int = 120) -> pd.DataFrame:
    dt = pd.date_range("2024-01-01", periods=n, freq="D")
    close = pd.Series(range(100, 100 + n), dtype=float)
    return pd.DataFrame(
        {
            "datetime": dt,
            "open": close - 0.5,
            "high": close + 1.0,
            "low": close - 1.0,
            "close": close,
            "volume": 1_000_000,
            "symbol": "TEST",
        }
    )


def test_bollinger_columns_in_feature_columns() -> None:
    cols = feature_columns()
    assert "bb_width" in cols
    assert "bb_pct_b" in cols


def test_bollinger_features_present_after_build() -> None:
    out = build_features(_sample_df())
    assert "bb_width" in out.columns
    assert "bb_pct_b" in out.columns


def test_bollinger_finite_after_warmup() -> None:
    out = build_features(_sample_df())
    tail = out.iloc[BB_WARMUP:]
    assert tail["bb_width"].notna().all()
    assert tail["bb_pct_b"].notna().all()
    assert np.isfinite(tail["bb_width"]).all()
    assert np.isfinite(tail["bb_pct_b"]).all()


def test_bb_pct_b_flat_band_guard() -> None:
    n = 40
    dt = pd.date_range("2024-01-01", periods=n, freq="D")
    flat_close = pd.Series([100.0] * n)
    df = pd.DataFrame(
        {
            "datetime": dt,
            "open": flat_close - 0.5,
            "high": flat_close + 1.0,
            "low": flat_close - 1.0,
            "close": flat_close,
            "volume": 1_000_000,
            "symbol": "TEST",
        }
    )
    out = build_features(df)
    tail = out.iloc[BB_WARMUP:]
    assert tail["bb_pct_b"].notna().all()
    assert np.isfinite(tail["bb_pct_b"]).all()
    assert (tail["bb_pct_b"] >= 0.0).all()
    assert (tail["bb_pct_b"] <= 1.0).all()


def test_ret_5_vs_spy_in_feature_columns() -> None:
    assert "ret_5_vs_spy" in feature_columns()


def test_ret_5_vs_spy_zero_without_spy_frame() -> None:
    out = build_features(_sample_df())
    assert "ret_5_vs_spy" in out.columns
    assert (out["ret_5_vs_spy"] == 0.0).all()


def test_ret_5_vs_spy_zero_with_empty_spy_frame() -> None:
    out = build_features(_sample_df(), spy_frame=pd.DataFrame())
    assert (out["ret_5_vs_spy"] == 0.0).all()


def test_ret_5_vs_spy_finite_with_spy_frame() -> None:
    n = 60
    dt = pd.date_range("2024-01-01", periods=n, freq="D")
    ticker_close = pd.Series(np.linspace(100, 130, n))
    spy_close = pd.Series(np.linspace(100, 115, n))
    ticker = pd.DataFrame(
        {
            "datetime": dt,
            "open": ticker_close - 0.5,
            "high": ticker_close + 1.0,
            "low": ticker_close - 1.0,
            "close": ticker_close,
            "volume": 1_000_000,
            "symbol": "AAPL",
        }
    )
    spy = pd.DataFrame(
        {
            "datetime": dt,
            "open": spy_close - 0.5,
            "high": spy_close + 1.0,
            "low": spy_close - 1.0,
            "close": spy_close,
            "volume": 2_000_000,
            "symbol": "SPY",
        }
    )
    out = build_features(ticker, spy_frame=spy)
    tail = out.iloc[10:]
    assert tail["ret_5_vs_spy"].notna().all()
    assert np.isfinite(tail["ret_5_vs_spy"]).all()
    assert (tail["ret_5_vs_spy"].abs() > 1e-6).any()


def test_ret_5_vs_spy_nan_safe_with_partial_spy() -> None:
    n = 40
    dt = pd.date_range("2024-01-01", periods=n, freq="D")
    ticker = _sample_df(n)
    spy = pd.DataFrame(
        {
            "datetime": dt[10:],
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.0,
            "volume": 1_000_000,
            "symbol": "SPY",
        }
    )
    out = build_features(ticker, spy_frame=spy)
    assert out["ret_5_vs_spy"].notna().all()
    assert np.isfinite(out["ret_5_vs_spy"]).all()


def test_resolve_model_features_from_payload() -> None:
    legacy = ["ret_1", "ret_5", "vol_20"]
    payload = {"classifiers": {}, "feature_columns": legacy}
    assert resolve_model_features(payload) == legacy


def test_model_features_stale_detects_legacy_artifact() -> None:
    legacy = [
        "ret_1",
        "ret_5",
        "ret_20",
        "vol_20",
        "vol_z",
        "gap",
        "rsi_14",
        "macd",
        "macd_signal",
        "day_of_week",
    ]
    payload = {"classifiers": {}, "feature_columns": legacy}
    assert model_features_stale(payload)
    payload["feature_columns"] = feature_columns()
    assert not model_features_stale(payload)
