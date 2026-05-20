from __future__ import annotations

import pandas as pd

from ml.features import build_features
from ml.labels import add_multi_horizon_labels


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


def test_labels_exist() -> None:
    df = _sample_df()
    out = add_multi_horizon_labels(df, {"1d": 1, "5d": 5})
    assert "ret_1d" in out.columns
    assert "up_1d" in out.columns
    assert "ret_5d" in out.columns
    assert "up_5d" in out.columns


def test_features_exist() -> None:
    df = _sample_df()
    out = build_features(df)
    for col in ["ret_1", "ret_5", "rsi_14", "macd", "day_of_week"]:
        assert col in out.columns
