from __future__ import annotations

import numpy as np
import pandas as pd
from ta.momentum import RSIIndicator
from ta.trend import MACD
from ta.volatility import BollingerBands
from ta.volatility import AverageTrueRange

VOL_Z_CLIP = 5.0
GAP_CLIP = 0.15  # 15% — larger than typical overnight gaps, catches halts


def build_features(
    frame: pd.DataFrame,
    spy_frame: pd.DataFrame | None = None,
    feature_clip: dict | None = None,
) -> pd.DataFrame:
    df = frame.sort_values("datetime").copy()
    df["ret_1"] = np.log(df["close"] / df["close"].shift(1))
    df["ret_5"] = np.log(df["close"] / df["close"].shift(5))
    df["ret_20"] = np.log(df["close"] / df["close"].shift(20))
    df["vol_20"] = df["ret_1"].rolling(20).std()
    df["atr_14"] = AverageTrueRange(
        high=df["high"], low=df["low"], close=df["close"], window=14
    ).average_true_range() / (df["close"] + 1e-9)
    df["vol_z"] = (df["volume"] - df["volume"].rolling(20).mean()) / (df["volume"].rolling(20).std() + 1e-9)
    df["gap"] = (df["open"] - df["close"].shift(1)) / (df["close"].shift(1) + 1e-9)
    _vol_z_clip = float((feature_clip or {}).get("vol_z", VOL_Z_CLIP))
    _gap_clip = float((feature_clip or {}).get("gap", GAP_CLIP))
    df["vol_z"] = df["vol_z"].clip(-_vol_z_clip, _vol_z_clip)
    df["gap"] = df["gap"].clip(-_gap_clip, _gap_clip)
    df["rsi_14"] = RSIIndicator(close=df["close"], window=14).rsi()
    macd = MACD(close=df["close"])
    df["macd"] = macd.macd()
    df["macd_signal"] = macd.macd_signal()
    df["day_of_week"] = pd.to_datetime(df["datetime"]).dt.dayofweek
    bb = BollingerBands(close=df["close"], window=20, window_dev=2)
    upper = bb.bollinger_hband()
    lower = bb.bollinger_lband()
    middle = bb.bollinger_mavg()
    df["bb_width"] = (upper - lower) / (middle + 1e-9)
    band_range = upper - lower
    df["bb_pct_b"] = np.where(
        band_range.abs() > 1e-9,
        (df["close"] - lower) / band_range,
        0.5,
    )
    df["bb_pct_b"] = pd.Series(df["bb_pct_b"], index=df.index).clip(0.0, 1.0)

    if spy_frame is not None and not spy_frame.empty:
        spy = spy_frame.sort_values("datetime").set_index("datetime")["close"]
        ticker_ret5 = np.log(df["close"] / df["close"].shift(5))
        spy_aligned = spy.reindex(pd.to_datetime(df["datetime"])).ffill()
        spy_ret5 = np.log(spy_aligned / spy_aligned.shift(5))
        df["ret_5_vs_spy"] = (ticker_ret5 - spy_ret5.values).fillna(0.0)
    else:
        df["ret_5_vs_spy"] = 0.0

    return df


def feature_columns() -> list[str]:
    return [
        "ret_1",
        "ret_5",
        "ret_20",
        "vol_20",
        "atr_14",
        "vol_z",
        "gap",
        "rsi_14",
        "macd",
        "macd_signal",
        "day_of_week",
        "bb_width",
        "bb_pct_b",
        "ret_5_vs_spy",
    ]


def resolve_model_features(payload: dict) -> list[str]:
    """Feature list used by a saved artifact (payload metadata or fitted estimator)."""
    saved = payload.get("feature_columns")
    if saved:
        return list(saved)
    classifiers = payload.get("classifiers", {})
    if classifiers:
        clf = next(iter(classifiers.values()))
        names = getattr(clf, "feature_names_in_", None)
        if names is not None:
            return list(names)
    return feature_columns()


def model_features_stale(payload: dict) -> bool:
    """True when the artifact was trained on a different feature set than the app uses now."""
    return set(resolve_model_features(payload)) != set(feature_columns())
