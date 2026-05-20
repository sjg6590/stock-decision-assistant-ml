from __future__ import annotations

import pandas as pd


def add_multi_horizon_labels(frame: pd.DataFrame, horizons: dict[str, int]) -> pd.DataFrame:
    df = frame.copy()
    for name, bars in horizons.items():
        fwd = df["close"].shift(-bars)
        ret = (fwd - df["close"]) / df["close"]
        df[f"ret_{name}"] = ret
        df[f"up_{name}"] = (ret > 0).astype(int)
    return df
