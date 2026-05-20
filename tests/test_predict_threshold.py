"""Tests for predict_symbol threshold fallback behaviour (train/serve skew fix)."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from ml.predict import predict_symbol
from sda_types import HorizonPrediction


# ---------------------------------------------------------------------------
# Stub helpers
# ---------------------------------------------------------------------------

HORIZONS = {"1d": 1, "3d": 3}
FEAT_COLS = ["ret_1", "ret_5", "ret_20", "vol_20", "vol_z", "gap", "rsi_14", "macd", "macd_signal", "day_of_week"]


class _FakeClf:
    def predict_proba(self, X):
        return np.array([[0.3, 0.7]])


class _FakeReg:
    def predict(self, X):
        return np.array([0.02])


def _payload(thresholds: dict[str, float] | None = None) -> dict:
    clfs = {hz: _FakeClf() for hz in HORIZONS}
    regs = {hz: _FakeReg() for hz in HORIZONS}
    p: dict = {"classifiers": clfs, "regressors": regs, "feature_columns": FEAT_COLS}
    if thresholds is not None:
        p["thresholds"] = thresholds
    return p


def _fake_feat_df() -> pd.DataFrame:
    row = {col: 0.1 for col in FEAT_COLS}
    return pd.DataFrame([row])


def _call_predict(payload: dict, fallback: float = 0.60) -> list[HorizonPrediction]:
    bars = pd.DataFrame({"close": [100.0]})
    with (
        patch("ml.predict.load_models", return_value=payload),
        patch("ml.predict.build_features", return_value=_fake_feat_df()),
    ):
        result = predict_symbol(
            model_dir=".",
            symbol="TEST",
            version="v1",
            bars=bars,
            horizons=HORIZONS,
            ml_buy_threshold_fallback=fallback,
        )
    return result.predictions


# ---------------------------------------------------------------------------
# predict_symbol threshold tests
# ---------------------------------------------------------------------------

def test_old_artifact_uses_fallback_not_zero() -> None:
    """Old artifacts (no 'thresholds' key) must store the fallback, not 0.0."""
    preds = _call_predict(_payload(thresholds=None), fallback=0.65)
    for p in preds:
        assert p.threshold == pytest.approx(0.65), (
            f"horizon {p.horizon}: expected fallback 0.65, got {p.threshold}"
        )


def test_old_artifact_default_fallback_is_0_60() -> None:
    """Default fallback is 0.60 when caller omits the parameter."""
    bars = pd.DataFrame({"close": [100.0]})
    with (
        patch("ml.predict.load_models", return_value=_payload(thresholds=None)),
        patch("ml.predict.build_features", return_value=_fake_feat_df()),
    ):
        result = predict_symbol(".", "TEST", "v1", bars, HORIZONS)
    for p in result.predictions:
        assert p.threshold == pytest.approx(0.60)


def test_saved_thresholds_are_used_verbatim() -> None:
    """Artifacts with saved thresholds must not be overridden by the fallback."""
    saved = {"1d": 0.55, "3d": 0.72}
    preds = _call_predict(_payload(thresholds=saved), fallback=0.60)
    by_hz = {p.horizon: p for p in preds}
    assert by_hz["1d"].threshold == pytest.approx(0.55)
    assert by_hz["3d"].threshold == pytest.approx(0.72)


def test_partial_saved_thresholds_fallback_for_missing() -> None:
    """Horizons absent from saved thresholds use the fallback."""
    saved = {"1d": 0.58}
    preds = _call_predict(_payload(thresholds=saved), fallback=0.65)
    by_hz = {p.horizon: p for p in preds}
    assert by_hz["1d"].threshold == pytest.approx(0.58)
    assert by_hz["3d"].threshold == pytest.approx(0.65)
