"""Tests for stale-feature detection in predict_symbol."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest

from ml.features import feature_columns


def _stale_payload() -> dict:
    """A payload whose feature_columns list differs from the current live set.

    Uses a strict subset of live columns so predict_symbol can still run after
    emitting the staleness warning (avoids KeyError on missing column).
    """
    live = feature_columns()
    # Drop the last feature to simulate an artifact trained on an older feature set.
    stale_features = live[:-1]
    clf_stub = _StubClf(stale_features)
    return {
        "feature_columns": stale_features,
        "classifiers": {"1d": clf_stub},
        "regressors": {"1d": clf_stub},
        "thresholds": {"1d": 0.5},
        "horizons": {"1d": 1},
    }


def _fresh_payload() -> dict:
    live = feature_columns()
    clf_stub = _StubClf(live)
    return {
        "feature_columns": list(live),
        "classifiers": {"1d": clf_stub},
        "regressors": {"1d": clf_stub},
        "thresholds": {"1d": 0.5},
        "horizons": {"1d": 1},
    }


class _StubClf:
    """Minimal classifier/regressor stub to avoid loading real XGBoost models."""
    def __init__(self, feature_names):
        self.feature_names_in_ = feature_names

    def predict_proba(self, X):
        return np.array([[0.4, 0.6]] * len(X))

    def predict(self, X):
        return np.array([0.01] * len(X))


def _make_bars(n: int = 60) -> pd.DataFrame:
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


def test_stale_features_emits_warning(monkeypatch, caplog):
    import ml.predict as pred_mod

    payload = _stale_payload()
    monkeypatch.setattr(pred_mod, "load_models", lambda *a, **kw: payload)

    bars = _make_bars()
    with caplog.at_level(logging.WARNING, logger="ml.predict"):
        pred_mod.predict_symbol(
            model_dir=None,
            symbol="AAPL",
            version="v1",
            bars=bars,
            horizons={"1d": 1},
        )

    assert any("stale_model_features" in r.message for r in caplog.records), (
        "Expected WARNING with 'stale_model_features' in message"
    )


def test_stale_features_lists_added_and_removed(monkeypatch, caplog):
    import ml.predict as pred_mod

    payload = _stale_payload()
    dropped = feature_columns()[-1]  # the feature missing from the stale payload
    monkeypatch.setattr(pred_mod, "load_models", lambda *a, **kw: payload)

    bars = _make_bars()
    with caplog.at_level(logging.WARNING, logger="ml.predict"):
        pred_mod.predict_symbol(
            model_dir=None,
            symbol="AAPL",
            version="v1",
            bars=bars,
            horizons={"1d": 1},
        )

    warning_msgs = [r.message for r in caplog.records if "stale_model_features" in r.message]
    assert warning_msgs, "No stale_model_features warning found"
    # The warning should list the feature added in the live set (removed from artifact)
    assert dropped in warning_msgs[0]


def test_fresh_features_no_warning(monkeypatch, caplog):
    import ml.predict as pred_mod

    payload = _fresh_payload()
    monkeypatch.setattr(pred_mod, "load_models", lambda *a, **kw: payload)

    bars = _make_bars()
    with caplog.at_level(logging.WARNING, logger="ml.predict"):
        pred_mod.predict_symbol(
            model_dir=None,
            symbol="AAPL",
            version="v1",
            bars=bars,
            horizons={"1d": 1},
        )

    assert not any("stale_model_features" in r.message for r in caplog.records)


def test_fail_on_stale_raises(monkeypatch):
    import ml.predict as pred_mod

    # Use a stale payload where the stale features ARE in the DataFrame so
    # fail_on_stale_features triggers before any KeyError from missing columns.
    payload = _stale_payload()
    monkeypatch.setattr(pred_mod, "load_models", lambda *a, **kw: payload)

    bars = _make_bars()
    with pytest.raises(ValueError, match="Stale model features"):
        pred_mod.predict_symbol(
            model_dir=None,
            symbol="AAPL",
            version="v1",
            bars=bars,
            horizons={"1d": 1},
            signal_config={"fail_on_stale_features": True},
        )
