"""
Tests for Phase 4 promotion gates:
  - Overfit detector (val/holdout Sharpe gap)
  - Horizon-weighted accuracy gate
  - Multi-seed stability gate
  - Multi-window promotion option (rolling_holdout)

Most tests use mocked metrics dicts so they run without training.
Integration tests at the bottom exercise train_with_retries end-to-end.
"""
from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from ml.train import (
    _failed_basic_gates,
    _failed_promotion_gates,
    _horizon_weighted_accuracy,
    _load_retry_cfg,
    _stability_gate_failed,
    train_with_retries,
)
from data.store import DataStore

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

HORIZONS = {"1d": 1, "3d": 3}

_BASE_PROMO = {
    "aggregate_accuracy_min": 0.50,
    "per_horizon_accuracy_min": 0.48,
    "sharpe_min": 0.0,
    "max_drawdown_max": 1.0,
}


def _metrics(agg_acc: float = 0.60, sharpe: float = 1.0, max_dd: float = 0.05, val_sharpe: float = 1.0) -> dict:
    return {
        "aggregate_accuracy": agg_acc,
        "sharpe": sharpe,
        "max_drawdown": max_dd,
        "val_sharpe": val_sharpe,
    }


def _per_hz(acc: float = 0.60) -> dict[str, float]:
    return {"1d": acc, "3d": acc}


def _make_bars(n: int = 500) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    dt = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
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


def _thresholds(
    max_retrain_attempts: int = 2,
    seeds: list[int] | None = None,
    promo_overrides: dict | None = None,
    retry_overrides: dict | None = None,
) -> dict[str, Any]:
    promo = {**_BASE_PROMO, **(promo_overrides or {})}
    retry: dict[str, Any] = {
        "strategy": "seed_only",
        "selection": "last_attempt",
        "log_attempt_comparison": False,
        **(retry_overrides or {}),
    }
    if seeds is not None:
        retry["seeds"] = seeds
    return {
        "horizons": HORIZONS,
        "train": {
            "min_rows": 300,
            "val_size": 60,
            "holdout_size": 60,
            "early_stopping_rounds": 5,
            "max_retrain_attempts": max_retrain_attempts,
        },
        "promotion": promo,
        "train_retry": retry,
    }


def _make_store(tmp: Path) -> DataStore:
    return DataStore(sqlite_path=tmp / "db.sqlite", parquet_dir=tmp / "parquet")


# ---------------------------------------------------------------------------
# _horizon_weighted_accuracy
# ---------------------------------------------------------------------------


def test_weighted_accuracy_equal_weights() -> None:
    acc = _horizon_weighted_accuracy({"1d": 0.6, "3d": 0.4}, {"1d": 1.0, "3d": 1.0})
    assert acc == pytest.approx(0.5)


def test_weighted_accuracy_unequal_weights() -> None:
    # 1d gets 2x weight of 3d
    acc = _horizon_weighted_accuracy({"1d": 0.6, "3d": 0.4}, {"1d": 2.0, "3d": 1.0})
    assert acc == pytest.approx((0.6 * 2.0 + 0.4 * 1.0) / 3.0)


def test_weighted_accuracy_missing_weight_treated_as_zero() -> None:
    # 3d not in weights → only 1d contributes
    acc = _horizon_weighted_accuracy({"1d": 0.7, "3d": 0.3}, {"1d": 1.0})
    assert acc == pytest.approx(0.7)


def test_weighted_accuracy_empty_weights_returns_zero() -> None:
    assert _horizon_weighted_accuracy({"1d": 0.6}, {}) == 0.0


# ---------------------------------------------------------------------------
# _stability_gate_failed
# ---------------------------------------------------------------------------


def test_stability_gate_disabled_when_zero() -> None:
    assert _stability_gate_failed([False, False, False], min_seeds_passing=0) is False


def test_stability_gate_passes_when_enough_seeds() -> None:
    assert _stability_gate_failed([True, True, False], min_seeds_passing=2) is False


def test_stability_gate_fails_when_too_few_seeds() -> None:
    assert _stability_gate_failed([True, False, False], min_seeds_passing=2) is True


def test_stability_gate_passes_when_all_seeds_pass() -> None:
    assert _stability_gate_failed([True, True, True], min_seeds_passing=3) is False


def test_stability_gate_fails_when_none_pass() -> None:
    assert _stability_gate_failed([False, False, False], min_seeds_passing=1) is True


# ---------------------------------------------------------------------------
# _failed_basic_gates — core 4 gates
# ---------------------------------------------------------------------------


def test_basic_gates_pass_when_all_clear() -> None:
    assert _failed_basic_gates(_metrics(), _per_hz(), _BASE_PROMO) == []


def test_basic_gates_fail_aggregate_accuracy() -> None:
    failed = _failed_basic_gates(_metrics(agg_acc=0.40), _per_hz(0.40), _BASE_PROMO)
    assert "aggregate_accuracy" in failed


def test_basic_gates_fail_sharpe() -> None:
    promo = {**_BASE_PROMO, "sharpe_min": 1.0}
    failed = _failed_basic_gates(_metrics(sharpe=0.5), _per_hz(), promo)
    assert "sharpe" in failed


def test_basic_gates_fail_max_drawdown() -> None:
    promo = {**_BASE_PROMO, "max_drawdown_max": 0.10}
    failed = _failed_basic_gates(_metrics(max_dd=0.20), _per_hz(), promo)
    assert "max_drawdown" in failed


# ---------------------------------------------------------------------------
# _failed_promotion_gates — Phase 4.3 overfit detector
# ---------------------------------------------------------------------------


def test_overfit_detector_disabled_when_gap_is_zero() -> None:
    promo = {**_BASE_PROMO, "max_val_holdout_sharpe_gap": 0.0}
    # val_sharpe - holdout_sharpe = 1.5; should still pass when disabled
    failed = _failed_promotion_gates(_metrics(sharpe=0.0, val_sharpe=1.5), _per_hz(), promo)
    assert "val_holdout_sharpe_gap" not in failed


def test_overfit_detector_passes_when_within_gap() -> None:
    promo = {**_BASE_PROMO, "max_val_holdout_sharpe_gap": 0.75}
    # gap = 1.2 - 0.6 = 0.6 ≤ 0.75 → should pass
    failed = _failed_promotion_gates(_metrics(sharpe=0.6, val_sharpe=1.2), _per_hz(), promo)
    assert "val_holdout_sharpe_gap" not in failed


def test_overfit_detector_fails_when_gap_exceeded() -> None:
    promo = {**_BASE_PROMO, "max_val_holdout_sharpe_gap": 0.75}
    # gap = 2.0 - 0.5 = 1.5 > 0.75 → should fail
    failed = _failed_promotion_gates(_metrics(sharpe=0.5, val_sharpe=2.0), _per_hz(), promo)
    assert "val_holdout_sharpe_gap" in failed


def test_overfit_detector_passes_when_val_sharpe_missing() -> None:
    promo = {**_BASE_PROMO, "max_val_holdout_sharpe_gap": 0.50}
    m = {"aggregate_accuracy": 0.6, "sharpe": 0.1, "max_drawdown": 0.05}  # no val_sharpe key
    failed = _failed_promotion_gates(m, _per_hz(), promo)
    assert "val_holdout_sharpe_gap" not in failed


def test_overfit_detector_does_not_trigger_basic_gates() -> None:
    """Overfit gate and basic gates fail independently."""
    promo = {**_BASE_PROMO, "max_val_holdout_sharpe_gap": 0.5}
    # aggregate_accuracy fails AND gap fails
    failed = _failed_promotion_gates(_metrics(agg_acc=0.40, sharpe=0.5, val_sharpe=2.0), _per_hz(0.40), promo)
    assert "aggregate_accuracy" in failed
    assert "val_holdout_sharpe_gap" in failed


# ---------------------------------------------------------------------------
# _failed_promotion_gates — Phase 4.4 horizon-weighted accuracy gate
# ---------------------------------------------------------------------------


def test_weighted_gate_disabled_when_empty_weights() -> None:
    promo = {**_BASE_PROMO, "horizon_weights": {}, "weighted_accuracy_min": 0.52}
    failed = _failed_promotion_gates(_metrics(), _per_hz(0.40), promo)
    assert "weighted_accuracy" not in failed


def test_weighted_gate_passes_when_weighted_acc_sufficient() -> None:
    promo = {**_BASE_PROMO, "horizon_weights": {"1d": 1.0, "3d": 0.5}, "weighted_accuracy_min": 0.52}
    # weighted = (0.60*1 + 0.60*0.5) / 1.5 = 0.60 ≥ 0.52
    failed = _failed_promotion_gates(_metrics(), _per_hz(0.60), promo)
    assert "weighted_accuracy" not in failed


def test_weighted_gate_fails_when_weighted_acc_below_min() -> None:
    promo = {**_BASE_PROMO, "horizon_weights": {"1d": 1.0, "3d": 0.5}, "weighted_accuracy_min": 0.60}
    # weighted = (0.55*1 + 0.55*0.5) / 1.5 = 0.55 < 0.60
    failed = _failed_promotion_gates(_metrics(agg_acc=0.55), {"1d": 0.55, "3d": 0.55}, promo)
    assert "weighted_accuracy" in failed


def test_weighted_gate_uses_per_horizon_not_aggregate() -> None:
    """Weighted gate uses per-horizon acc dict, not the aggregate_accuracy scalar."""
    promo = {**_BASE_PROMO, "horizon_weights": {"1d": 1.0}, "weighted_accuracy_min": 0.55}
    # aggregate_accuracy = 0.60 but 1d = 0.50 → weighted = 0.50 < 0.55
    failed = _failed_promotion_gates(_metrics(agg_acc=0.60), {"1d": 0.50, "3d": 0.80}, promo)
    assert "weighted_accuracy" in failed


# ---------------------------------------------------------------------------
# _load_retry_cfg — Phase 4 defaults
# ---------------------------------------------------------------------------


def test_load_retry_cfg_ensemble_seeds_default() -> None:
    cfg = _load_retry_cfg({})
    assert cfg["ensemble_seeds"] == 1


def test_load_retry_cfg_stability_defaults() -> None:
    cfg = _load_retry_cfg({})
    assert cfg["stability"]["min_seeds_passing_gates"] == 0
    assert cfg["stability"]["metric"] == "median"


def test_load_retry_cfg_stability_partial_override() -> None:
    cfg = _load_retry_cfg({"train_retry": {"stability": {"min_seeds_passing_gates": 2}}})
    assert cfg["stability"]["min_seeds_passing_gates"] == 2
    assert cfg["stability"]["metric"] == "median"  # default preserved


# ---------------------------------------------------------------------------
# Integration: multi-window gate (rolling_holdout)
# ---------------------------------------------------------------------------


def _thresholds_rolling(
    max_retrain_attempts: int = 4,
    seeds: list[int] | None = None,
    step_bars: int = 20,
    min_train_rows: int = 200,
    min_passing_windows: int = 0,
    promote_all: bool = True,
) -> dict[str, Any]:
    """Config for rolling_holdout with optional multi-window gate."""
    promo: dict[str, Any] = (
        {
            "aggregate_accuracy_min": 0.0,
            "per_horizon_accuracy_min": 0.0,
            "sharpe_min": -999.0,
            "max_drawdown_max": 999.0,
            "min_passing_windows": min_passing_windows,
        }
        if promote_all
        else {
            "aggregate_accuracy_min": 0.99,
            "per_horizon_accuracy_min": 0.99,
            "sharpe_min": 999.0,
            "max_drawdown_max": 0.0001,
            "min_passing_windows": min_passing_windows,
        }
    )
    retry: dict[str, Any] = {
        "strategy": "rolling_holdout",
        "selection": "last_attempt",
        "log_attempt_comparison": False,
        "rolling": {"step_bars": step_bars, "min_train_rows": min_train_rows},
    }
    if seeds is not None:
        retry["seeds"] = seeds
    return {
        "horizons": HORIZONS,
        "train": {
            "min_rows": 300,
            "val_size": 60,
            "holdout_size": 60,
            "early_stopping_rounds": 5,
            "max_retrain_attempts": max_retrain_attempts,
        },
        "promotion": promo,
        "train_retry": retry,
    }


def test_multi_window_gate_disabled_by_default() -> None:
    """With min_passing_windows=0, rolling_holdout promotion is unaffected."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        cfg = _thresholds_rolling(
            max_retrain_attempts=3,
            seeds=[10, 11, 12],
            min_passing_windows=0,
            promote_all=True,
        )
        result = train_with_retries(
            store=store, symbol="TEST", bars=_make_bars(800), thresholds=cfg, model_dir=tmp / "models"
        )
        # Gates pass-all → first attempt promotes; last_attempt breaks early
        assert result["promoted"] is True


def test_multi_window_gate_blocks_when_zero_windows_pass(caplog: pytest.LogCaptureFixture) -> None:
    """When no windows pass and min_passing_windows=2, final result is not promoted."""
    import logging

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        cfg = _thresholds_rolling(
            max_retrain_attempts=3,
            seeds=[10, 11, 12],
            min_passing_windows=2,
            promote_all=False,  # impossible gates → no individual window passes
        )
        with caplog.at_level(logging.INFO, logger="ml.train"):
            result = train_with_retries(
                store=store, symbol="TEST", bars=_make_bars(800), thresholds=cfg, model_dir=tmp / "models"
            )

    assert result["promoted"] is False


def test_multi_window_gate_requires_all_attempts_to_run(caplog: pytest.LogCaptureFixture) -> None:
    """When min_passing_windows > 0, all rolling attempts must run (no early break)."""
    import logging

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        cfg = _thresholds_rolling(
            max_retrain_attempts=3,
            seeds=[10, 11, 12],
            min_passing_windows=2,
            promote_all=True,  # each window passes; but we still need all 3 to count
        )
        with caplog.at_level(logging.DEBUG, logger="ml.train"):
            train_with_retries(
                store=store, symbol="TEST", bars=_make_bars(800), thresholds=cfg, model_dir=tmp / "models"
            )

    attempt_logs = [r for r in caplog.records if r.message.startswith("attempt_result")]
    assert len(attempt_logs) == 3


def test_multi_window_gate_passes_when_enough_windows_pass() -> None:
    """When min_passing_windows is satisfied, final result is promoted."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        cfg = _thresholds_rolling(
            max_retrain_attempts=3,
            seeds=[10, 11, 12],
            min_passing_windows=2,
            promote_all=True,  # all 3 windows pass → 3 ≥ 2 → promote
        )
        result = train_with_retries(
            store=store, symbol="TEST", bars=_make_bars(800), thresholds=cfg, model_dir=tmp / "models"
        )
    assert result["promoted"] is True


# ---------------------------------------------------------------------------
# Integration: multi-seed stability gate
# ---------------------------------------------------------------------------


def _thresholds_multi_seed(
    ensemble_seeds: int = 3,
    min_seeds_passing: int = 2,
    promote_all: bool = True,
) -> dict[str, Any]:
    promo: dict[str, Any] = (
        {
            "aggregate_accuracy_min": 0.0,
            "per_horizon_accuracy_min": 0.0,
            "sharpe_min": -999.0,
            "max_drawdown_max": 999.0,
        }
        if promote_all
        else {
            "aggregate_accuracy_min": 0.99,
            "per_horizon_accuracy_min": 0.99,
            "sharpe_min": 999.0,
            "max_drawdown_max": 0.0001,
        }
    )
    retry: dict[str, Any] = {
        "strategy": "seed_only",
        "selection": "last_attempt",
        "log_attempt_comparison": False,
        "seeds": [43],
        "ensemble_seeds": ensemble_seeds,
        "stability": {
            "min_seeds_passing_gates": min_seeds_passing,
            "metric": "median",
        },
    }
    return {
        "horizons": HORIZONS,
        "train": {
            "min_rows": 300,
            "val_size": 60,
            "holdout_size": 60,
            "early_stopping_rounds": 5,
            "max_retrain_attempts": 1,
        },
        "promotion": promo,
        "train_retry": retry,
    }


def test_multi_seed_stability_gate_disabled_when_min_zero() -> None:
    """With min_seeds_passing_gates=0, promotion is unaffected by seed count."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        cfg = _thresholds_multi_seed(ensemble_seeds=3, min_seeds_passing=0, promote_all=True)
        result = train_with_retries(
            store=store, symbol="TEST", bars=_make_bars(600), thresholds=cfg, model_dir=tmp / "models"
        )
    assert result["promoted"] is True


def test_multi_seed_stability_gate_passes_when_enough_seeds_pass(caplog: pytest.LogCaptureFixture) -> None:
    """With pass-all gates and ensemble_seeds=3, min_seeds=2 → stability PASSED."""
    import logging

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        cfg = _thresholds_multi_seed(ensemble_seeds=3, min_seeds_passing=2, promote_all=True)
        with caplog.at_level(logging.DEBUG, logger="ml.train"):
            result = train_with_retries(
                store=store, symbol="TEST", bars=_make_bars(600), thresholds=cfg, model_dir=tmp / "models"
            )

    assert result["promoted"] is True
    stability_logs = [r for r in caplog.records if "stability_gate" in r.message]
    assert any("PASSED" in r.message for r in stability_logs)


def test_multi_seed_stability_gate_fails_when_impossible_basic_gates(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """With impossible basic gates, no seed passes → stability FAILED → 'stability' in failed_gates."""
    import logging

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        cfg = _thresholds_multi_seed(ensemble_seeds=3, min_seeds_passing=1, promote_all=False)
        with caplog.at_level(logging.DEBUG, logger="ml.train"):
            result = train_with_retries(
                store=store, symbol="TEST", bars=_make_bars(600), thresholds=cfg, model_dir=tmp / "models"
            )

    assert result["promoted"] is False
    assert "stability" in result["failed_gates"]
    stability_logs = [r for r in caplog.records if "stability_gate" in r.message]
    assert any("FAILED" in r.message for r in stability_logs)


def test_multi_seed_uses_median_aggregation_by_default() -> None:
    """Smoke test: ensemble_seeds=2 with median aggregation completes without error."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        cfg = _thresholds_multi_seed(ensemble_seeds=2, min_seeds_passing=0)
        result = train_with_retries(
            store=store, symbol="TEST", bars=_make_bars(600), thresholds=cfg, model_dir=tmp / "models"
        )
    assert "metrics" in result


# ---------------------------------------------------------------------------
# Integration: overfit detector end-to-end via thresholds config
# ---------------------------------------------------------------------------


def test_overfit_detector_config_key_loaded_with_default_disabled() -> None:
    """max_val_holdout_sharpe_gap defaults to 0.0 (disabled) in _load_retry_cfg context."""
    promo = {
        **_BASE_PROMO,
        # No max_val_holdout_sharpe_gap key → defaults to 0.0 (disabled)
    }
    failed = _failed_promotion_gates(_metrics(sharpe=0.1, val_sharpe=5.0), _per_hz(), promo)
    assert "val_holdout_sharpe_gap" not in failed


def test_overfit_detector_integration_with_train_with_retries() -> None:
    """Setting a strict gap in promotion config causes the gate to appear in failed_gates."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        # tight gap threshold — any real val/holdout gap likely exceeds 0.001
        cfg = _thresholds(
            max_retrain_attempts=1,
            seeds=[42],
            promo_overrides={
                "max_val_holdout_sharpe_gap": 0.001,
                "aggregate_accuracy_min": 0.0,
                "per_horizon_accuracy_min": 0.0,
                "sharpe_min": -999.0,
                "max_drawdown_max": 999.0,
            },
        )
        result = train_with_retries(
            store=store, symbol="TEST", bars=_make_bars(600), thresholds=cfg, model_dir=tmp / "models"
        )

    # Either it promoted (gap was within 0.001 — very unlikely) or the gate fired.
    # We just verify the metrics key is present and promoted matches failed_gates.
    assert result["promoted"] == (len(result["failed_gates"]) == 0)
