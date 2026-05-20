from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from config_loader import load_thresholds, normalize_thresholds, train_cold_start
from data.store import DataStore
from ml.train import _load_retry_cfg, _resolve_training_tier, train_with_retries

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

HORIZONS = {"1d": 1, "3d": 3}


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
    promote_never: bool = True,
) -> dict[str, Any]:
    promo: dict[str, Any] = (
        {
            "aggregate_accuracy_min": 0.99,
            "per_horizon_accuracy_min": 0.99,
            "sharpe_min": 999.0,
            "max_drawdown_max": 0.0001,
        }
        if promote_never
        else {
            "aggregate_accuracy_min": 0.0,
            "per_horizon_accuracy_min": 0.0,
            "sharpe_min": -999.0,
            "max_drawdown_max": 999.0,
        }
    )
    retry: dict[str, Any] = {
        "strategy": "seed_only",
        "selection": "last_attempt",
        "log_attempt_comparison": True,
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
# Tests
# ---------------------------------------------------------------------------


def test_attempt_count_matches_max_retrain_attempts() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds(max_retrain_attempts=3, seeds=[10, 11, 12])

        result = train_with_retries(
            store=store,
            symbol="TEST",
            bars=_make_bars(),
            thresholds=cfg,
            model_dir=model_dir,
        )

        # With impossible gates, no attempt promotes — all 3 must run.
        assert result["promoted"] is False
        # Three registry rows must exist (one per attempt).
        rows = store.get_latest_model("TEST")
        assert rows is not None
        # The metrics JSON on the last row should report attempt=3.
        assert rows[1]["attempt"] == 3


def test_metrics_contain_registry_metadata() -> None:
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds(max_retrain_attempts=1, seeds=[77])

        result = train_with_retries(
            store=store,
            symbol="TEST",
            bars=_make_bars(),
            thresholds=cfg,
            model_dir=model_dir,
        )

        m = result["metrics"]
        assert m["seed"] == 77
        assert m["strategy"] == "seed_only"
        assert m["selection_reason"] == "last_attempt"
        assert "train_end" in m["split_boundaries"]
        assert "val_end" in m["split_boundaries"]
        assert m["split_boundaries"]["train_end"] < m["split_boundaries"]["val_end"]
        assert "val_aggregate_accuracy" in m
        assert "val_per_horizon_accuracy" in m
        assert "label_balance" in m


def test_seeds_from_config_are_used() -> None:
    """Seed stored in metrics must match the configured seeds list."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds(max_retrain_attempts=2, seeds=[200, 201])

        train_with_retries(
            store=store,
            symbol="TEST",
            bars=_make_bars(),
            thresholds=cfg,
            model_dir=model_dir,
        )

        # Last model in registry is attempt 2 with seed 201.
        _, last_metrics, _ = store.get_latest_model("TEST")
        assert last_metrics["seed"] == 201


def test_default_seeds_fall_back_to_42_plus_attempt() -> None:
    """Without a seeds list, seed must equal 42 + attempt (legacy behaviour)."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds(max_retrain_attempts=1, seeds=None)

        train_with_retries(
            store=store,
            symbol="TEST",
            bars=_make_bars(),
            thresholds=cfg,
            model_dir=model_dir,
        )

        _, last_metrics, _ = store.get_latest_model("TEST")
        # attempt=1 → seed = 42 + 1 = 43
        assert last_metrics["seed"] == 43


def test_load_retry_cfg_defaults() -> None:
    """_load_retry_cfg must fill all default keys when train_retry is absent."""
    cfg = _load_retry_cfg({})
    assert cfg["strategy"] == "seed_only"
    assert cfg["selection"] == "last_attempt"
    assert cfg["seeds"] is None
    assert cfg["log_attempt_comparison"] is True


def test_load_retry_cfg_overrides() -> None:
    cfg = _load_retry_cfg({"train_retry": {"strategy": "rolling_holdout", "seeds": [1, 2, 3]}})
    assert cfg["strategy"] == "rolling_holdout"
    assert cfg["seeds"] == [1, 2, 3]
    assert cfg["selection"] == "last_attempt"  # default still present


def test_attempt_summary_logged(caplog: pytest.LogCaptureFixture) -> None:
    """INFO summary must appear when log_attempt_comparison is true."""
    import logging

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds(max_retrain_attempts=2, seeds=[10, 11])

        with caplog.at_level(logging.INFO, logger="ml.train"):
            train_with_retries(
                store=store,
                symbol="TEST",
                bars=_make_bars(),
                thresholds=cfg,
                model_dir=model_dir,
            )

    summary_messages = [r.message for r in caplog.records if "train_attempt_summary" in r.message]
    assert len(summary_messages) == 1
    assert "symbol=TEST" in summary_messages[0]
    assert "total=2" in summary_messages[0]


def _thresholds_best_val(max_retrain_attempts: int = 2, seeds: list[int] | None = None) -> dict[str, Any]:
    """Config with best_val_score selection and pass-all gates."""
    base = _thresholds(max_retrain_attempts=max_retrain_attempts, seeds=seeds, promote_never=False)
    base["train_retry"]["selection"] = "best_val_score"
    base["train_retry"]["val_score_weights"] = {"accuracy": 0.5, "sharpe": 0.5}
    return base


def test_payload_contains_per_horizon_thresholds() -> None:
    """Model artifact must include a 'thresholds' dict with a key per horizon."""
    import pickle

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds(max_retrain_attempts=1, seeds=[42])

        result = train_with_retries(
            store=store, symbol="TEST", bars=_make_bars(), thresholds=cfg, model_dir=model_dir,
        )

        artifact_path = Path(result["artifact"])
        with artifact_path.open("rb") as fh:
            payload = pickle.load(fh)

        assert "thresholds" in payload
        for hz in HORIZONS:
            assert hz in payload["thresholds"]
            t = payload["thresholds"][hz]
            assert 0.45 <= t <= 0.65


def test_metrics_contain_per_horizon_thresholds() -> None:
    """Metrics dict returned by train_with_retries must expose per_horizon_thresholds."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds(max_retrain_attempts=1, seeds=[55])

        result = train_with_retries(
            store=store, symbol="TEST", bars=_make_bars(), thresholds=cfg, model_dir=model_dir,
        )

        m = result["metrics"]
        assert "per_horizon_thresholds" in m
        for hz in HORIZONS:
            assert hz in m["per_horizon_thresholds"]


def test_metrics_contain_val_sharpe_and_composite() -> None:
    """Metrics must include val_sharpe and val_composite fields."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds(max_retrain_attempts=1, seeds=[33])

        result = train_with_retries(
            store=store, symbol="TEST", bars=_make_bars(), thresholds=cfg, model_dir=model_dir,
        )

        m = result["metrics"]
        assert "val_sharpe" in m
        assert "val_composite" in m
        assert isinstance(m["val_sharpe"], float)
        assert isinstance(m["val_composite"], float)


def test_best_val_attempt_selected_when_last_is_worse(caplog: pytest.LogCaptureFixture) -> None:
    """With best_val_score, the returned val_composite equals the max logged across all attempts."""
    import logging
    import re

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds_best_val(max_retrain_attempts=3, seeds=[10, 11, 12])

        with caplog.at_level(logging.INFO, logger="ml.train"):
            result = train_with_retries(
                store=store, symbol="TEST", bars=_make_bars(), thresholds=cfg, model_dir=model_dir,
            )

        assert result["metrics"]["selection_reason"] == "best_val_score"

        # Parse val_composite values from the attempt summary log table rows
        summary = next(
            (r.message for r in caplog.records if "train_attempt_summary" in r.message), ""
        )
        # Each data row ends with a float for val_cmp
        composites = [float(m) for m in re.findall(r"(\d+\.\d+)\s+(?:True|False)", summary)]
        assert composites, "no composite values found in summary log"
        # Summary table rounds val_cmp to 3 decimals; allow display rounding slack.
        assert result["val_composite"] == pytest.approx(max(composites), abs=0.01)


def test_best_val_score_runs_all_attempts(caplog: pytest.LogCaptureFixture) -> None:
    """best_val_score never exits early — all max_retrain_attempts must run."""
    import logging

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        # promote_never=False so gates would pass on first attempt with last_attempt selection
        cfg = _thresholds_best_val(max_retrain_attempts=3, seeds=[1, 2, 3])

        with caplog.at_level(logging.DEBUG, logger="ml.train"):
            train_with_retries(
                store=store, symbol="TEST", bars=_make_bars(), thresholds=cfg, model_dir=model_dir,
            )

        attempt_logs = [r for r in caplog.records if r.message.startswith("attempt_result")]
        assert len(attempt_logs) == 3


def _thresholds_rolling(
    max_retrain_attempts: int = 3,
    seeds: list[int] | None = None,
    step_bars: int = 30,
    min_train_rows: int = 200,
    promote_never: bool = True,
) -> dict[str, Any]:
    base = _thresholds(max_retrain_attempts=max_retrain_attempts, seeds=seeds, promote_never=promote_never)
    base["train_retry"]["strategy"] = "rolling_holdout"
    base["train_retry"]["rolling"] = {"step_bars": step_bars, "min_train_rows": min_train_rows}
    return base


def test_rolling_holdout_completes_and_reports_strategy() -> None:
    """rolling_holdout strategy runs all attempts and records strategy in metrics."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds_rolling(max_retrain_attempts=3, seeds=[10, 11, 12])

        result = train_with_retries(
            store=store,
            symbol="TEST",
            bars=_make_bars(800),
            thresholds=cfg,
            model_dir=model_dir,
        )

        assert "metrics" in result
        assert result["metrics"]["strategy"] == "rolling_holdout"
        assert "holdout_end" in result["metrics"]["split_boundaries"]


def test_rolling_holdout_uses_different_split_windows(caplog: pytest.LogCaptureFixture) -> None:
    """Each attempt in rolling_holdout logs a distinct val_end."""
    import logging
    import re

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds_rolling(max_retrain_attempts=3, seeds=[10, 11, 12], step_bars=30)

        with caplog.at_level(logging.DEBUG, logger="ml.train"):
            train_with_retries(
                store=store,
                symbol="TEST",
                bars=_make_bars(800),
                thresholds=cfg,
                model_dir=model_dir,
            )

        split_logs = [
            r.message for r in caplog.records
            if r.message.startswith("split ") and "strategy=rolling_holdout" in r.message
        ]
        assert len(split_logs) == 3
        val_ends = [int(re.search(r"val_end=(\d+)", msg).group(1)) for msg in split_logs]
        assert len(set(val_ends)) == 3  # all distinct


def test_rolling_holdout_respects_min_train_rows() -> None:
    """Attempts are skipped when min_train_rows would be violated."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        # 500 rows: after labels/dropna ~497; with step=200 and min_train=200:
        # k=0: train_end ~=317, k=1: ~=117 < 200 -> only 1 split
        cfg = _thresholds_rolling(
            max_retrain_attempts=3, seeds=[10, 11, 12], step_bars=200, min_train_rows=300
        )

        result = train_with_retries(
            store=store,
            symbol="TEST",
            bars=_make_bars(500),
            thresholds=cfg,
            model_dir=model_dir,
        )

        # Only 1 split would be valid; result should be from attempt 1
        assert result["metrics"]["attempt"] == 1


def _thresholds_purged_cv(
    max_retrain_attempts: int = 2,
    seeds: list[int] | None = None,
    n_splits: int = 2,
    purge_bars: int = 3,
    embargo_bars: int = 2,
    promote_never: bool = True,
) -> dict[str, Any]:
    """Config for purged_cv strategy with small purge/embargo values safe for test data."""
    base = _thresholds(max_retrain_attempts=max_retrain_attempts, seeds=seeds, promote_never=promote_never)
    base["train_retry"]["strategy"] = "purged_cv"
    base["train_retry"]["purged_cv"] = {
        "n_splits": n_splits,
        "purge_bars": purge_bars,
        "embargo_bars": embargo_bars,
    }
    return base


def test_purged_cv_runs_and_records_strategy() -> None:
    """purged_cv strategy completes and stores 'purged_cv' in metrics."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds_purged_cv(max_retrain_attempts=1, seeds=[42])

        result = train_with_retries(
            store=store, symbol="TEST", bars=_make_bars(), thresholds=cfg, model_dir=model_dir,
        )

        assert "metrics" in result
        assert result["metrics"]["strategy"] == "purged_cv"


def test_purged_cv_fold_metrics_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Each purged CV fold emits a cv_fold debug log line per horizon."""
    import logging

    n_splits = 2
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds_purged_cv(max_retrain_attempts=1, seeds=[10], n_splits=n_splits)

        with caplog.at_level(logging.DEBUG, logger="ml.train"):
            train_with_retries(
                store=store, symbol="TEST", bars=_make_bars(), thresholds=cfg, model_dir=model_dir,
            )

    fold_logs = [r for r in caplog.records if r.message.startswith("cv_fold")]
    # n_folds × n_horizons log lines expected
    assert len(fold_logs) >= n_splits * len(HORIZONS)


def test_purged_cv_holdout_evaluated_once_per_attempt(caplog: pytest.LogCaptureFixture) -> None:
    """Holdout gate check runs exactly once per attempt, not once per fold."""
    import logging

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds_purged_cv(max_retrain_attempts=2, seeds=[10, 11])

        with caplog.at_level(logging.DEBUG, logger="ml.train"):
            train_with_retries(
                store=store, symbol="TEST", bars=_make_bars(), thresholds=cfg, model_dir=model_dir,
            )

    attempt_logs = [r for r in caplog.records if r.message.startswith("attempt_result")]
    assert len(attempt_logs) == 2


def test_purged_cv_val_metrics_present() -> None:
    """Result must include CV-averaged val_aggregate_accuracy and val_sharpe."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds_purged_cv(max_retrain_attempts=1, seeds=[42])

        result = train_with_retries(
            store=store, symbol="TEST", bars=_make_bars(), thresholds=cfg, model_dir=model_dir,
        )

        m = result["metrics"]
        assert "val_aggregate_accuracy" in m
        assert "val_sharpe" in m
        assert isinstance(m["val_aggregate_accuracy"], float)
        assert isinstance(m["val_sharpe"], float)


def test_purged_cv_artifact_has_thresholds() -> None:
    """Model artifact produced by purged_cv must include per-horizon thresholds."""
    import pickle

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds_purged_cv(max_retrain_attempts=1, seeds=[42])

        result = train_with_retries(
            store=store, symbol="TEST", bars=_make_bars(), thresholds=cfg, model_dir=model_dir,
        )

        with Path(result["artifact"]).open("rb") as fh:
            payload = pickle.load(fh)

        assert "thresholds" in payload
        for hz in HORIZONS:
            assert hz in payload["thresholds"]


def test_load_retry_cfg_purged_cv_defaults() -> None:
    """purged_cv defaults must be present even when only strategy is set."""
    cfg = _load_retry_cfg({"train_retry": {"strategy": "purged_cv"}})
    assert cfg["purged_cv"]["n_splits"] == 3
    assert cfg["purged_cv"]["purge_bars"] == 22
    assert "embargo_bars" in cfg["purged_cv"]


def test_purged_cv_partial_config_merges_defaults() -> None:
    """Partial purged_cv override must keep unset defaults."""
    cfg = _load_retry_cfg({"train_retry": {"strategy": "purged_cv", "purged_cv": {"n_splits": 5}}})
    assert cfg["purged_cv"]["n_splits"] == 5
    assert cfg["purged_cv"]["purge_bars"] == 22  # default preserved


def _thresholds_cold_start(
    max_retrain_attempts: int = 1,
    seeds: list[int] | None = None,
    promote_never: bool = True,
    cold_enabled: bool = True,
) -> dict[str, Any]:
    base = _thresholds(
        max_retrain_attempts=max_retrain_attempts,
        seeds=seeds,
        promote_never=promote_never,
    )
    base["train"]["min_rows"] = 1200
    base["train"]["val_size"] = 126
    base["train"]["holdout_size"] = 180
    base["train_cold_start"] = {
        "enabled": cold_enabled,
        "min_rows": 600,
        "val_size": 60,
        "holdout_size": 60,
        "promotion": {
            "aggregate_accuracy_min": 0.57,
            "sharpe_min": 0.60,
            "max_drawdown_max": 0.12,
        },
        "tag": "cold_start",
        "confidence_discount": 0.1,
    }
    return normalize_thresholds(base)


def test_cold_start_fallback_on_700_bars(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        cfg = _thresholds_cold_start(max_retrain_attempts=1, seeds=[42])

        with caplog.at_level(logging.INFO, logger="ml.train"):
            result = train_with_retries(
                store=store,
                symbol="TEST",
                bars=_make_bars(700),
                thresholds=cfg,
                model_dir=model_dir,
            )

        assert "cold_start_training" in caplog.text
        assert "cold_start=True" in caplog.text
        assert result["metrics"]["cold_start"] is True
        assert result["metrics"]["tag"] == "cold_start"
        assert result["metrics"]["split_boundaries"]["val_end"] - result["metrics"]["split_boundaries"]["train_end"] == 60

        with store._connect() as conn:
            row = conn.execute(
                "SELECT tag FROM model_registry WHERE symbol = ?", ("TEST",)
            ).fetchone()
        assert row is not None
        assert row[0] == "cold_start"


def test_below_cold_start_min_fails_with_info_log(caplog: pytest.LogCaptureFixture) -> None:
    import logging

    cfg = _thresholds_cold_start()
    with pytest.raises(ValueError, match="Not enough rows"):
        with caplog.at_level(logging.INFO, logger="ml.train"):
            _resolve_training_tier("TEST", 500, cfg)
    assert "insufficient_history" in caplog.text


def test_resolve_training_tier_full_when_enough_bars() -> None:
    cfg = _thresholds_cold_start()
    cfg["promotion"] = {
        "aggregate_accuracy_min": 0.55,
        "per_horizon_accuracy_min": 0.50,
        "sharpe_min": 0.50,
        "max_drawdown_max": 0.15,
    }
    train_cfg, promo_cfg, tag, is_cold = _resolve_training_tier("TEST", 1500, cfg)
    assert is_cold is False
    assert tag is None
    assert train_cfg["val_size"] == 126
    assert train_cfg["holdout_size"] == 180
    assert train_cfg["min_rows"] == 1200
    assert promo_cfg["aggregate_accuracy_min"] == 0.55


def test_thresholds_example_phase_d_train_sizes() -> None:
    """Phase D defaults live in the committed example config (thresholds.yaml is local-only)."""
    cfg = load_thresholds(Path("config/thresholds.example.yaml"))
    train = cfg["train"]
    assert train["min_rows"] == 1200
    assert train["val_size"] == 126
    assert train["holdout_size"] == 180
    assert cfg["train_retry"]["strategy"] == "purged_cv"
    assert cfg["train_retry"]["purged_cv"]["n_splits"] == 3
    assert cfg["train_retry"]["purged_cv"]["purge_bars"] == 22
    assert cfg["train_retry"]["purged_cv"]["embargo_bars"] == 5


def test_insufficient_rows_below_full_min_when_cold_start_disabled(
    caplog: pytest.LogCaptureFixture,
) -> None:
    import logging

    cfg = _thresholds_cold_start(cold_enabled=False)
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        with pytest.raises(ValueError, match="Not enough rows"):
            with caplog.at_level(logging.INFO, logger="ml.train"):
                train_with_retries(
                    store=store,
                    symbol="TEST",
                    bars=_make_bars(1000),
                    thresholds=cfg,
                    model_dir=tmp / "models",
                )
    assert "insufficient_history" in caplog.text


def test_train_cold_start_normalized_defaults() -> None:
    cfg = normalize_thresholds({"train_cold_start": {"enabled": True}})
    cold = train_cold_start(cfg)
    assert cold["min_rows"] == 600
    assert cold["promotion"]["sharpe_min"] == 0.60


def test_promotion_outcome_unchanged_with_new_config() -> None:
    """Adding train_retry config must not flip promotion when gates are passable."""
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        store = _make_store(tmp)
        model_dir = tmp / "models"
        # Pass-all gates
        cfg = _thresholds(max_retrain_attempts=3, seeds=[1, 2, 3], promote_never=False)

        result = train_with_retries(
            store=store,
            symbol="TEST",
            bars=_make_bars(),
            thresholds=cfg,
            model_dir=model_dir,
        )

        # Promotion outcome is driven entirely by gates, not by new metadata.
        assert result["promoted"] == (not result["failed_gates"])
