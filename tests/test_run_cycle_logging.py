"""Tests for issue #7: silent fuse/alert failures must always produce a log line.

Acceptance criterion: run_cycle logs an ERROR with exc_info for every symbol
whose cycle raises, and continues processing remaining symbols.
"""
from __future__ import annotations

import logging
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from main import run_cycle


def _make_settings() -> MagicMock:
    settings = MagicMock()
    settings.sqlite_path = ":memory:"
    settings.parquet_dir = "/tmp/test_parquet"
    return settings


def test_run_cycle_logs_error_when_symbol_cycle_raises(caplog: pytest.LogCaptureFixture) -> None:
    """When run_symbol_cycle raises, run_cycle must emit an ERROR log for that symbol."""
    with (
        patch("main.DataStore"),
        patch("main._load_thresholds", return_value={}),
        patch("main.build_client"),
        patch("main._load_reference_bars", return_value=pd.DataFrame()),
        patch("main.run_symbol_cycle", side_effect=RuntimeError("fuse exploded")),
    ):
        with caplog.at_level(logging.ERROR, logger="stock-assistant"):
            run_cycle(_make_settings(), symbols=["AAPL"])

    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any(
        "cycle failed" in r.message and "AAPL" in r.message for r in errors
    ), f"Expected error log for AAPL; got: {[r.message for r in errors]}"


def test_run_cycle_continues_after_symbol_failure(caplog: pytest.LogCaptureFixture) -> None:
    """A failure on one symbol must not prevent subsequent symbols from being processed."""
    processed: list[str] = []

    def _fake_cycle(_settings, _store, symbol, *args, **kwargs) -> None:
        processed.append(symbol)
        if symbol == "AAPL":
            raise RuntimeError("fuse exploded")

    with (
        patch("main.DataStore"),
        patch("main._load_thresholds", return_value={}),
        patch("main.build_client"),
        patch("main._load_reference_bars", return_value=pd.DataFrame()),
        patch("main.run_symbol_cycle", side_effect=_fake_cycle),
    ):
        with caplog.at_level(logging.ERROR, logger="stock-assistant"):
            run_cycle(_make_settings(), symbols=["AAPL", "IBM"])

    assert processed == ["AAPL", "IBM"], "IBM must be attempted even after AAPL fails"
    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert any("AAPL" in r.message for r in errors)
    assert not any("IBM" in r.message for r in errors)


def test_run_cycle_logs_fuse_exception_via_propagation(caplog: pytest.LogCaptureFixture) -> None:
    """An exception raised inside fuse_signals propagates to run_cycle's handler."""
    with (
        patch("main.DataStore"),
        patch("main._load_thresholds", return_value={}),
        patch("main.build_client"),
        patch("main._load_reference_bars", return_value=pd.DataFrame()),
        patch("main.run_symbol_cycle", side_effect=RuntimeError("fuse boom")),
    ):
        with caplog.at_level(logging.ERROR, logger="stock-assistant"):
            run_cycle(_make_settings(), symbols=["IBM"])

    errors = [r for r in caplog.records if r.levelname == "ERROR"]
    assert errors, "At least one ERROR log expected"
    assert any("cycle failed" in r.message for r in errors)
    # exc_info must be attached so the stack trace appears in logs
    assert any(r.exc_info is not None for r in errors)
