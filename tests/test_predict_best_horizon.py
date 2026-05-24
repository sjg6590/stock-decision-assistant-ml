"""Tests for the margin-above-threshold best-horizon selection."""
from __future__ import annotations

import pytest

from ml.predict import _horizon_confidence_margin
from sda_types import HorizonPrediction


def _hp(horizon: str, prob_up: float, threshold: float, exp_ret: float = 0.01) -> HorizonPrediction:
    return HorizonPrediction(
        horizon=horizon, bars=1, probability_up=prob_up, expected_return=exp_ret, threshold=threshold
    )


def test_selects_largest_margin_above_threshold():
    preds = [
        _hp("1d", prob_up=0.70, threshold=0.55),   # margin = 0.15
        _hp("5d", prob_up=0.80, threshold=0.70),   # margin = 0.10
        _hp("1mo", prob_up=0.60, threshold=0.45),  # margin = 0.15 (tie)
    ]
    best = max(preds, key=_horizon_confidence_margin)
    # "1d" and "1mo" tie on margin (0.15); tiebreaker is expected_return (both 0.01 here → first in sort wins)
    assert best.horizon in ("1d", "1mo")


def test_tiebreaker_by_expected_return():
    preds = [
        _hp("1d", prob_up=0.70, threshold=0.55, exp_ret=0.005),   # margin=0.15, ret=0.005
        _hp("5d", prob_up=0.70, threshold=0.55, exp_ret=0.030),   # margin=0.15, ret=0.030 → winner
    ]
    best = max(preds, key=_horizon_confidence_margin)
    assert best.horizon == "5d"


def test_negative_expected_return_clamped_to_zero_for_tiebreak():
    preds = [
        _hp("1d", prob_up=0.70, threshold=0.55, exp_ret=-0.05),  # margin=0.15, clamped_ret=0.0
        _hp("5d", prob_up=0.70, threshold=0.55, exp_ret=0.001),  # margin=0.15, clamped_ret=0.001 → wins
    ]
    best = max(preds, key=_horizon_confidence_margin)
    assert best.horizon == "5d"


def test_all_horizons_below_threshold():
    preds = [
        _hp("1d", prob_up=0.40, threshold=0.55),   # margin = -0.15
        _hp("5d", prob_up=0.45, threshold=0.55),   # margin = -0.10 → least negative → selected
        _hp("1mo", prob_up=0.35, threshold=0.55),  # margin = -0.20
    ]
    best = max(preds, key=_horizon_confidence_margin)
    assert best.horizon == "5d"


def test_single_horizon():
    preds = [_hp("1d", prob_up=0.6, threshold=0.5)]
    best = max(preds, key=_horizon_confidence_margin)
    assert best.horizon == "1d"


def test_threshold_zero_uses_half_as_baseline():
    # When threshold=0, the helper should fall back to 0.5
    p = _hp("1d", prob_up=0.7, threshold=0.0)
    margin, _ = _horizon_confidence_margin(p)
    assert abs(margin - 0.2) < 1e-9  # 0.7 - 0.5 = 0.2
