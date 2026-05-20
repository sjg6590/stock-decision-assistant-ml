from __future__ import annotations

from datetime import datetime, timezone

import pytest

from sda_types import HorizonPrediction, ModelPrediction, SentimentResult
from signals.fuse import fuse_signals


def test_buy_signal_requires_ml_and_sentiment_agreement() -> None:
    pred = ModelPrediction(
        symbol="AAPL",
        generated_at=datetime.now(tz=timezone.utc),
        predictions=[
            HorizonPrediction("1d", 1, 0.7, 0.02),
            HorizonPrediction("3d", 3, 0.72, 0.03),
        ],
        best_horizon="3d",
        ml_confidence=0.71,
    )
    sentiment = SentimentResult(
        symbol="AAPL",
        sentiment_label="bullish",
        confidence=0.8,
        bullish_factors=["Strong demand"],
        bearish_factors=[],
        macro_risks=[],
        summary="Positive setup",
        recommended_action="align",
    )
    signal = fuse_signals("AAPL", pred, sentiment, ml_threshold=0.6, sentiment_threshold=0.55, latest_close=100.0)
    assert signal.should_notify is True
    assert signal.signal_type == "BUY_CANDIDATE"


def _bullish_sentiment() -> SentimentResult:
    return SentimentResult(
        symbol="AAPL",
        sentiment_label="bullish",
        confidence=0.8,
        bullish_factors=["Strong demand"],
        bearish_factors=[],
        macro_risks=[],
        summary="Positive setup",
        recommended_action="align",
    )


def test_per_horizon_threshold_used_when_set() -> None:
    """Horizons with a saved threshold must be compared against it, not the global."""
    pred = ModelPrediction(
        symbol="AAPL",
        generated_at=datetime.now(tz=timezone.utc),
        # prob_up=0.65, saved threshold=0.70 → NOT bullish
        predictions=[
            HorizonPrediction("1d", 1, 0.65, 0.02, threshold=0.70),
            HorizonPrediction("3d", 3, 0.65, 0.03, threshold=0.70),
        ],
        best_horizon="3d",
        ml_confidence=0.65,
    )
    signal = fuse_signals("AAPL", pred, _bullish_sentiment(), ml_threshold=0.60, sentiment_threshold=0.55, latest_close=100.0)
    # Both horizons fail their saved threshold, so no buy even though global=0.60 would pass
    assert signal.should_notify is False
    assert signal.signal_type == "NO_ALERT"


def test_global_threshold_used_when_horizon_threshold_is_zero() -> None:
    """Horizons with threshold==0.0 (old artifacts) must fall back to the global."""
    pred = ModelPrediction(
        symbol="AAPL",
        generated_at=datetime.now(tz=timezone.utc),
        # prob_up=0.65, threshold=0.0 → should use global 0.60 → bullish
        predictions=[
            HorizonPrediction("1d", 1, 0.65, 0.02, threshold=0.0),
            HorizonPrediction("3d", 3, 0.65, 0.03, threshold=0.0),
        ],
        best_horizon="3d",
        ml_confidence=0.65,
    )
    signal = fuse_signals("AAPL", pred, _bullish_sentiment(), ml_threshold=0.60, sentiment_threshold=0.55, latest_close=100.0)
    assert signal.should_notify is True
    assert signal.signal_type == "BUY_CANDIDATE"


def test_mixed_thresholds_only_passing_horizons_count() -> None:
    """One horizon passes its saved threshold, one fails — only one bullish horizon → no buy."""
    pred = ModelPrediction(
        symbol="AAPL",
        generated_at=datetime.now(tz=timezone.utc),
        predictions=[
            HorizonPrediction("1d", 1, 0.75, 0.02, threshold=0.70),  # passes saved
            HorizonPrediction("3d", 3, 0.65, 0.03, threshold=0.70),  # fails saved
        ],
        best_horizon="1d",
        ml_confidence=0.70,
    )
    signal = fuse_signals("AAPL", pred, _bullish_sentiment(), ml_threshold=0.60, sentiment_threshold=0.55, latest_close=100.0)
    # Only 1 bullish horizon; fuse_signals requires >= 2
    assert signal.should_notify is False


def test_cold_start_tag_lowers_fused_confidence() -> None:
    pred = ModelPrediction(
        symbol="AAPL",
        generated_at=datetime.now(tz=timezone.utc),
        predictions=[
            HorizonPrediction("1d", 1, 0.7, 0.02),
            HorizonPrediction("3d", 3, 0.72, 0.03),
        ],
        best_horizon="3d",
        ml_confidence=0.71,
    )
    sentiment = SentimentResult(
        symbol="AAPL",
        sentiment_label="bullish",
        confidence=0.8,
        bullish_factors=["Strong demand"],
        bearish_factors=[],
        macro_risks=[],
        summary="Positive setup",
        recommended_action="align",
    )
    baseline = fuse_signals(
        "AAPL", pred, sentiment, ml_threshold=0.6, sentiment_threshold=0.55, latest_close=100.0
    )
    discounted = fuse_signals(
        "AAPL",
        pred,
        sentiment,
        ml_threshold=0.6,
        sentiment_threshold=0.55,
        latest_close=100.0,
        model_tag="cold_start",
        cold_start_confidence_discount=0.1,
    )
    assert discounted.confidence == pytest.approx(baseline.confidence - 0.1, abs=1e-6)
    assert discounted.confidence < baseline.confidence


def test_filtered_signal_is_not_buy_candidate() -> None:
    pred = ModelPrediction(
        symbol="AAPL",
        generated_at=datetime.now(tz=timezone.utc),
        predictions=[
            HorizonPrediction("1d", 1, 0.7, 0.02),
            HorizonPrediction("3d", 3, 0.72, 0.03),
        ],
        best_horizon="3d",
        ml_confidence=0.71,
    )
    sentiment = SentimentResult(
        symbol="AAPL",
        sentiment_label="bearish",
        confidence=0.8,
        bullish_factors=[],
        bearish_factors=["Weak demand"],
        macro_risks=[],
        summary="Negative setup",
        recommended_action="align",
    )
    signal = fuse_signals("AAPL", pred, sentiment, ml_threshold=0.6, sentiment_threshold=0.55, latest_close=100.0)
    assert signal.should_notify is False
    assert signal.signal_type == "NO_ALERT"
