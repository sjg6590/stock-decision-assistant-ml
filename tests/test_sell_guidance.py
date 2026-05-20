from __future__ import annotations

from datetime import date, datetime, timezone

from schwab_client.market_data import parse_quote_price
from sda_types import HorizonPrediction, ModelPrediction
from signals.sell_guidance import build_sell_guidance


def _sample_pred(expected_return: float = 0.03) -> ModelPrediction:
    return ModelPrediction(
        symbol="GOOGL",
        generated_at=datetime.now(tz=timezone.utc),
        predictions=[HorizonPrediction("3d", 3, 0.7, expected_return)],
        best_horizon="3d",
        ml_confidence=0.7,
    )


def test_sell_guidance_close_only_when_no_live_quote() -> None:
    guide = build_sell_guidance(
        _sample_pred(),
        latest_close=100.0,
        last_bar_date=date(2026, 5, 19),
        live_price=None,
    )
    assert "Reference close $100.00 (2026-05-19)" in guide
    assert "live quote unavailable" in guide
    assert "Take-profit (actionable, from last close): $102.40-$103.60" in guide
    assert "Stop (from last close): $97.00" in guide


def test_sell_guidance_live_anchored_targets() -> None:
    guide = build_sell_guidance(
        _sample_pred(),
        latest_close=100.0,
        last_bar_date=date(2026, 5, 19),
        live_price=105.0,
    )
    assert "live $105.00" in guide
    assert "Take-profit (actionable, from live): $107.52-$108.78" in guide
    assert "Stop (from live): $101.85" in guide
    assert "Take-profit vs last close (historical): $102.40-$103.60" in guide
    assert "Stop vs last close: $97.00" in guide
    assert "Preferred exit horizon: 3d (modeled upside +3.00%)" in guide


def test_sell_guidance_uses_largest_positive_expected_return() -> None:
    pred = ModelPrediction(
        symbol="AAPL",
        generated_at=datetime.now(tz=timezone.utc),
        predictions=[
            HorizonPrediction("3w", 15, 0.75, 0.0024, threshold=0.55),
            HorizonPrediction("1mo", 22, 0.65, 0.025, threshold=0.51),
        ],
        best_horizon="3w",
        ml_confidence=0.70,
    )
    guide = build_sell_guidance(
        pred,
        latest_close=302.25,
        last_bar_date=date(2026, 5, 20),
        live_price=302.00,
        ml_threshold=0.60,
    )
    assert "Exit timing horizon: 3w (strongest ML conviction)" in guide
    assert "Take-profit uses 1mo modeled upside (+2.50%)" in guide
    assert "Take-profit (actionable, from live): $308.04-$311.06" in guide
    assert "very small move" not in guide


def test_sell_guidance_splits_timing_and_return_horizons() -> None:
    pred = ModelPrediction(
        symbol="AAPL",
        generated_at=datetime.now(tz=timezone.utc),
        predictions=[
            HorizonPrediction("3w", 15, 0.80, 0.002, threshold=0.55),
            HorizonPrediction("1mo", 22, 0.62, 0.03, threshold=0.51),
        ],
        best_horizon="3w",
        ml_confidence=0.70,
    )
    guide = build_sell_guidance(
        pred,
        latest_close=100.0,
        live_price=100.0,
        ml_threshold=0.60,
    )
    assert "Exit timing horizon: 3w" in guide
    assert "Take-profit uses 1mo modeled upside (+3.00%)" in guide


def test_sell_guidance_priced_in_note_when_live_above_close_band() -> None:
    guide = build_sell_guidance(
        _sample_pred(),
        latest_close=100.0,
        last_bar_date=date(2026, 5, 19),
        live_price=104.0,
    )
    assert "already be priced in" in guide
    assert "historical context" in guide


def test_sell_guidance_partial_priced_in_note() -> None:
    guide = build_sell_guidance(
        _sample_pred(),
        latest_close=100.0,
        last_bar_date=date(2026, 5, 19),
        live_price=102.50,
    )
    assert "inside the take-profit band vs last close" in guide
    assert "prefer live-anchored targets" in guide.lower() or "Prefer live-anchored" in guide


def test_parse_quote_price_mark_and_last() -> None:
    payload = {
        "AAPL": {
            "quote": {
                "mark": 150.25,
                "lastPrice": 150.20,
            }
        }
    }
    assert parse_quote_price("AAPL", payload) == 150.25

    payload_last_only = {"AAPL": {"quote": {"lastPrice": 149.5}}}
    assert parse_quote_price("aapl", payload_last_only) == 149.5
