from __future__ import annotations

from datetime import date, datetime, timezone

from notify.base import format_alert
from sda_types import FusedSignal, HorizonPrediction, ModelPrediction, SentimentResult
from signals.sell_guidance import build_sell_guidance


def test_sell_guidance_does_not_repeat_bearish_factors() -> None:
    sentiment = SentimentResult(
        symbol="GOOGL",
        sentiment_label="bullish",
        confidence=0.72,
        bullish_factors=["Institutional buying"],
        bearish_factors=["Sector weakness"],
        macro_risks=["Fed uncertainty"],
        summary="Mixed",
        recommended_action="agree",
    )
    pred = ModelPrediction(
        symbol="GOOGL",
        generated_at=datetime.now(tz=timezone.utc),
        predictions=[HorizonPrediction("3d", 3, 0.7, 0.03)],
        best_horizon="3d",
        ml_confidence=0.7,
    )
    guide = build_sell_guidance(
        pred,
        latest_close=100.0,
        last_bar_date=date(2026, 5, 19),
        live_price=101.0,
    )
    assert "Bearish factors" not in guide
    assert "Sector weakness" not in guide

    fused = FusedSignal(
        symbol="GOOGL",
        signal_type="BUY_CANDIDATE",
        confidence=0.66,
        should_notify=True,
        reason="test",
        sell_guide=guide,
    )
    msg = format_alert("GOOGL", fused, sentiment)
    assert msg.body.count("Sector weakness") == 1
    assert "Bearish factors:" in msg.body
    assert "Macro risks:" in msg.body
    assert "Fed uncertainty" in msg.body
