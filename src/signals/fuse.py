from __future__ import annotations

import logging
from datetime import date

from sda_types import FusedSignal, ModelPrediction, SentimentResult
from signals.sell_guidance import build_sell_guidance

logger = logging.getLogger(__name__)


def fuse_signals(
    symbol: str,
    prediction: ModelPrediction,
    sentiment: SentimentResult,
    ml_threshold: float,
    sentiment_threshold: float,
    latest_close: float,
    last_bar_date: date | None = None,
    live_price: float | None = None,
    model_tag: str | None = None,
    cold_start_confidence_discount: float = 0.0,
) -> FusedSignal:
    # Use per-horizon threshold from training when available; fall back to global ml_threshold
    bullish_horizons = [
        p for p in prediction.predictions
        if p.probability_up >= (p.threshold if p.threshold > 0 else ml_threshold)
    ]
    has_ml_buy = len(bullish_horizons) >= 2

    logger.debug(
        "fusion_ml symbol=%s bullish_horizons=%d/%d (global_threshold=%.2f) has_ml_buy=%s horizons=%s",
        symbol,
        len(bullish_horizons),
        len(prediction.predictions),
        ml_threshold,
        has_ml_buy,
        [p.horizon for p in bullish_horizons],
    )

    sentiment_label_ok = sentiment.sentiment_label in {"bullish", "neutral"}
    sentiment_conf_ok = sentiment.confidence >= sentiment_threshold
    sentiment_action_ok = sentiment.recommended_action != "disagree"
    sentiment_ok = sentiment_conf_ok and sentiment_label_ok and sentiment_action_ok

    logger.debug(
        "fusion_sentiment symbol=%s label=%s (label_ok=%s) confidence=%.3f (conf_ok=%s) action=%s (action_ok=%s) sentiment_ok=%s",
        symbol,
        sentiment.sentiment_label,
        sentiment_label_ok,
        sentiment.confidence,
        sentiment_conf_ok,
        sentiment.recommended_action,
        sentiment_action_ok,
        sentiment_ok,
    )

    should_notify = bool(has_ml_buy and sentiment_ok)
    signal_type = "BUY_CANDIDATE" if should_notify else "NO_ALERT"
    fused_conf = min(1.0, (prediction.ml_confidence + sentiment.confidence) / 2)
    if model_tag == "cold_start" and cold_start_confidence_discount > 0:
        fused_conf = max(0.0, fused_conf - cold_start_confidence_discount)
        logger.debug(
            "fusion_cold_start_discount symbol=%s discount=%.2f fused_confidence=%.3f",
            symbol, cold_start_confidence_discount, fused_conf,
        )

    logger.debug(
        "fusion_result symbol=%s signal=%s should_notify=%s fused_confidence=%.3f model_tag=%s",
        symbol, signal_type, should_notify, fused_conf, model_tag,
    )

    reason = (
        f"ML bullish horizons={len(bullish_horizons)}, "
        f"sentiment={sentiment.sentiment_label} ({sentiment.confidence:.2f}), "
        f"recommended_action={sentiment.recommended_action}"
    )
    return FusedSignal(
        symbol=symbol,
        signal_type=signal_type,
        confidence=fused_conf,
        should_notify=should_notify,
        reason=reason,
        sell_guide=build_sell_guidance(
            prediction,
            latest_close=latest_close,
            last_bar_date=last_bar_date,
            live_price=live_price,
            ml_threshold=ml_threshold,
        ),
    )
