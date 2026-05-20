from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

import httpx

from data.news_utils import article_summary
from sda_types import SentimentResult
from sentiment.aggregate import aggregate_articles
from sentiment.llm_provider import analyze_with_llm
from settings import Settings

logger = logging.getLogger(__name__)


def _window_key(hours: int = 48) -> str:
    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(hours=hours)
    return f"{start.isoformat()}::{end.isoformat()}"


def _article_limit(settings: Settings) -> int:
    provider = settings.llm_provider.lower().strip()
    if provider == "ollama":
        return min(settings.llm_article_limit, 12)
    if provider == "anthropic":
        return min(settings.llm_article_limit, 8)
    return settings.llm_article_limit


def _blurb_char_limit(settings: Settings) -> int:
    if settings.llm_provider.lower().strip() == "anthropic":
        return 100
    return 200


def _neutral_sentiment(symbol: str, reason: str) -> SentimentResult:
    return SentimentResult(
        symbol=symbol,
        sentiment_label="neutral",
        confidence=0.0,
        bullish_factors=[],
        bearish_factors=[reason],
        macro_risks=[],
        summary=f"Sentiment unavailable: {reason}",
        recommended_action="neutral",
    )


def analyze_symbol_sentiment(
    settings: Settings,
    symbol: str,
    articles: list[dict[str, Any]],
    model_hint: str,
) -> tuple[str, SentimentResult]:
    selected = aggregate_articles(articles, limit=_article_limit(settings))
    logger.debug(
        "sentiment_articles symbol=%s total=%d selected=%d provider=%s",
        symbol, len(articles), len(selected), settings.llm_provider,
    )
    article_lines = []
    for a in selected:
        blurb = article_summary(a)
        line = (
            f"- [{a.get('published_at', 'unknown date')}] {a.get('title', '').strip()} "
            f"(source: {a.get('source', 'unknown')})"
        )
        if blurb:
            line += f" — {blurb[:_blurb_char_limit(settings)]}"
        article_lines.append(line)
    prompt = (
        f"Ticker: {symbol}\n"
        f"ML signal context: {model_hint}\n"
        "The ML signal context above describes the horizon and confidence of a quantitative bullish prediction.\n"
        "Your task: determine whether the news sentiment supports ('agree'), contradicts ('disagree'),\n"
        "or is ambiguous about ('neutral') this ML bullish signal.\n\n"
        "When market prices are provided below, treat them as the authoritative current reference.\n"
        "Do not use stale prices from older articles if they conflict with the live quote.\n"
        "Focus on near-term (days to weeks) price-relevant information.\n"
        "Weight articles published most recently more heavily.\n"
        "Treat earnings, guidance, analyst upgrades/downgrades, regulatory events, and M&A as high-signal.\n"
        "Ignore articles that are not specific to this ticker's near-term outlook.\n\n"
        f"News articles ({len(article_lines)} total, newest first):\n"
        + "\n".join(article_lines)
    )
    logger.debug("llm_prompt symbol=%s prompt_chars=%d articles=%d", symbol, len(prompt), len(article_lines))
    try:
        payload = analyze_with_llm(settings, prompt)
    except (httpx.TimeoutException, httpx.HTTPError) as exc:
        logger.warning("LLM sentiment failed for symbol=%s: %s", symbol, exc)
        return _window_key(), _neutral_sentiment(symbol, str(exc))
    except Exception as exc:
        logger.warning("LLM sentiment failed for symbol=%s: %s", symbol, exc)
        return _window_key(), _neutral_sentiment(symbol, str(exc))
    result = SentimentResult(
        symbol=symbol,
        sentiment_label=str(payload.get("sentiment_label", "neutral")).lower(),
        confidence=float(payload.get("confidence", 0.0)),
        bullish_factors=list(payload.get("bullish_factors", [])),
        bearish_factors=list(payload.get("bearish_factors", [])),
        macro_risks=list(payload.get("macro_risks", [])),
        summary=str(payload.get("summary", "")),
        recommended_action=str(payload.get("recommended_action", "neutral")).lower(),
    )
    logger.debug(
        "sentiment_result symbol=%s label=%s confidence=%.3f action=%s bullish=%s bearish=%s",
        symbol,
        result.sentiment_label,
        result.confidence,
        result.recommended_action,
        result.bullish_factors,
        result.bearish_factors,
    )
    return _window_key(), result


def sentiment_to_json(result: SentimentResult) -> str:
    return json.dumps(result.__dict__, indent=2)
