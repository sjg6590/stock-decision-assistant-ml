from __future__ import annotations

import hashlib
import logging
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# After a daily rate-limit notice, skip further Alpha Vantage calls for this process.
_alphavantage_rate_limited = False


def reset_alphavantage_rate_limit_state() -> None:
    """Reset session state (for tests)."""
    global _alphavantage_rate_limited
    _alphavantage_rate_limited = False


def _is_rate_limit_notice(notice: str) -> bool:
    text = notice.lower()
    return (
        "rate limit" in text
        or "requests per day" in text
        or "thank you for using alpha vantage" in text
        or "spreading out your api requests" in text
    )


def _hash_article(symbol: str, title: str, url: str) -> str:
    return hashlib.sha256(f"{symbol}|{title}|{url}".encode("utf-8")).hexdigest()


def fetch_alpha_vantage_news_sentiment(symbol: str, api_key: str, limit: int = 50) -> list[dict[str, Any]]:
    global _alphavantage_rate_limited

    if not api_key:
        return []
    if _alphavantage_rate_limited:
        return []

    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": symbol,
        "sort": "LATEST",
        "limit": limit,
        "apikey": api_key,
    }
    with httpx.Client(timeout=20) as client:
        resp = client.get("https://www.alphavantage.co/query", params=params)
        resp.raise_for_status()
    payload = resp.json()
    notice = payload.get("Note") or payload.get("Information")
    if notice:
        if _is_rate_limit_notice(str(notice)):
            _alphavantage_rate_limited = True
            logger.warning(
                "Alpha Vantage rate limited for symbol=%s; skipping Alpha Vantage for the rest of this run",
                symbol,
            )
        else:
            logger.warning("Alpha Vantage news skipped symbol=%s: %s", symbol, notice)
        return []
    feed = payload.get("feed", [])
    out: list[dict[str, Any]] = []
    for item in feed:
        title = item.get("title", "")
        url = item.get("url", "")
        out.append(
            {
                "article_hash": _hash_article(symbol, title, url),
                "symbol": symbol,
                "source": "alphavantage",
                "title": title,
                "url": url,
                "summary": item.get("summary", ""),
                "overall_sentiment_score": float(item.get("overall_sentiment_score", 0.0)),
                "overall_sentiment_label": item.get("overall_sentiment_label", "Neutral"),
                "ticker_sentiment": item.get("ticker_sentiment", []),
                "published_at": item.get("time_published", datetime.now(tz=timezone.utc).isoformat()),
            }
        )
    return out
