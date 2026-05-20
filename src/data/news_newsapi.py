from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# After a 429, skip further NewsAPI calls for this process to avoid hammering the API.
_newsapi_rate_limited = False
_last_newsapi_request_at: float | None = None


def reset_newsapi_rate_limit_state() -> None:
    """Reset session state (for tests)."""
    global _newsapi_rate_limited, _last_newsapi_request_at
    _newsapi_rate_limited = False
    _last_newsapi_request_at = None


def _hash_article(symbol: str, title: str, url: str) -> str:
    base = f"{symbol}|{title}|{url}".encode("utf-8")
    return hashlib.sha256(base).hexdigest()


def fetch_newsapi(
    symbol: str,
    api_key: str,
    limit: int = 20,
    *,
    inter_request_delay_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    global _newsapi_rate_limited, _last_newsapi_request_at

    if not api_key:
        return []
    if _newsapi_rate_limited:
        return []

    if inter_request_delay_seconds > 0 and _last_newsapi_request_at is not None:
        elapsed = time.monotonic() - _last_newsapi_request_at
        if elapsed < inter_request_delay_seconds:
            time.sleep(inter_request_delay_seconds - elapsed)

    params = {
        "q": symbol,
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": limit,
        "apiKey": api_key,
    }
    with httpx.Client(timeout=15) as client:
        resp = client.get("https://newsapi.org/v2/everything", params=params)
        _last_newsapi_request_at = time.monotonic()

    if resp.status_code == 429:
        _newsapi_rate_limited = True
        logger.warning(
            "NewsAPI rate limited (HTTP 429) for symbol=%s; skipping NewsAPI for the rest of this run",
            symbol,
        )
        return []

    if resp.status_code >= 400:
        logger.warning("NewsAPI request failed symbol=%s status=%s", symbol, resp.status_code)
        return []

    payload = resp.json()
    articles = payload.get("articles", [])
    result = []
    for item in articles:
        title = item.get("title", "")
        url = item.get("url", "")
        published = item.get("publishedAt") or datetime.now(tz=timezone.utc).isoformat()
        description = item.get("description", "") or ""
        result.append(
            {
                "article_hash": _hash_article(symbol, title, url),
                "symbol": symbol,
                "source": "newsapi",
                "title": title,
                "url": url,
                "description": description,
                "summary": description,
                "published_at": published,
            }
        )
    return result
