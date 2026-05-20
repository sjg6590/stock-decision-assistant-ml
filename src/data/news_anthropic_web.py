from __future__ import annotations

import hashlib
import logging
import time
from datetime import datetime, timezone
from typing import Any

from anthropic import Anthropic, RateLimitError

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
# Haiku and other models require direct-only tool invocation for server tools like web_search.
_WEB_SEARCH_TOOL = {
    "type": "web_search_20260209",
    "name": "web_search",
    "allowed_callers": ["direct"],
}

# After HTTP 429, skip further Anthropic web-search calls for this process.
_anthropic_rate_limited = False
_last_anthropic_call_at: float | None = None


def reset_anthropic_rate_limit_state() -> None:
    """Reset session state (for tests)."""
    global _anthropic_rate_limited, _last_anthropic_call_at
    _anthropic_rate_limited = False
    _last_anthropic_call_at = None


def is_anthropic_rate_limited() -> bool:
    return _anthropic_rate_limited


def mark_anthropic_rate_limited(symbol: str, *, context: str) -> None:
    global _anthropic_rate_limited
    if not _anthropic_rate_limited:
        _anthropic_rate_limited = True
        logger.warning(
            "Anthropic rate limited during %s for symbol=%s; skipping Anthropic web search for the rest of this run",
            context,
            symbol,
        )


def maybe_anthropic_delay(delay_seconds: float) -> None:
    """Space out Anthropic calls when processing a large watchlist."""
    global _last_anthropic_call_at
    if delay_seconds <= 0:
        return
    effective = delay_seconds * 2 if _anthropic_rate_limited else delay_seconds
    if _last_anthropic_call_at is not None:
        elapsed = time.monotonic() - _last_anthropic_call_at
        if elapsed < effective:
            time.sleep(effective - elapsed)


def rate_limit_retry_seconds(exc: BaseException, *, default: float = 60.0) -> float:
    """Seconds to wait before retrying after HTTP 429 (uses Retry-After when present)."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None) if response is not None else None
    if headers is not None:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
        if retry_after is not None:
            try:
                return max(float(retry_after), 1.0)
            except (TypeError, ValueError):
                pass
    return default


def note_anthropic_call() -> None:
    global _last_anthropic_call_at
    _last_anthropic_call_at = time.monotonic()


def create_anthropic_client(api_key: str) -> Anthropic:
    # Fail fast on 429; the SDK default retries can block for 25s+ per symbol.
    return Anthropic(api_key=api_key, max_retries=0)


def _hash_article(symbol: str, title: str, url: str) -> str:
    return hashlib.sha256(f"{symbol}|{title}|{url}".encode("utf-8")).hexdigest()


def _article_from_result(symbol: str, title: str, url: str, summary: str = "") -> dict[str, Any]:
    published = datetime.now(tz=timezone.utc).isoformat()
    return {
        "article_hash": _hash_article(symbol, title, url),
        "symbol": symbol,
        "source": "anthropic_web",
        "title": title.strip(),
        "url": url.strip(),
        "publisher": "",
        "published_at": published,
        "summary": summary.strip(),
    }


def extract_articles_from_response(symbol: str, response: Any, *, limit: int = 15) -> list[dict[str, Any]]:
    """Turn Anthropic web-search message content into normalized article dicts."""
    articles: list[dict[str, Any]] = []
    seen_urls: set[str] = set()

    def add(title: str, url: str, summary: str = "") -> None:
        if not title or not url or url in seen_urls:
            return
        seen_urls.add(url)
        articles.append(_article_from_result(symbol, title, url, summary))

    for block in getattr(response, "content", []) or []:
        block_type = getattr(block, "type", None)
        if block_type == "web_search_tool_result":
            content = getattr(block, "content", None)
            if isinstance(content, list):
                for item in content:
                    if getattr(item, "type", None) == "web_search_result":
                        add(
                            str(getattr(item, "title", "") or ""),
                            str(getattr(item, "url", "") or ""),
                        )
        elif block_type == "text":
            from sentiment.llm_provider import _parse_llm_json

            text = getattr(block, "text", "") or ""
            parsed = _parse_llm_json(text)
            if isinstance(parsed, list):
                for row in parsed:
                    if isinstance(row, dict):
                        add(
                            str(row.get("title", "") or ""),
                            str(row.get("url", "") or ""),
                            str(row.get("summary", "") or row.get("description", "") or ""),
                        )
            elif isinstance(parsed, dict):
                rows = parsed.get("articles")
                if isinstance(rows, list):
                    for row in rows:
                        if isinstance(row, dict):
                            add(
                                str(row.get("title", "") or ""),
                                str(row.get("url", "") or ""),
                                str(row.get("summary", "") or row.get("description", "") or ""),
                            )

    return articles[:limit]


def anthropic_web_search_tool(*, max_uses: int) -> dict[str, Any]:
    return {**_WEB_SEARCH_TOOL, "max_uses": max_uses}


def is_web_search_tool_error(exc: BaseException) -> bool:
    """True when the API rejected web_search for this model or tool config."""
    message = str(exc).lower()
    return "web_search" in message and (
        "allowed_callers" in message
        or "programmatic tool calling" in message
        or "does not support" in message
    )


def is_rate_limit_error(exc: BaseException) -> bool:
    if isinstance(exc, RateLimitError):
        return True
    message = str(exc).lower()
    return "429" in message or "rate limit" in message


def fetch_anthropic_web_news(
    symbol: str,
    api_key: str,
    *,
    model: str = _DEFAULT_MODEL,
    max_uses: int = 3,
    limit: int = 15,
    delay_seconds: float = 0.0,
) -> list[dict[str, Any]]:
    """Use Anthropic's web search tool to find recent headlines for a ticker."""
    if not api_key or is_anthropic_rate_limited():
        return []

    prompt = (
        f"Search the web for recent financial news about {symbol} stock (ticker {symbol}).\n"
        "Focus on the last 48–72 hours: earnings, guidance, analyst actions, regulatory events, "
        "and other near-term price drivers.\n"
        f"Return ONLY a JSON array (no markdown) of up to {limit} objects:\n"
        '[{"title": "...", "url": "https://...", "summary": "one sentence", "published_at": "ISO-8601 UTC optional"}]\n'
        "Include only articles clearly about this ticker."
    )

    maybe_anthropic_delay(delay_seconds)
    client = create_anthropic_client(api_key)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=2048,
            temperature=0.2,
            messages=[{"role": "user", "content": prompt}],
            tools=[anthropic_web_search_tool(max_uses=max_uses)],
        )
    except Exception as exc:
        if is_rate_limit_error(exc):
            mark_anthropic_rate_limited(symbol, context="web news ingest")
        else:
            logger.warning("Anthropic web news skipped symbol=%s: %s", symbol, exc)
        return []
    finally:
        note_anthropic_call()

    articles = extract_articles_from_response(symbol, response, limit=limit)
    logger.info("anthropic web news symbol=%s count=%d", symbol, len(articles))
    return articles
