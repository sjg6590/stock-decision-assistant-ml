from __future__ import annotations

import hashlib
import re
from datetime import datetime, timezone
from typing import Any

import yfinance as yf

_HTML_TAG_RE = re.compile(r"<[^>]+>")


def _hash_article(symbol: str, title: str, url: str) -> str:
    return hashlib.sha256(f"{symbol}|{title}|{url}".encode("utf-8")).hexdigest()


def _strip_html(text: str) -> str:
    return _HTML_TAG_RE.sub("", text).strip()


def normalize_yfinance_item(item: dict[str, Any]) -> dict[str, str] | None:
    """Map a yfinance news item (legacy or 2024+ nested content schema) to flat fields."""
    content = item.get("content") if isinstance(item.get("content"), dict) else {}

    title = str(content.get("title") or item.get("title") or "").strip()
    if not title:
        return None

    url = ""
    for key in ("canonicalUrl", "clickThroughUrl"):
        nested = content.get(key)
        if isinstance(nested, dict) and nested.get("url"):
            url = str(nested["url"])
            break
    if not url:
        url = str(item.get("link") or "")

    publisher = ""
    provider = content.get("provider")
    if isinstance(provider, dict):
        publisher = str(provider.get("displayName") or "")
    if not publisher:
        publisher = str(item.get("publisher") or "")

    published = ""
    pub_date = content.get("pubDate")
    if pub_date:
        published = str(pub_date)
    else:
        provider_publish_time = item.get("providerPublishTime")
        if provider_publish_time:
            published = datetime.fromtimestamp(
                int(provider_publish_time), tz=timezone.utc
            ).isoformat()

    summary = str(content.get("summary") or item.get("summary") or "").strip()
    if not summary:
        description = str(content.get("description") or item.get("description") or "")
        summary = _strip_html(description)

    return {
        "title": title,
        "url": url,
        "publisher": publisher,
        "published_at": published,
        "summary": summary,
    }


def fetch_yfinance_news(symbol: str, limit: int = 20) -> list[dict[str, Any]]:
    ticker = yf.Ticker(symbol)
    items = getattr(ticker, "news", [])[:limit]
    out: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        parsed = normalize_yfinance_item(item)
        if not parsed:
            continue
        title = parsed["title"]
        link = parsed["url"]
        published = parsed["published_at"] or datetime.now(tz=timezone.utc).isoformat()
        article: dict[str, Any] = {
            "article_hash": _hash_article(symbol, title, link),
            "symbol": symbol,
            "source": "yfinance",
            "title": title,
            "url": link,
            "publisher": parsed["publisher"],
            "published_at": published,
        }
        if parsed["summary"]:
            article["summary"] = parsed["summary"]
        out.append(article)
    return out
