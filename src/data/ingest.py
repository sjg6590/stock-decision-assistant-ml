from __future__ import annotations

import logging
from typing import Any

import pandas as pd
import yfinance as yf

from data.news_alphavantage import fetch_alpha_vantage_news_sentiment
from data.news_anthropic_web import fetch_anthropic_web_news
from data.news_newsapi import fetch_newsapi
from data.news_utils import normalize_article
from data.news_yfinance import fetch_yfinance_news
from data.store import DataStore
from schwab_client.market_data import fetch_daily_history

logger = logging.getLogger(__name__)

# Merge SQLite cache when a live fetch returns fewer than this many articles.
_MIN_ARTICLES_BEFORE_CACHE = 5


def backfill_from_yfinance(symbol: str, years: int = 5) -> pd.DataFrame:
    period = f"{years}y" if years > 0 else "max"
    hist = yf.download(symbol, period=period, interval="1d", progress=False, auto_adjust=False)
    if hist.empty:
        return pd.DataFrame()
    hist = hist.reset_index().rename(
        columns={
            "Date": "datetime",
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
    )
    hist["symbol"] = symbol
    return hist[["datetime", "open", "high", "low", "close", "volume", "symbol"]]


def ingest_symbol_bars(store: DataStore, schwab_client: Any, symbol: str, years: int = 5) -> pd.DataFrame:
    bars = fetch_daily_history(schwab_client, symbol, years=years)
    if bars.empty:
        bars = backfill_from_yfinance(symbol, years=years)
    if bars.empty:
        return bars
    bars["datetime"] = pd.to_datetime(bars["datetime"], utc=True).dt.tz_localize(None)
    store.save_bars(symbol, bars)
    return bars


def _count_by_source(articles: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for article in articles:
        source = str(article.get("source", "unknown"))
        counts[source] = counts.get(source, 0) + 1
    return counts


def ingest_news_for_symbol(
    store: DataStore,
    symbol: str,
    newsapi_key: str,
    alphavantage_key: str,
    *,
    newsapi_delay_seconds: float = 0.0,
    anthropic_api_key: str = "",
    anthropic_web_search_enabled: bool = False,
    anthropic_model: str = "claude-haiku-4-5-20251001",
    anthropic_web_search_max_uses: int = 3,
    anthropic_delay_seconds: float = 0.0,
) -> list[dict]:
    all_articles: list[dict] = []
    # Free / higher-quota sources first; NewsAPI last (strict daily limits).
    fetchers: list[tuple[str, Any]] = [
        ("yfinance", lambda: fetch_yfinance_news(symbol)),
        ("alphavantage", lambda: fetch_alpha_vantage_news_sentiment(symbol, alphavantage_key)),
    ]
    if anthropic_web_search_enabled and anthropic_api_key:
        fetchers.append(
            (
                "anthropic_web",
                lambda: fetch_anthropic_web_news(
                    symbol,
                    anthropic_api_key,
                    model=anthropic_model,
                    max_uses=anthropic_web_search_max_uses,
                    delay_seconds=anthropic_delay_seconds,
                ),
            )
        )
    fetchers.append(
        (
            "newsapi",
            lambda: fetch_newsapi(
                symbol,
                newsapi_key,
                inter_request_delay_seconds=newsapi_delay_seconds,
            ),
        )
    )
    for source, fetcher in fetchers:
        try:
            articles = fetcher()
        except Exception as exc:
            logger.warning("news fetch failed symbol=%s source=%s err=%s", symbol, source, exc)
            articles = []
        if articles:
            normalized = [normalize_article(a) for a in articles]
            store.save_articles(symbol, source, normalized)
            all_articles.extend(normalized)

    if len(all_articles) < _MIN_ARTICLES_BEFORE_CACHE:
        cached = [normalize_article(a) for a in store.load_recent_articles(symbol, limit=40)]
        if cached:
            logger.info(
                "news cache merge symbol=%s live=%d cached=%d",
                symbol,
                len(all_articles),
                len(cached),
            )
            seen = {a.get("article_hash") for a in all_articles if a.get("article_hash")}
            for article in cached:
                h = article.get("article_hash")
                if h and h not in seen:
                    all_articles.append(article)
                    seen.add(h)

    counts = _count_by_source(all_articles)
    logger.info(
        "news symbol=%s total=%d by_source=%s",
        symbol,
        len(all_articles),
        counts,
    )
    return all_articles
