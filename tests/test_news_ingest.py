from __future__ import annotations

from unittest.mock import MagicMock, patch

from data.ingest import ingest_news_for_symbol
from data.news_alphavantage import (
    fetch_alpha_vantage_news_sentiment,
    reset_alphavantage_rate_limit_state,
)
from data.news_newsapi import fetch_newsapi, reset_newsapi_rate_limit_state
from data.news_utils import article_summary, normalize_article
from data.news_yfinance import fetch_yfinance_news, normalize_yfinance_item


def test_article_summary_prefers_summary_then_description() -> None:
    assert article_summary({"summary": "beat earnings"}) == "beat earnings"
    assert article_summary({"description": "revenue up"}) == "revenue up"
    assert article_summary({"summary": "a", "description": "b"}) == "a"


def test_normalize_article_copies_description_to_summary() -> None:
    article = normalize_article({"title": "x", "description": "details"})
    assert article["summary"] == "details"


def test_normalize_yfinance_item_nested_content_schema() -> None:
    parsed = normalize_yfinance_item(
        {
            "id": "abc",
            "content": {
                "title": "Apple beats earnings",
                "summary": "EPS up 12% year over year.",
                "pubDate": "2026-05-19T19:40:37Z",
                "provider": {"displayName": "Reuters"},
                "canonicalUrl": {"url": "https://finance.yahoo.com/news/apple-beats.html"},
            },
        }
    )
    assert parsed is not None
    assert parsed["title"] == "Apple beats earnings"
    assert parsed["url"] == "https://finance.yahoo.com/news/apple-beats.html"
    assert parsed["publisher"] == "Reuters"
    assert parsed["published_at"] == "2026-05-19T19:40:37Z"
    assert parsed["summary"] == "EPS up 12% year over year."


def test_normalize_yfinance_item_legacy_schema() -> None:
    parsed = normalize_yfinance_item(
        {
            "title": "Legacy headline",
            "link": "https://example.com/legacy",
            "publisher": "Bloomberg",
            "providerPublishTime": 1_700_000_000,
        }
    )
    assert parsed is not None
    assert parsed["title"] == "Legacy headline"
    assert parsed["url"] == "https://example.com/legacy"
    assert parsed["publisher"] == "Bloomberg"
    assert "2023" in parsed["published_at"]


def test_normalize_yfinance_item_skips_empty_title() -> None:
    assert normalize_yfinance_item({"content": {"summary": "no title"}}) is None
    assert normalize_yfinance_item({}) is None


def test_fetch_yfinance_news_uses_nested_schema() -> None:
    nested_item = {
        "id": "abc",
        "content": {
            "title": "Apple beats earnings",
            "summary": "EPS up 12%.",
            "pubDate": "2026-05-19T19:40:37Z",
            "canonicalUrl": {"url": "https://finance.yahoo.com/news/apple-beats.html"},
        },
    }
    ticker = MagicMock()
    ticker.news = [nested_item]

    with patch("data.news_yfinance.yf.Ticker", return_value=ticker):
        articles = fetch_yfinance_news("AAPL", limit=5)

    assert len(articles) == 1
    assert articles[0]["title"] == "Apple beats earnings"
    assert articles[0]["summary"] == "EPS up 12%."
    assert articles[0]["url"] == "https://finance.yahoo.com/news/apple-beats.html"


def test_fetch_alphavantage_rate_limit_short_circuits() -> None:
    reset_alphavantage_rate_limit_state()
    response = MagicMock()
    response.status_code = 200
    response.json.return_value = {
        "Note": (
            "Thank you for using Alpha Vantage! Our standard API rate limit is "
            "25 requests per day. Please subscribe to any of the premium plans."
        )
    }

    with patch("data.news_alphavantage.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.get.return_value = response
        assert fetch_alpha_vantage_news_sentiment("AAPL", "test-key") == []
        assert fetch_alpha_vantage_news_sentiment("MSFT", "test-key") == []

    assert client_cls.return_value.__enter__.return_value.get.call_count == 1


def test_fetch_newsapi_handles_429_without_raising() -> None:
    reset_newsapi_rate_limit_state()
    response = MagicMock()
    response.status_code = 429

    with patch("data.news_newsapi.httpx.Client") as client_cls:
        client_cls.return_value.__enter__.return_value.get.return_value = response
        assert fetch_newsapi("AAPL", "test-key") == []
        # Second call should short-circuit without HTTP
        assert fetch_newsapi("MSFT", "test-key") == []


def test_ingest_merges_cache_when_live_fetch_is_empty() -> None:
    store = MagicMock()
    store.load_recent_articles.return_value = [
        {
            "article_hash": "cached1",
            "symbol": "AAPL",
            "source": "newsapi",
            "title": "Cached headline",
            "summary": "Cached body",
            "published_at": "2026-05-19T00:00:00Z",
        }
    ]

    with (
        patch("data.ingest.fetch_yfinance_news", return_value=[]),
        patch("data.ingest.fetch_alpha_vantage_news_sentiment", return_value=[]),
        patch("data.ingest.fetch_newsapi", return_value=[]),
    ):
        articles = ingest_news_for_symbol(store, "AAPL", "news-key", "av-key")

    assert len(articles) == 1
    assert articles[0]["title"] == "Cached headline"
