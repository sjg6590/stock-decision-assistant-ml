from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from anthropic import RateLimitError

from data.news_anthropic_web import (
    extract_articles_from_response,
    fetch_anthropic_web_news,
    is_anthropic_rate_limited,
    reset_anthropic_rate_limit_state,
)


def _result_block(title: str, url: str) -> SimpleNamespace:
    return SimpleNamespace(type="web_search_result", title=title, url=url)


def _tool_result_block(*results: SimpleNamespace) -> SimpleNamespace:
    return SimpleNamespace(type="web_search_tool_result", content=list(results))


def test_extract_articles_from_web_search_blocks() -> None:
    response = SimpleNamespace(
        content=[
            _tool_result_block(
                _result_block("Apple beats earnings", "https://example.com/aapl-earnings"),
                _result_block("Apple guidance raised", "https://example.com/aapl-guide"),
            )
        ]
    )
    articles = extract_articles_from_response("AAPL", response)
    assert len(articles) == 2
    assert articles[0]["source"] == "anthropic_web"
    assert articles[0]["symbol"] == "AAPL"
    assert articles[0]["title"] == "Apple beats earnings"
    assert articles[0]["url"] == "https://example.com/aapl-earnings"


def test_fetch_anthropic_web_news_returns_empty_without_key() -> None:
    reset_anthropic_rate_limit_state()
    assert fetch_anthropic_web_news("AAPL", "") == []


def test_fetch_anthropic_web_news_rate_limit_short_circuits() -> None:
    reset_anthropic_rate_limit_state()
    with patch("data.news_anthropic_web.create_anthropic_client") as client_factory:
        client_factory.return_value.messages.create.side_effect = RateLimitError(
            "rate limited", response=MagicMock(), body=None
        )
        assert fetch_anthropic_web_news("AAPL", "test-key") == []
        assert fetch_anthropic_web_news("MSFT", "test-key") == []

    assert is_anthropic_rate_limited()
    assert client_factory.return_value.messages.create.call_count == 1


def test_fetch_anthropic_web_news_calls_api() -> None:
    reset_anthropic_rate_limit_state()
    mock_response = SimpleNamespace(
        content=[
            _tool_result_block(
                _result_block("MSFT cloud growth", "https://example.com/msft"),
            )
        ]
    )
    with patch("data.news_anthropic_web.create_anthropic_client") as client_factory:
        client_factory.return_value.messages.create.return_value = mock_response
        articles = fetch_anthropic_web_news("MSFT", "test-key", max_uses=2)

    assert len(articles) == 1
    assert articles[0]["title"] == "MSFT cloud growth"
    client_factory.return_value.messages.create.assert_called_once()
    tools = client_factory.return_value.messages.create.call_args.kwargs["tools"]
    assert tools[0]["type"] == "web_search_20260209"
    assert tools[0]["max_uses"] == 2
    assert tools[0]["allowed_callers"] == ["direct"]


def test_is_web_search_tool_error() -> None:
    from data.news_anthropic_web import is_web_search_tool_error

    exc = Exception(
        "Error code: 400 - does not support programmatic tool calling. "
        "web_search. allowed_callers=[\"direct\"]"
    )
    assert is_web_search_tool_error(exc) is True
    assert is_web_search_tool_error(Exception("timeout")) is False
