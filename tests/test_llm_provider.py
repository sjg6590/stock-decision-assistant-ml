from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from anthropic import RateLimitError

from data.news_anthropic_web import rate_limit_retry_seconds, reset_anthropic_rate_limit_state
from sentiment.analyze import _article_limit, _blurb_char_limit
from sentiment.llm_provider import analyze_with_llm
from settings import Settings


def test_rate_limit_retry_seconds_uses_retry_after_header() -> None:
    response = MagicMock()
    response.headers = {"retry-after": "45"}
    exc = RateLimitError("rate limited", response=response, body=None)
    assert rate_limit_retry_seconds(exc) == 45.0


def test_rate_limit_retry_seconds_defaults_to_sixty() -> None:
    exc = RateLimitError("rate limited", response=MagicMock(headers={}), body=None)
    assert rate_limit_retry_seconds(exc) == 60.0


def test_anthropic_article_and_blurb_limits() -> None:
    settings = Settings(LLM_PROVIDER="anthropic", LLM_ARTICLE_LIMIT=25)
    assert _article_limit(settings) == 8
    assert _blurb_char_limit(settings) == 100


@patch("sentiment.llm_provider.time.sleep")
@patch("sentiment.llm_provider.create_anthropic_client")
def test_analyze_with_llm_retries_sentiment_after_rate_limit(
    client_factory: MagicMock,
    sleep_mock: MagicMock,
) -> None:
    reset_anthropic_rate_limit_state()
    settings = Settings(
        LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_WEB_SEARCH_ON_SENTIMENT=False,
    )
    ok_response = SimpleNamespace(
        content=[SimpleNamespace(text='{"sentiment_label":"bullish","confidence":0.7,'
                                  '"recommended_action":"agree","bullish_factors":[],"bearish_factors":[],'
                                  '"macro_risks":[],"summary":"ok"}')]
    )
    client = client_factory.return_value
    client.messages.create.side_effect = [
        RateLimitError("rate limited", response=MagicMock(headers={"retry-after": "1"}), body=None),
        ok_response,
    ]

    result = analyze_with_llm(settings, "Ticker: TEST\nNews articles:\n- headline")

    assert result["sentiment_label"] == "bullish"
    assert client.messages.create.call_count == 2
    sleep_mock.assert_called_once_with(1.0)


@patch("sentiment.llm_provider.create_anthropic_client")
def test_analyze_with_llm_drops_web_search_tools_on_rate_limit(
    client_factory: MagicMock,
) -> None:
    reset_anthropic_rate_limit_state()
    settings = Settings(
        LLM_PROVIDER="anthropic",
        ANTHROPIC_API_KEY="test-key",
        ANTHROPIC_WEB_SEARCH_ENABLED=True,
        ANTHROPIC_WEB_SEARCH_ON_SENTIMENT=True,
    )
    ok_response = SimpleNamespace(
        content=[SimpleNamespace(text='{"sentiment_label":"neutral","confidence":0.5,'
                                  '"recommended_action":"neutral","bullish_factors":[],"bearish_factors":[],'
                                  '"macro_risks":[],"summary":"ok"}')]
    )
    client = client_factory.return_value
    client.messages.create.side_effect = [
        RateLimitError("rate limited", response=MagicMock(headers={}), body=None),
        ok_response,
    ]

    with patch("sentiment.llm_provider.time.sleep"):
        analyze_with_llm(settings, "Ticker: TEST\nNews articles:\n- headline")

    assert client.messages.create.call_count == 2
    assert "tools" not in client.messages.create.call_args_list[1].kwargs
