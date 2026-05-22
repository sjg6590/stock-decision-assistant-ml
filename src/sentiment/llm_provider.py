from __future__ import annotations

import json
import logging
import re
import time
from typing import Any

import httpx
from anthropic import Anthropic
from openai import OpenAI

from data.news_anthropic_web import (
    anthropic_web_search_tool,
    create_anthropic_client,
    is_anthropic_rate_limited,
    is_rate_limit_error,
    is_web_search_tool_error,
    mark_anthropic_rate_limited,
    maybe_anthropic_delay,
    note_anthropic_call,
    rate_limit_retry_seconds,
)
from settings import Settings

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a quantitative financial news analyst screening stocks for a short-term algorithmic trading system.
Your job is to assess whether recent news sentiment supports or contradicts a bullish ML signal for a given ticker.

Output ONLY a single JSON object — no prose, no markdown, no code fences — with exactly these keys:

{
  "sentiment_label": "<bullish|neutral|bearish>",
  "confidence": <float 0.0–1.0>,
  "recommended_action": "<agree|neutral|disagree>",
  "bullish_factors": ["<specific factor>", ...],
  "bearish_factors": ["<specific factor>", ...],
  "macro_risks": ["<systemic or sector-wide risk>", ...],
  "summary": "<2–3 sentence synthesis>"
}

Field rules:
- sentiment_label: MUST be exactly one of "bullish", "neutral", or "bearish". No other values.
- confidence: calibrated probability that the sentiment_label is correct.
  0.0 = no usable signal (e.g. no relevant articles).
  0.3–0.5 = weak or mixed evidence.
  0.6–0.75 = moderate conviction with supporting evidence.
  0.8+ = strong, consistent signal across multiple credible sources.
  Default most outputs to 0.3–0.65; reserve 0.75+ for exceptional clarity.
- recommended_action: MUST be exactly one of:
  "agree"     — news sentiment supports the ML bullish signal.
  "neutral"   — sentiment is ambiguous or unrelated to price direction.
  "disagree"  — news sentiment contradicts or weakens the ML bullish signal.
- bullish_factors: 2–5 specific, concrete drivers (e.g. "Beat Q2 EPS by 12%", "FDA approval granted").
  Avoid vague phrases like "positive news" or "good results".
- bearish_factors: 2–5 specific headwinds. Include even when sentiment_label is bullish.
- macro_risks: 1–3 systemic or sector-level risks (e.g. "Fed rate uncertainty", "semiconductor supply crunch").
  Distinct from company-specific bearish_factors.
- summary: Synthesize the overall picture in 2–3 sentences. Mention how the news aligns or conflicts
  with the ML model signal horizon provided in the user message.
- When the user message includes authoritative market prices (last close / live quote), base any
  percent-upside or price-level commentary on those figures, not on prices quoted in older articles.

Source weighting:
- Prioritise articles published within the last 24 hours over older ones.
- Weight major financial outlets (Reuters, Bloomberg, WSJ, FT) more heavily than aggregators or blogs.
- If sources conflict, acknowledge the disagreement and lower confidence accordingly.
- Ignore articles clearly unrelated to the ticker's business or near-term price drivers.
"""


def _httpx_timeout(settings: Settings) -> httpx.Timeout:
    seconds = float(settings.llm_timeout_seconds)
    return httpx.Timeout(seconds, connect=min(10.0, seconds))


def _neutral_payload(reason: str) -> dict[str, Any]:
    return {
        "bullish_factors": [],
        "bearish_factors": [reason],
        "macro_risks": [],
        "sentiment_label": "neutral",
        "confidence": 0.0,
        "summary": f"Sentiment unavailable: {reason}",
        "recommended_action": "neutral",
    }


_THINKING_BLOCK_RE = re.compile(
    r"<(?:think|thinking|redacted_reasoning|redacted_thinking)[^>]*>.*?</(?:think|thinking|redacted_reasoning|redacted_thinking)>",
    re.IGNORECASE | re.DOTALL,
)
_THINKING_OPEN_RE = re.compile(
    r"^<(?:think|thinking|redacted_reasoning|redacted_thinking)[^>]*>.*",
    re.IGNORECASE | re.DOTALL,
)
_CODE_FENCE_RE = re.compile(r"```(?:json)?\s*\n?(.*?)\n?```", re.DOTALL | re.IGNORECASE)


def _strip_thinking_blocks(text: str) -> str:
    cleaned = _THINKING_BLOCK_RE.sub("", text)
    cleaned = _THINKING_OPEN_RE.sub("", cleaned)
    return cleaned.strip()


def _find_balanced_json_object(text: str, start: int = 0) -> str | None:
    idx = text.find("{", start)
    if idx < 0:
        return None
    depth = 0
    in_string = False
    escape = False
    for i in range(idx, len(text)):
        ch = text[i]
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return text[idx : i + 1]
    return None


def _json_candidates(text: str) -> list[str]:
    candidates: list[str] = []
    seen: set[str] = set()

    def add(candidate: str) -> None:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            return
        seen.add(candidate)
        candidates.append(candidate)

    add(text)
    stripped = _strip_thinking_blocks(text)
    add(stripped)
    for source in (text, stripped):
        for match in _CODE_FENCE_RE.finditer(source):
            add(match.group(1))
        pos = 0
        while True:
            obj = _find_balanced_json_object(source, start=pos)
            if not obj:
                break
            add(obj)
            pos = source.find("{", pos) + 1
    return candidates


def _parse_llm_json(text: str) -> dict[str, Any] | None:
    for candidate in _json_candidates(text):
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _safe_json(text: str) -> dict[str, Any]:
    parsed = _parse_llm_json(text)
    if parsed is not None:
        return parsed
    logger.warning("LLM response was not valid JSON after extraction attempts")
    return {
        "bullish_factors": [],
        "bearish_factors": ["LLM response was not valid JSON"],
        "macro_risks": [],
        "sentiment_label": "neutral",
        "confidence": 0.0,
        "summary": text[:300],
        "recommended_action": "neutral",
    }


_OLLAMA_NUM_PREDICT = 2048
_OLLAMA_MAX_ATTEMPTS = 2


def _ollama_raw_call(
    settings: Settings,
    prompt: str,
    temperature: float = 0.2,
) -> str:
    with httpx.Client(timeout=_httpx_timeout(settings)) as client:
        resp = client.post(
            f"{settings.ollama_base_url.rstrip('/')}/api/generate",
            json={
                "model": settings.ollama_model,
                "prompt": f"{SYSTEM_PROMPT}\n\n{prompt}",
                "stream": False,
                "format": "json",
                "options": {"num_predict": _OLLAMA_NUM_PREDICT, "temperature": temperature},
            },
        )
        resp.raise_for_status()
        return resp.json().get("response", "{}")


def analyze_with_llm(settings: Settings, prompt: str) -> dict[str, Any]:
    provider = settings.llm_provider.lower().strip()
    if provider == "openai":
        client = OpenAI(api_key=settings.openai_api_key)
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            temperature=0.2,
            messages=[{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": prompt}],
        )
        text = resp.choices[0].message.content or "{}"
        return _safe_json(text)
    if provider == "anthropic":
        maybe_anthropic_delay(settings.anthropic_delay_seconds)
        client = create_anthropic_client(settings.anthropic_api_key)
        create_kwargs: dict[str, Any] = {
            "model": settings.anthropic_model,
            "max_tokens": 1024,
            "temperature": 0.2,
            "system": SYSTEM_PROMPT,
            "messages": [{"role": "user", "content": prompt}],
        }
        use_web_search = (
            settings.anthropic_web_search_enabled
            and settings.anthropic_web_search_on_sentiment
            and not is_anthropic_rate_limited()
        )
        if use_web_search:
            create_kwargs["tools"] = [
                anthropic_web_search_tool(max_uses=min(settings.anthropic_web_search_max_uses, 1))
            ]
        resp = None
        try:
            resp = client.messages.create(**create_kwargs)
        except Exception as exc:
            if is_rate_limit_error(exc):
                mark_anthropic_rate_limited("*", context="sentiment")
                if create_kwargs.get("tools"):
                    create_kwargs.pop("tools", None)
                    try:
                        resp = client.messages.create(**create_kwargs)
                    except Exception as retry_exc:
                        if is_rate_limit_error(retry_exc):
                            wait_s = rate_limit_retry_seconds(retry_exc)
                            logger.warning(
                                "Anthropic rate limited on sentiment; retrying in %.0fs",
                                wait_s,
                            )
                            time.sleep(wait_s)
                            maybe_anthropic_delay(settings.anthropic_delay_seconds)
                            resp = client.messages.create(**create_kwargs)
                        else:
                            raise
                else:
                    wait_s = rate_limit_retry_seconds(exc)
                    logger.warning(
                        "Anthropic rate limited on sentiment; retrying in %.0fs",
                        wait_s,
                    )
                    time.sleep(wait_s)
                    maybe_anthropic_delay(settings.anthropic_delay_seconds)
                    resp = client.messages.create(**create_kwargs)
            elif create_kwargs.get("tools") and is_web_search_tool_error(exc):
                logger.warning(
                    "Anthropic web search unavailable for sentiment; retrying without web search: %s",
                    exc,
                )
                create_kwargs.pop("tools", None)
                resp = client.messages.create(**create_kwargs)
            else:
                raise
        finally:
            note_anthropic_call()
        if resp is None:
            raise RuntimeError("Anthropic sentiment call returned no response")
        text = "".join(chunk.text for chunk in resp.content if hasattr(chunk, "text"))
        return _safe_json(text)
    if provider == "ollama":
        try:
            text = _ollama_raw_call(settings, prompt, temperature=0.2)
            parsed = _parse_llm_json(text)
            if parsed is not None:
                return parsed
            for attempt in range(2, _OLLAMA_MAX_ATTEMPTS + 1):
                logger.warning("retrying Ollama for JSON attempt=%d", attempt)
                text = _ollama_raw_call(settings, prompt, temperature=0.0)
                parsed = _parse_llm_json(text)
                if parsed is not None:
                    return parsed
            return _safe_json(text)
        except httpx.TimeoutException as exc:
            logger.warning("Ollama request timed out after %ss: %s", settings.llm_timeout_seconds, exc)
            return _neutral_payload(f"Ollama timed out after {settings.llm_timeout_seconds}s")
        except httpx.HTTPError as exc:
            logger.warning("Ollama request failed: %s", exc)
            return _neutral_payload(str(exc))
    raise ValueError(f"Unsupported LLM_PROVIDER: {settings.llm_provider}")
