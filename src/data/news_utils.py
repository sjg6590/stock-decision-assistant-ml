from __future__ import annotations

from typing import Any


def article_summary(article: dict[str, Any]) -> str:
    """Return the best available short text for an article (summary or description)."""
    for key in ("summary", "description"):
        text = str(article.get(key, "") or "").strip()
        if text:
            return text
    return ""


def normalize_article(article: dict[str, Any]) -> dict[str, Any]:
    """Ensure articles expose a summary field for downstream LLM prompts."""
    out = dict(article)
    if not str(out.get("summary", "") or "").strip():
        summary = article_summary(out)
        if summary:
            out["summary"] = summary
    return out
