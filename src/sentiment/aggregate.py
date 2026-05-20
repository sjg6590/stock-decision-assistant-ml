from __future__ import annotations

from typing import Any


def aggregate_articles(articles: list[dict[str, Any]], limit: int = 25) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for article in sorted(articles, key=lambda a: a.get("published_at", ""), reverse=True):
        h = article.get("article_hash")
        if not h or h in seen:
            continue
        seen.add(h)
        deduped.append(article)
        if len(deduped) >= limit:
            break
    return deduped


def alpha_vantage_soft_score(articles: list[dict[str, Any]]) -> float:
    vals = []
    for article in articles:
        if article.get("source") == "alphavantage":
            vals.append(float(article.get("overall_sentiment_score", 0.0)))
    if not vals:
        return 0.0
    return sum(vals) / len(vals)
