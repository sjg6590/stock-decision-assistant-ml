from __future__ import annotations

from dataclasses import dataclass

from sda_types import FusedSignal, SentimentResult


@dataclass
class AlertMessage:
    subject: str
    body: str


def _format_factor_block(title: str, factors: list[str], *, limit: int = 5) -> str:
    items = [item.strip() for item in factors if item.strip()][:limit]
    if not items:
        return f"{title}: (none)\n"
    lines = [f"{title}:"] + [f"  - {item}" for item in items]
    return "\n".join(lines) + "\n"


def format_alert(symbol: str, signal: FusedSignal, sentiment: SentimentResult) -> AlertMessage:
    subject = f"[Stock Decision Assistant] {symbol} BUY candidate"
    body = (
        f"Symbol: {symbol}\n"
        f"Signal: {signal.signal_type}\n"
        f"Confidence: {signal.confidence:.2f}\n"
        f"Reason: {signal.reason}\n\n"
        f"Sentiment: {sentiment.sentiment_label} ({sentiment.confidence:.2f})\n"
        f"{_format_factor_block('Bullish factors', sentiment.bullish_factors)}"
        f"{_format_factor_block('Bearish factors', sentiment.bearish_factors)}"
        f"{_format_factor_block('Macro risks', sentiment.macro_risks, limit=3)}\n"
        f"Sell guidance: {signal.sell_guide}\n"
    )
    return AlertMessage(subject=subject, body=body)
