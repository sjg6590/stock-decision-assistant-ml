from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

_QUOTE_PRICE_FIELDS = (
    "mark",
    "lastPrice",
    "regularMarketLastPrice",
    "closePrice",
    "bidPrice",
)


def _candles_to_frame(payload: dict[str, Any]) -> pd.DataFrame:
    candles = payload.get("candles", [])
    rows = []
    for c in candles:
        rows.append(
            {
                "datetime": datetime.fromtimestamp(c["datetime"] / 1000, tz=timezone.utc),
                "open": float(c["open"]),
                "high": float(c["high"]),
                "low": float(c["low"]),
                "close": float(c["close"]),
                "volume": float(c["volume"]),
            }
        )
    return pd.DataFrame(rows)


def fetch_daily_history(client: Any, symbol: str, years: int = 5) -> pd.DataFrame:
    response = client.get_price_history_every_day(symbol, need_extended_hours_data=False, need_previous_close=True)
    response.raise_for_status()
    frame = _candles_to_frame(response.json())
    if frame.empty:
        return frame
    if years > 0:
        cutoff = frame["datetime"].max() - pd.Timedelta(days=365 * years)
        frame = frame.loc[frame["datetime"] >= cutoff]
    frame["symbol"] = symbol
    return frame.sort_values("datetime")


def fetch_quotes(client: Any, symbols: list[str]) -> dict[str, Any]:
    response = client.get_quotes(symbols)
    response.raise_for_status()
    return response.json()


def parse_quote_price(symbol: str, payload: dict[str, Any]) -> float | None:
    """Extract a usable last/mark price from a Schwab get_quotes payload."""
    if not payload:
        return None

    key = symbol.upper()
    entry = payload.get(key) or payload.get(symbol)
    if entry is None and len(payload) == 1:
        entry = next(iter(payload.values()))
    if not isinstance(entry, dict):
        return None

    quote = entry.get("quote")
    if not isinstance(quote, dict):
        quote = entry.get("regular")
    if not isinstance(quote, dict):
        quote = entry

    if not isinstance(quote, dict):
        return None

    for field in _QUOTE_PRICE_FIELDS:
        raw = quote.get(field)
        if raw is None:
            continue
        try:
            price = float(raw)
        except (TypeError, ValueError):
            continue
        if price > 0:
            return price
    return None


def fetch_live_price(client: Any, symbol: str) -> float | None:
    try:
        payload = fetch_quotes(client, [symbol])
    except Exception as exc:
        logger.warning("live quote fetch failed symbol=%s err=%s", symbol, exc)
        return None
    price = parse_quote_price(symbol, payload)
    if price is None:
        logger.warning("live quote parse failed symbol=%s", symbol)
    return price
