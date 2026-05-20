from __future__ import annotations

from config_loader import market_reference_symbol, normalize_thresholds


def test_market_reference_symbol_default() -> None:
    thresholds = normalize_thresholds({})
    assert market_reference_symbol(thresholds) == "SPY"


def test_market_reference_symbol_override() -> None:
    thresholds = normalize_thresholds({"features": {"market_reference_symbol": "qqq"}})
    assert market_reference_symbol(thresholds) == "QQQ"
