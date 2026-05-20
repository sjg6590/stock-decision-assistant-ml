from __future__ import annotations

import logging

from logging_config import ColoredFormatter, _highlight_message, setup_logging


def test_highlight_no_alert():
    msg = "no alert symbol=AAPL reason=ML bullish horizons=0, sentiment=neutral (0.00)"
    styled = _highlight_message(msg)
    assert "no alert" in styled
    assert "AAPL" in styled


def test_setup_logging_no_color():
    setup_logging(colors=False, level=logging.INFO)
    root = logging.getLogger()
    assert len(root.handlers) == 1
    assert isinstance(root.handlers[0].formatter, ColoredFormatter)
    assert root.handlers[0].formatter._use_color is False
