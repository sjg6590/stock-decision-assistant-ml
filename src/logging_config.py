from __future__ import annotations

import logging
import os
import re
import sys


class _Ansi:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[31m"
    GREEN = "\033[32m"
    YELLOW = "\033[33m"
    BLUE = "\033[34m"
    MAGENTA = "\033[35m"
    CYAN = "\033[36m"
    GRAY = "\033[90m"


_LEVEL_STYLES: dict[int, tuple[str, str]] = {
    logging.DEBUG: (_Ansi.GRAY, "DEBUG"),
    logging.INFO: (_Ansi.BLUE, "INFO"),
    logging.WARNING: (_Ansi.YELLOW, "WARNING"),
    logging.ERROR: (_Ansi.RED, "ERROR"),
    logging.CRITICAL: (_Ansi.RED + _Ansi.BOLD, "CRITICAL"),
}

# (regex, style) — first match wins; applied to the message body only.
_MESSAGE_RULES: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"^HTTP Request:"), _Ansi.DIM + _Ansi.GRAY),
    (re.compile(r"^(Loading|Loaded|Updating) token"), _Ansi.DIM + _Ansi.GRAY),
    (re.compile(r"^no alert\b"), _Ansi.YELLOW),
    (re.compile(r"^suppressed duplicate alert\b"), _Ansi.DIM + _Ansi.YELLOW),
    (re.compile(r"^DRY_RUN alert\b"), _Ansi.BOLD + _Ansi.GREEN),
    (re.compile(r"^retrain result\b"), _Ansi.CYAN),
    (re.compile(r"^using existing non-promoted model\b"), _Ansi.BLUE),
    (re.compile(r"^processing symbol="), _Ansi.BOLD + _Ansi.CYAN),
    (re.compile(r"^scheduler started\b"), _Ansi.BOLD + _Ansi.MAGENTA),
    (re.compile(r"^training (failed|produced)"), _Ansi.YELLOW),
    (re.compile(r"^cycle failed\b"), _Ansi.RED),
    (re.compile(r"^LLM sentiment failed\b"), _Ansi.YELLOW),
]

_SENTIMENT_WORDS: list[tuple[re.Pattern[str], str]] = [
    (re.compile(r"\bbullish\b", re.IGNORECASE), _Ansi.GREEN),
    (re.compile(r"\bbearish\b", re.IGNORECASE), _Ansi.RED),
    (re.compile(r"\bneutral\b", re.IGNORECASE), _Ansi.GRAY),
]

_SYMBOL_RE = re.compile(r"\bsymbol=([A-Z][A-Z0-9.-]*)\b")


def _use_color(stream: object | None = None) -> bool:
    if os.environ.get("NO_COLOR"):
        return False
    if stream is None:
        stream = sys.stderr
    return hasattr(stream, "isatty") and stream.isatty()


def _wrap(text: str, *codes: str) -> str:
    prefix = "".join(codes)
    return f"{prefix}{text}{_Ansi.RESET}"


def _highlight_message(message: str) -> str:
    for pattern, style in _MESSAGE_RULES:
        if pattern.search(message):
            message = _wrap(message, style)
            break

    for pattern, style in _SENTIMENT_WORDS:
        message = pattern.sub(lambda m, s=style: _wrap(m.group(0), s), message)

    message = _SYMBOL_RE.sub(
        lambda m: f"symbol={_wrap(m.group(1), _Ansi.BOLD, _Ansi.CYAN)}",
        message,
    )
    return message


class ColoredFormatter(logging.Formatter):
    def __init__(self, *, use_color: bool = True) -> None:
        super().__init__(
            fmt="%(asctime)s %(levelname)s %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        self._use_color = use_color

    def format(self, record: logging.LogRecord) -> str:
        if self._use_color:
            color, _ = _LEVEL_STYLES.get(record.levelno, (_Ansi.RESET, record.levelname))
            record.levelname = _wrap(record.levelname, color, _Ansi.BOLD)
            message = _highlight_message(record.getMessage())
            if record.levelno == logging.DEBUG:
                message = _wrap(message, _Ansi.DIM)
            record.msg = message
            record.args = ()
        return super().format(record)


def setup_logging(*, colors: bool | None = None, level: int = logging.INFO) -> None:
    """Configure root logging with optional ANSI colors for the terminal."""
    stream = sys.stderr
    if colors is None:
        colors = _use_color(stream)

    root = logging.getLogger()
    root.handlers.clear()
    root.setLevel(level)

    # Matplotlib font-cache and debug logs are never useful at DEBUG level.
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("matplotlib.font_manager").setLevel(logging.WARNING)

    handler = logging.StreamHandler(stream)
    handler.setFormatter(ColoredFormatter(use_color=colors))
    root.addHandler(handler)
