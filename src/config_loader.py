from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

_FEATURES_DEFAULTS: dict[str, Any] = {
    "market_reference_symbol": "SPY",
}


_COLD_START_DEFAULTS: dict[str, Any] = {
    "enabled": False,
    "min_rows": 600,
    "val_size": 60,
    "holdout_size": 60,
    "promotion": {
        "aggregate_accuracy_min": 0.57,
        "sharpe_min": 0.60,
        "max_drawdown_max": 0.12,
    },
    "tag": "cold_start",
    "confidence_discount": 0.1,
}


def load_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def normalize_train_cold_start(data: dict[str, Any]) -> dict[str, Any]:
    """Merge train_cold_start with defaults; deep-merge promotion overrides."""
    raw = data.get("train_cold_start")
    if raw is None:
        data["train_cold_start"] = dict(_COLD_START_DEFAULTS)
        return data
    merged: dict[str, Any] = {**_COLD_START_DEFAULTS, **raw}
    merged["promotion"] = {
        **_COLD_START_DEFAULTS["promotion"],
        **(raw.get("promotion") or {}),
    }
    data["train_cold_start"] = merged
    return data


def normalize_features(data: dict[str, Any]) -> dict[str, Any]:
    """Merge features block with defaults."""
    raw = data.get("features")
    if raw is None:
        data["features"] = dict(_FEATURES_DEFAULTS)
    else:
        data["features"] = {**_FEATURES_DEFAULTS, **raw}
    return data


def normalize_thresholds(data: dict[str, Any]) -> dict[str, Any]:
    """Apply defaults for optional config blocks (e.g. train_cold_start, features)."""
    normalize_features(data)
    return normalize_train_cold_start(data)


def load_thresholds(path: Path) -> dict[str, Any]:
    return normalize_thresholds(load_yaml(path))


def train_cold_start(thresholds: dict[str, Any]) -> dict[str, Any]:
    """Return the normalized train_cold_start block."""
    return thresholds.get("train_cold_start", _COLD_START_DEFAULTS)


def cold_start_confidence_discount(thresholds: dict[str, Any]) -> float:
    return float(train_cold_start(thresholds).get("confidence_discount", 0.1))


def market_reference_symbol(thresholds: dict[str, Any]) -> str:
    """Return the benchmark ticker used for cross-sectional return features."""
    features = thresholds.get("features", _FEATURES_DEFAULTS)
    return str(features.get("market_reference_symbol", _FEATURES_DEFAULTS["market_reference_symbol"])).upper()
