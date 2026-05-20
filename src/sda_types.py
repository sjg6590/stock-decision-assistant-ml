from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class HorizonPrediction:
    horizon: str
    bars: int
    probability_up: float
    expected_return: float
    threshold: float = 0.0  # per-horizon buy threshold from training; 0.0 → caller uses global fallback


@dataclass
class ModelPrediction:
    symbol: str
    generated_at: datetime
    predictions: list[HorizonPrediction] = field(default_factory=list)
    best_horizon: str = ""
    ml_confidence: float = 0.0


@dataclass
class SentimentResult:
    symbol: str
    sentiment_label: str
    confidence: float
    bullish_factors: list[str]
    bearish_factors: list[str]
    macro_risks: list[str]
    summary: str
    recommended_action: str


@dataclass
class FusedSignal:
    symbol: str
    signal_type: str
    confidence: float
    should_notify: bool
    reason: str
    sell_guide: str
