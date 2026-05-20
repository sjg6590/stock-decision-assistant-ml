from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def directional_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
    if len(y_true) == 0:
        return 0.0
    return float((y_true.values == y_pred).mean())


def directional_accuracy_at_threshold(y_true: pd.Series, proba: np.ndarray, threshold: float) -> float:
    """Accuracy using probability threshold instead of hard 0.5 predict()."""
    if len(y_true) == 0:
        return 0.0
    pred = (proba >= threshold).astype(int)
    return float((y_true.values == pred).mean())


def tune_threshold_on_val(y_val: pd.Series, proba_val: np.ndarray) -> float:
    """Grid-search [0.45, 0.65] in 0.01 steps; return threshold maximising F1 on val."""
    best_threshold = 0.5
    best_f1 = -1.0
    for t in np.arange(0.45, 0.651, 0.01):
        pred = (proba_val >= t).astype(int)
        if pred.sum() == 0:
            continue
        score = f1_score(y_val, pred, zero_division=0.0)
        if score > best_f1:
            best_f1 = score
            best_threshold = float(t)
    return best_threshold


def strategy_metrics_from_proba(
    proba: np.ndarray,
    horizon_returns: pd.Series,
    horizon_bars: int,
    threshold: float,
) -> tuple[float, float]:
    """Compute strategy Sharpe and max-drawdown using a probability threshold."""
    pred_up = (proba >= threshold).astype(int)
    return horizon_strategy_metrics(pred_up, horizon_returns, horizon_bars)


def sharpe_ratio(returns: pd.Series) -> float:
    if returns.empty or returns.std() == 0:
        return 0.0
    return float((returns.mean() / returns.std()) * np.sqrt(252))


def max_drawdown(curve: pd.Series) -> float:
    if curve.empty:
        return 0.0
    rolling_max = curve.cummax()
    drawdown = (curve - rolling_max) / rolling_max
    return abs(float(drawdown.min()))


def horizon_strategy_returns(
    predicted_up: np.ndarray,
    horizon_returns: pd.Series,
    horizon_bars: int,
) -> pd.Series:
    """Daily-equivalent PnL for one horizon (scale multi-day returns by bar count)."""
    daily_returns = horizon_returns.values / max(horizon_bars, 1)
    return pd.Series(predicted_up * daily_returns, dtype=float)


def horizon_strategy_metrics(
    predicted_up: np.ndarray,
    horizon_returns: pd.Series,
    horizon_bars: int,
) -> tuple[float, float]:
    strategy = horizon_strategy_returns(predicted_up, horizon_returns, horizon_bars)
    equity = (1 + strategy.fillna(0.0)).cumprod()
    return sharpe_ratio(strategy), max_drawdown(equity)


def aggregate_strategy_metrics(per_horizon: list[tuple[float, float]]) -> tuple[float, float]:
    if not per_horizon:
        return 0.0, 0.0
    sharpes, drawdowns = zip(*per_horizon)
    return float(np.mean(sharpes)), float(max(drawdowns))
