from __future__ import annotations

from datetime import date

from sda_types import HorizonPrediction, ModelPrediction


def _horizon_threshold(p: HorizonPrediction, ml_threshold: float) -> float:
    return p.threshold if p.threshold > 0 else ml_threshold


def _bullish_predictions(pred: ModelPrediction, ml_threshold: float) -> list[HorizonPrediction]:
    return [
        p
        for p in pred.predictions
        if p.probability_up >= _horizon_threshold(p, ml_threshold)
    ]


def _pick_timing_horizon(pred: ModelPrediction, ml_threshold: float) -> HorizonPrediction:
    """Horizon with strongest directional conviction (for exit timing)."""
    bullish = _bullish_predictions(pred, ml_threshold)
    if bullish:
        return max(bullish, key=lambda p: p.probability_up)
    return max(pred.predictions, key=lambda p: p.probability_up)


def _pick_return_horizon(pred: ModelPrediction) -> HorizonPrediction:
    positive = [p for p in pred.predictions if p.expected_return > 0]
    if positive:
        return max(positive, key=lambda p: p.expected_return)
    return max(pred.predictions, key=lambda p: p.expected_return)


def _targets(ref_price: float, expected_return: float) -> tuple[float, float, float]:
    take_profit_low = ref_price * (1 + expected_return * 0.8)
    take_profit_high = ref_price * (1 + expected_return * 1.2)
    stop_loss = ref_price * (1 - 0.03)
    return take_profit_low, take_profit_high, stop_loss


def _format_bar_date(last_bar_date: date | None) -> str:
    return last_bar_date.isoformat() if last_bar_date else "unknown"


def _priced_in_note(live_price: float, close_tp_low: float, close_tp_high: float) -> str | None:
    if live_price >= close_tp_high:
        return (
            "Live price is at or above the take-profit band vs last close — "
            "much of the modeled move may already be priced in; treat close-based targets as historical context."
        )
    if live_price >= close_tp_low:
        return (
            "Live price is inside the take-profit band vs last close — "
            "part of the modeled move may already be reflected; prefer live-anchored targets for new entries."
        )
    return None


def build_sell_guidance(
    pred: ModelPrediction,
    *,
    latest_close: float,
    last_bar_date: date | None = None,
    live_price: float | None = None,
    ml_threshold: float = 0.60,
) -> str:
    if not pred.predictions:
        return "No sell guidance because no ML prediction is available."

    timing = _pick_timing_horizon(pred, ml_threshold)
    return_horizon = _pick_return_horizon(pred)
    exp_ret = return_horizon.expected_return
    exp_pct = exp_ret * 100

    close_tp_low, close_tp_high, close_stop = _targets(latest_close, exp_ret)
    bar_date = _format_bar_date(last_bar_date)
    anchor = live_price if live_price is not None and live_price > 0 else latest_close
    anchor_label = "live" if live_price is not None and live_price > 0 else "last close"

    parts: list[str] = [
        f"Reference close ${latest_close:.2f} ({bar_date})",
    ]
    if live_price is not None and live_price > 0:
        parts.append(f"live ${live_price:.2f}")
    else:
        parts.append("live quote unavailable")

    if timing.horizon != return_horizon.horizon:
        parts.append(
            f"Exit timing horizon: {timing.horizon} (strongest ML conviction). "
            f"Take-profit uses {return_horizon.horizon} modeled upside ({exp_pct:+.2f}%)"
        )
    else:
        parts.append(f"Preferred exit horizon: {timing.horizon} (modeled upside {exp_pct:+.2f}%)")

    live_tp_low, live_tp_high, live_stop = _targets(anchor, exp_ret)
    parts.append(
        f"Take-profit (actionable, from {anchor_label}): ${live_tp_low:.2f}-${live_tp_high:.2f}"
    )
    parts.append(f"Stop (from {anchor_label}): ${live_stop:.2f}")
    parts.append(
        f"Take-profit vs last close (historical): ${close_tp_low:.2f}-${close_tp_high:.2f}"
    )
    parts.append(f"Stop vs last close: ${close_stop:.2f}")

    if abs(exp_ret) < 0.005:
        parts.append(
            "Note: ML regressor expects a very small move on all horizons; "
            "news price targets may exceed this quantitative band."
        )
    elif live_price is not None and live_price > 0:
        note = _priced_in_note(live_price, close_tp_low, close_tp_high)
        if note:
            parts.append(f"Note: {note}")

    return ". ".join(parts) + "."
