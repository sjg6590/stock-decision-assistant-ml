from __future__ import annotations

import logging
from datetime import datetime, timezone

from ml.features import build_features, resolve_model_features
from ml.registry import load_models
from sda_types import HorizonPrediction, ModelPrediction

logger = logging.getLogger(__name__)


def _horizon_confidence_margin(p: HorizonPrediction) -> tuple[float, float]:
    """Primary: margin above decision boundary. Tiebreaker: expected_return (clamped to 0)."""
    margin = p.probability_up - (p.threshold if p.threshold > 0 else 0.5)
    return (margin, max(p.expected_return, 0.0))


def predict_symbol(
    model_dir,
    symbol: str,
    version: str,
    bars,
    horizons: dict[str, int],
    debug: bool = False,
    ml_buy_threshold_fallback: float = 0.60,
    spy_frame=None,
    signal_config: dict | None = None,
) -> ModelPrediction:
    payload = load_models(model_dir, symbol, version)
    feat_df = build_features(bars, spy_frame).dropna().reset_index(drop=True)
    if feat_df.empty:
        return ModelPrediction(symbol=symbol, generated_at=datetime.now(tz=timezone.utc))

    feats = resolve_model_features(payload)
    x_last = feat_df[feats].tail(1)

    logger.debug(
        "feature_values symbol=%s %s",
        symbol,
        {col: round(float(x_last[col].iloc[0]), 5) for col in feats},
    )

    # Per-horizon thresholds saved during training; absent in older artifacts
    saved_thresholds: dict[str, float] = payload.get("thresholds", {})

    preds: list[HorizonPrediction] = []
    for hz, bars_out in horizons.items():
        clf = payload["classifiers"][hz]
        reg = payload["regressors"][hz]
        prob_up = float(clf.predict_proba(x_last)[0][1])
        exp_ret = float(reg.predict(x_last)[0])
        threshold = saved_thresholds.get(hz, ml_buy_threshold_fallback)
        logger.debug(
            "horizon_pred symbol=%s horizon=%s prob_up=%.3f exp_ret=%.4f threshold=%.2f",
            symbol, hz, prob_up, exp_ret, threshold if threshold > 0 else float("nan"),
        )
        preds.append(
            HorizonPrediction(
                horizon=hz,
                bars=bars_out,
                probability_up=prob_up,
                expected_return=exp_ret,
                threshold=threshold,
            )
        )

    selection_method = (signal_config or {}).get("best_horizon_selection", "margin_above_threshold")
    if selection_method == "proba_x_return":
        best = max(preds, key=lambda p: p.probability_up * max(p.expected_return, 0.0001))
    else:
        best = max(preds, key=_horizon_confidence_margin)
    avg_conf = sum(p.probability_up for p in preds) / len(preds)

    logger.debug(
        "prediction_summary symbol=%s best_horizon=%s ml_confidence=%.3f",
        symbol, best.horizon, avg_conf,
    )

    if debug:
        from ml.debug_plots import show_predict_plots
        # Use the best horizon's saved threshold for the plot marker; fall back to 0.60
        plot_threshold = best.threshold if best.threshold > 0 else ml_buy_threshold_fallback
        show_predict_plots(
            symbol=symbol,
            preds=preds,
            best_horizon=best.horizon,
            ml_threshold=plot_threshold,
            latest_close=float(bars["close"].iloc[-1]) if len(bars) > 0 else 0.0,
        )

    return ModelPrediction(
        symbol=symbol,
        generated_at=datetime.now(tz=timezone.utc),
        predictions=preds,
        best_horizon=best.horizon,
        ml_confidence=avg_conf,
    )
