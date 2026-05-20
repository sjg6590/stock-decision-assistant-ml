"""Debug visualizations for ML training and prediction. Only used when --debug is passed."""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from sda_types import HorizonPrediction

logger = logging.getLogger(__name__)

try:
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


def _check() -> bool:
    if not _HAS_MATPLOTLIB:
        logger.warning("matplotlib not installed — run: pip install matplotlib")
        return False
    return True


def setup_interactive() -> None:
    """Call once when debug mode starts to enable non-blocking plot display."""
    if _check():
        plt.ion()


def show_train_plots(
    symbol: str,
    total_rows: int,
    train_end: int,
    val_end: int,
    model_payload: dict[str, Any],
    feats: list[str],
    metrics: dict[str, Any],
    per_horizon_acc: dict[str, float],
    promo_cfg: dict[str, Any],
) -> None:
    """Four-panel figure summarizing a training run."""
    if not _check():
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"{symbol}  —  Training Debug", fontsize=14, fontweight="bold")

    # ── Panel 1: Walk-forward split ───────────────────────────────────────────
    ax = axes[0, 0]
    n_train = train_end
    n_val = val_end - train_end
    n_hold = total_rows - val_end
    segments = [("Train", n_train, "#4CAF50"), ("Val", n_val, "#FF9800"), ("Holdout", n_hold, "#EF5350")]
    left = 0
    for label, width, color in segments:
        ax.barh(0, width, left=left, color=color, height=0.5)
        ax.text(left + width / 2, 0, f"{label}\n{width}", ha="center", va="center",
                color="white", fontsize=9, fontweight="bold")
        left += width
    ax.set_xlim(0, total_rows)
    ax.set_yticks([])
    ax.set_xlabel("Rows (bars)")
    ax.set_title("Walk-Forward Split")

    # ── Panel 2: Feature importance (mean across horizon classifiers) ─────────
    ax = axes[0, 1]
    clf_map: dict = model_payload.get("classifiers", {})
    if clf_map:
        importances = np.zeros(len(feats))
        for clf in clf_map.values():
            importances += clf.feature_importances_
        importances /= len(clf_map)
        idx = np.argsort(importances)
        ax.barh(range(len(feats)), importances[idx], color="#42A5F5")
        ax.set_yticks(range(len(feats)))
        ax.set_yticklabels([feats[i] for i in idx], fontsize=9)
        ax.set_xlabel("Mean importance (averaged across horizons)")
    ax.set_title("Feature Importance")

    # ── Panel 3: Per-horizon directional accuracy ─────────────────────────────
    ax = axes[1, 0]
    per_min = float(promo_cfg.get("per_horizon_accuracy_min", 0.50))
    agg_min = float(promo_cfg.get("aggregate_accuracy_min", 0.55))
    hz_names = list(per_horizon_acc.keys())
    acc_vals = [per_horizon_acc[h] for h in hz_names]
    bar_colors = ["#4CAF50" if v >= per_min else "#EF5350" for v in acc_vals]
    ax.bar(hz_names, acc_vals, color=bar_colors)
    ax.axhline(per_min, color="#EF5350", linestyle="--", linewidth=1.5,
               label=f"per-horizon min ({per_min})")
    ax.axhline(agg_min, color="#FF9800", linestyle="--", linewidth=1.5,
               label=f"aggregate min ({agg_min})")
    for x, v in enumerate(acc_vals):
        ax.text(x, v + 0.01, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Directional accuracy")
    ax.set_title(f"Per-Horizon Accuracy  (agg={metrics['aggregate_accuracy']:.3f})")
    ax.legend(fontsize=8)

    # ── Panel 4: Promotion gate results table ─────────────────────────────────
    ax = axes[1, 1]
    ax.axis("off")
    sharpe_min = float(promo_cfg.get("sharpe_min", 0.50))
    dd_max = float(promo_cfg.get("max_drawdown_max", 0.15))
    agg_acc = metrics["aggregate_accuracy"]
    sharpe = metrics["sharpe"]
    dd = metrics["max_drawdown"]

    def _tick(passed: bool) -> str:
        return "✓" if passed else "✗"

    rows = [
        ["Agg accuracy", f"{agg_acc:.3f}", f"≥ {agg_min}", _tick(agg_acc >= agg_min)],
        ["Per-hz accuracy", f"{min(per_horizon_acc.values()):.3f}", f"≥ {per_min}", _tick(min(per_horizon_acc.values()) >= per_min)],
        ["Sharpe ratio", f"{sharpe:.3f}", f"≥ {sharpe_min}", _tick(sharpe >= sharpe_min)],
        ["Max drawdown", f"{dd:.3f}", f"≤ {dd_max}", _tick(dd <= dd_max)],
    ]
    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value", "Threshold", "Pass?"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.8)
    # Colour the Pass? column cells
    for row_idx, row in enumerate(rows, start=1):
        cell = table[row_idx, 3]
        cell.set_facecolor("#C8E6C9" if row[3] == "✓" else "#FFCDD2")

    promoted = not metrics.get("failed_gates", [])
    label = f"PROMOTED ✓" if promoted else "NOT PROMOTED ✗"
    color = "#2E7D32" if promoted else "#C62828"
    ax.text(0.5, 0.08, label, transform=ax.transAxes, ha="center", va="center",
            fontsize=13, fontweight="bold", color=color)
    ax.set_title(f"Promotion Gates  (attempt {metrics.get('attempt', '?')})")

    fig.tight_layout()
    plt.draw()
    plt.pause(0.001)


def show_predict_plots(
    symbol: str,
    preds: list[HorizonPrediction],
    best_horizon: str,
    ml_threshold: float,
    latest_close: float,
) -> None:
    """Two-panel figure showing per-horizon predictions for a symbol."""
    if not _check():
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        f"{symbol}  —  Prediction Debug  (close={latest_close:.2f})",
        fontsize=13, fontweight="bold",
    )

    hz_names = [p.horizon for p in preds]
    prob_ups = [p.probability_up for p in preds]
    exp_rets = [p.expected_return * 100 for p in preds]  # → percent

    # ── Panel 1: Probability up per horizon ──────────────────────────────────
    ax = axes[0]
    bar_colors = ["#4CAF50" if p >= ml_threshold else "#90A4AE" for p in prob_ups]
    bars = ax.bar(hz_names, prob_ups, color=bar_colors)
    ax.axhline(ml_threshold, color="#EF5350", linestyle="--", linewidth=1.5,
               label=f"buy threshold ({ml_threshold})")
    ax.axhline(0.5, color="#9E9E9E", linestyle=":", linewidth=1, label="50% baseline")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("P(up)")
    ax.set_title("Probability Up by Horizon")
    ax.legend(fontsize=8)
    for bar, val in zip(bars, prob_ups):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.2f}", ha="center", va="bottom", fontsize=9)
    for bar, hz in zip(bars, hz_names):
        if hz == best_horizon:
            bar.set_edgecolor("gold")
            bar.set_linewidth(3)
            ax.text(bar.get_x() + bar.get_width() / 2, -0.05, "BEST",
                    ha="center", va="top", fontsize=8, color="#F9A825", fontweight="bold")

    # ── Panel 2: Expected return per horizon ─────────────────────────────────
    ax = axes[1]
    ret_colors = ["#4CAF50" if r > 0 else "#EF5350" for r in exp_rets]
    bars2 = ax.bar(hz_names, exp_rets, color=ret_colors)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_ylabel("Expected return (%)")
    ax.set_title("Expected Return by Horizon")
    for bar, val in zip(bars2, exp_rets):
        offset = 0.1 if val >= 0 else -0.3
        ax.text(bar.get_x() + bar.get_width() / 2, val + offset,
                f"{val:.2f}%", ha="center", va="bottom", fontsize=9)
    for bar, hz in zip(bars2, hz_names):
        if hz == best_horizon:
            bar.set_edgecolor("gold")
            bar.set_linewidth(3)

    fig.tight_layout()
    plt.draw()
    plt.pause(0.001)


def show_attempt_comparison(
    symbol: str,
    all_attempts: list[dict],
    promo_cfg: dict | None = None,
) -> None:
    """Two-panel figure comparing val vs holdout accuracy and Sharpe across all attempts."""
    if not _check() or not all_attempts:
        return

    promo_cfg = promo_cfg or {}
    acc_threshold = float(promo_cfg.get("aggregate_accuracy_min", 0.55))
    sharpe_threshold = float(promo_cfg.get("sharpe_min", 0.50))

    attempts = [a["attempt"] for a in all_attempts]
    val_accs = [a["val_agg_acc"] for a in all_attempts]
    hold_accs = [a["holdout_agg_acc"] for a in all_attempts]
    val_sharpes = [a["val_sharpe"] for a in all_attempts]
    hold_sharpes = [a["sharpe"] for a in all_attempts]
    promoted = [a["promoted"] for a in all_attempts]

    x = np.arange(len(attempts))
    width = 0.35

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"{symbol}  —  Attempt Comparison", fontsize=13, fontweight="bold")

    # ── Panel 1: Accuracy ────────────────────────────────────────────────────
    ax = axes[0]
    bars_val = ax.bar(x - width / 2, val_accs, width, label="val", color="#42A5F5", alpha=0.85)
    bars_hold = ax.bar(x + width / 2, hold_accs, width, label="holdout", color="#26A69A", alpha=0.85)

    # Outline holdout bars red when they failed promotion
    for bar, prom in zip(bars_hold, promoted):
        if not prom:
            bar.set_edgecolor("#EF5350")
            bar.set_linewidth(2)

    ax.axhline(acc_threshold, color="#EF5350", linestyle="--", linewidth=1.5,
               label=f"promo min ({acc_threshold})")
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{a}" for a in attempts])
    ax.set_ylabel("Directional accuracy")
    ax.set_title("Val vs Holdout Accuracy")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    for bar, val in zip(list(bars_val) + list(bars_hold), val_accs + hold_accs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    # ── Panel 2: Sharpe ──────────────────────────────────────────────────────
    ax = axes[1]
    ax.bar(x - width / 2, val_sharpes, width, label="val", color="#42A5F5", alpha=0.85)
    bars_hold_sh = ax.bar(x + width / 2, hold_sharpes, width, label="holdout", color="#26A69A", alpha=0.85)

    for bar, prom in zip(bars_hold_sh, promoted):
        if not prom:
            bar.set_edgecolor("#EF5350")
            bar.set_linewidth(2)

    ax.axhline(sharpe_threshold, color="#EF5350", linestyle="--", linewidth=1.5,
               label=f"promo min ({sharpe_threshold})")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([f"#{a}" for a in attempts])
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("Val vs Holdout Sharpe")
    ax.legend(fontsize=8)

    fig.tight_layout()
    plt.draw()
    plt.pause(0.001)


def wait_for_close() -> None:
    """Block until the user closes all debug plot windows."""
    if _HAS_MATPLOTLIB and plt.get_fignums():
        logger.info("debug plots open — close all windows to exit")
        plt.show(block=True)
