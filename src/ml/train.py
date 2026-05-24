from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any

import numpy as np
import pandas as pd
from xgboost import XGBClassifier, XGBRegressor

from config_loader import train_cold_start
from data.store import DataStore
from ml.evaluate import (
    aggregate_strategy_metrics,
    directional_accuracy_at_threshold,
    strategy_metrics_from_proba,
    tune_threshold_on_val,
)
from ml.features import build_features, feature_columns
from ml.labels import add_multi_horizon_labels
from ml.registry import save_models
from ml.splits import SplitIndices, fixed_tail_split, purged_cv_folds, rolling_holdout_splits

logger = logging.getLogger(__name__)


def _load_retry_cfg(thresholds: dict[str, Any]) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "strategy": "seed_only",
        "selection": "last_attempt",
        "seeds": None,
        "log_attempt_comparison": True,
        "val_score_weights": {"accuracy": 0.5, "sharpe": 0.5},
        "rolling": {"step_bars": 30, "min_train_rows": 600},
        "purged_cv": {"n_splits": 3, "purge_bars": 22, "embargo_bars": 5},
        "ensemble_seeds": 1,
        "stability": {"min_seeds_passing_gates": 0, "metric": "median"},
    }
    cfg = {**defaults, **thresholds.get("train_retry", {})}
    # Deep-merge nested dicts so partial overrides keep defaults for missing keys
    cfg["val_score_weights"] = {**defaults["val_score_weights"], **cfg.get("val_score_weights", {})}
    cfg["rolling"] = {**defaults["rolling"], **cfg.get("rolling", {})}
    cfg["purged_cv"] = {**defaults["purged_cv"], **cfg.get("purged_cv", {})}
    cfg["stability"] = {**defaults["stability"], **cfg.get("stability", {})}
    return cfg


def _horizon_weighted_accuracy(per_horizon_acc: dict[str, float], horizon_weights: dict) -> float:
    """Weighted accuracy across horizons; horizons absent from weights contribute 0."""
    total_w = sum(float(horizon_weights.get(hz, 0.0)) for hz in per_horizon_acc)
    if total_w == 0.0:
        return 0.0
    return sum(float(horizon_weights.get(hz, 0.0)) * acc for hz, acc in per_horizon_acc.items()) / total_w


def _stability_gate_failed(per_seed_pass: list[bool], min_seeds_passing: int) -> bool:
    """Return True (gate failed) when fewer than min_seeds_passing seeds passed basic gates."""
    if min_seeds_passing <= 0:
        return False
    return sum(per_seed_pass) < min_seeds_passing


def _build_splits(
    base: pd.DataFrame,
    retry_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    max_attempts: int,
) -> list[SplitIndices]:
    val_size = int(train_cfg["val_size"])
    holdout_size = int(train_cfg["holdout_size"])
    strategy = retry_cfg.get("strategy", "seed_only")

    if strategy == "rolling_holdout":
        rolling_cfg = retry_cfg["rolling"]
        return rolling_holdout_splits(
            base,
            val_size=val_size,
            holdout_size=holdout_size,
            n_attempts=max_attempts,
            step_bars=int(rolling_cfg["step_bars"]),
            min_train_rows=int(rolling_cfg["min_train_rows"]),
        )

    # seed_only and purged_cv both use a fixed tail holdout; CV folds are
    # computed per-attempt inside the training loop for purged_cv.
    fixed = fixed_tail_split(base, val_size=val_size, holdout_size=holdout_size)
    return [fixed] * max_attempts


def _val_composite_score(val_agg_acc: float, val_sharpe: float, weights: dict[str, float]) -> float:
    w_acc = float(weights.get("accuracy", 0.5))
    w_sharpe = float(weights.get("sharpe", 0.5))
    return w_acc * val_agg_acc + w_sharpe * val_sharpe


def _log_attempt_summary(symbol: str, all_attempts: list[dict[str, Any]]) -> None:
    header = (
        f"{'att':>3}  {'seed':>5}  {'val_acc':>7}  {'h_acc':>7}  {'gap':>6}  "
        f"{'v_shrp':>6}  {'sharpe':>6}  {'max_dd':>6}  {'val_cmp':>7}  promoted"
    )
    rows = [header]
    for a in all_attempts:
        rows.append(
            f"{a['attempt']:>3}  {a['seed']:>5}  {a['val_agg_acc']:>7.3f}  "
            f"{a['holdout_agg_acc']:>7.3f}  {a['val_holdout_acc_gap']:>+6.3f}  "
            f"{a['val_sharpe']:>6.3f}  {a['sharpe']:>6.3f}  {a['max_drawdown']:>6.3f}  "
            f"{a['val_composite']:>7.3f}  {a['promoted']}"
        )
    logger.info("train_attempt_summary symbol=%s total=%d\n%s", symbol, len(all_attempts), "\n".join(rows))



def _build_models(seed: int, early_stopping_rounds: int) -> tuple[XGBClassifier, XGBRegressor]:
    clf = XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed,
    )
    reg = XGBRegressor(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="rmse",
        early_stopping_rounds=early_stopping_rounds,
        random_state=seed,
    )
    return clf, reg


def _failed_basic_gates(
    metrics: dict[str, Any],
    per_horizon_acc: dict[str, float],
    promo_cfg: dict[str, Any],
) -> list[str]:
    """Core 4 promotion gates — used for per-seed stability checks (Phase 4)."""
    failed: list[str] = []
    if metrics["aggregate_accuracy"] < float(promo_cfg["aggregate_accuracy_min"]):
        failed.append("aggregate_accuracy")
    if min(per_horizon_acc.values()) < float(promo_cfg["per_horizon_accuracy_min"]):
        failed.append("per_horizon_accuracy")
    if metrics["sharpe"] < float(promo_cfg["sharpe_min"]):
        failed.append("sharpe")
    if metrics["max_drawdown"] > float(promo_cfg["max_drawdown_max"]):
        failed.append("max_drawdown")
    return failed


def _resolve_training_tier(
    symbol: str,
    bar_count: int,
    thresholds: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], str | None, bool]:
    """Pick full vs cold-start train/promotion config from raw bar count."""
    train_cfg = dict(thresholds["train"])
    promo_cfg = dict(thresholds["promotion"])
    cold_cfg = train_cold_start(thresholds)
    full_min = int(train_cfg["min_rows"])

    if bar_count >= full_min:
        return train_cfg, promo_cfg, None, False

    if not cold_cfg.get("enabled", False):
        logger.info(
            "insufficient_history symbol=%s bars=%d full_min=%d cold_start=disabled",
            symbol, bar_count, full_min,
        )
        raise ValueError(f"Not enough rows for {symbol}: {bar_count} < {full_min}")

    cold_min = int(cold_cfg["min_rows"])
    if bar_count < cold_min:
        logger.info(
            "insufficient_history symbol=%s bars=%d full_min=%d cold_min=%d",
            symbol, bar_count, full_min, cold_min,
        )
        raise ValueError(f"Not enough rows for {symbol}: {bar_count} < {cold_min}")

    logger.info(
        "cold_start_training symbol=%s bars=%d full_min=%d cold_start=True",
        symbol, bar_count, full_min,
    )
    train_cfg["val_size"] = int(cold_cfg["val_size"])
    train_cfg["holdout_size"] = int(cold_cfg["holdout_size"])
    train_cfg["min_rows"] = cold_min
    tag = str(cold_cfg.get("tag", "cold_start"))
    promo_cfg = {**promo_cfg, **cold_cfg.get("promotion", {})}
    return train_cfg, promo_cfg, tag, True


def _failed_promotion_gates(
    metrics: dict[str, Any],
    per_horizon_acc: dict[str, float],
    promo_cfg: dict[str, Any],
) -> list[str]:
    failed = _failed_basic_gates(metrics, per_horizon_acc, promo_cfg)

    # Phase 4.3: overfit detector — fail if val_sharpe - holdout_sharpe exceeds threshold
    max_gap = float(promo_cfg.get("max_val_holdout_sharpe_gap", 0.0))
    if max_gap > 0.0:
        val_sharpe = metrics.get("val_sharpe")
        if val_sharpe is not None and val_sharpe - metrics["sharpe"] > max_gap:
            failed.append("val_holdout_sharpe_gap")

    # Phase 4.4: horizon-weighted accuracy gate
    horizon_weights = promo_cfg.get("horizon_weights") or {}
    if horizon_weights:
        weighted_acc = _horizon_weighted_accuracy(per_horizon_acc, horizon_weights)
        weighted_min = float(promo_cfg.get("weighted_accuracy_min", 0.52))
        if weighted_acc < weighted_min:
            failed.append("weighted_accuracy")

    return failed


def train_with_retries(
    store: DataStore,
    symbol: str,
    bars: pd.DataFrame,
    thresholds: dict[str, Any],
    model_dir,
    debug: bool = False,
    spy_frame: pd.DataFrame | None = None,
) -> dict[str, Any]:
    horizons = thresholds["horizons"]
    retry_cfg = _load_retry_cfg(thresholds)
    train_cfg, promo_cfg, model_tag, is_cold_start = _resolve_training_tier(
        symbol, len(bars), thresholds
    )
    max_attempts = int(train_cfg["max_retrain_attempts"])
    early_stopping_rounds = int(train_cfg["early_stopping_rounds"])
    min_rows = int(train_cfg["min_rows"])
    selection = retry_cfg["selection"]
    strategy = retry_cfg["strategy"]

    # Phase 4: multi-seed stability gate config
    ensemble_seeds_count = int(retry_cfg.get("ensemble_seeds", 1))
    stability_cfg = retry_cfg["stability"]
    min_seeds_passing = int(stability_cfg.get("min_seeds_passing_gates", 0))
    _agg_fn = np.median if stability_cfg.get("metric", "median") == "median" else np.mean
    # Multi-seed only applies to seed_only/rolling_holdout (purged_cv already has fold variance)
    use_multi_seed = ensemble_seeds_count > 1 and strategy != "purged_cv"

    # Phase 4: multi-window gate — must run all attempts when active
    min_passing_windows = int(promo_cfg.get("min_passing_windows", 0))
    force_all_attempts = selection == "best_val_score" or (
        min_passing_windows > 0 and strategy == "rolling_holdout"
    )

    min_return_threshold = float(train_cfg.get("label_min_return_threshold", 0.0))
    logger.debug("label_min_return_threshold symbol=%s threshold=%.4f", symbol, min_return_threshold)
    base = add_multi_horizon_labels(
        build_features(bars, spy_frame), horizons, min_return_threshold=min_return_threshold
    ).dropna().reset_index(drop=True)
    if len(base) < min_rows:
        logger.info(
            "insufficient_history symbol=%s feature_rows=%d min_rows=%d cold_start=%s",
            symbol, len(base), min_rows, is_cold_start,
        )
        raise ValueError(f"Not enough rows for {symbol}: {len(base)} < {min_rows}")
    feats = feature_columns()

    logger.debug(
        "train_data symbol=%s total_rows=%d features=%d horizons=%s",
        symbol, len(base), len(feats), list(horizons.keys()),
    )

    configured_seeds: list[int] | None = retry_cfg.get("seeds")
    has_datetime = "datetime" in base.columns
    bars_through_date = str(base["datetime"].iloc[-1].date()) if has_datetime else None

    splits = _build_splits(base, retry_cfg, train_cfg, max_attempts)

    all_attempts: list[dict[str, Any]] = []
    all_attempt_results: list[dict[str, Any]] = []

    for attempt_idx, split in enumerate(splits):
        attempt = attempt_idx + 1
        seed = (
            int(configured_seeds[attempt_idx])
            if configured_seeds and attempt_idx < len(configured_seeds)
            else 42 + attempt
        )

        train_df = base.iloc[:split.train_end]
        val_df = base.iloc[split.train_end:split.val_end]
        holdout_df = base.iloc[split.val_end:split.holdout_end]
        if train_df.empty or val_df.empty or holdout_df.empty:
            continue

        holdout_start_date = str(base["datetime"].iloc[split.val_end].date()) if has_datetime else "?"
        holdout_end_date = str(base["datetime"].iloc[split.holdout_end - 1].date()) if has_datetime else "?"

        logger.debug(
            "split symbol=%s attempt=%d seed=%d strategy=%s train_end=%d val_end=%d "
            "train=%d val=%d holdout=%d holdout_start=%s holdout_end=%s",
            symbol, attempt, seed, strategy,
            split.train_end, split.val_end,
            len(train_df), len(val_df), len(holdout_df),
            holdout_start_date, holdout_end_date,
        )

        x_val = val_df[feats]
        x_holdout = holdout_df[feats]

        # purged_cv: generate folds over the combined train+val region
        cv_folds = []
        train_val_df = None
        if strategy == "purged_cv":
            pv_cfg = retry_cfg["purged_cv"]
            train_val_df = base.iloc[:split.val_end]
            cv_folds = purged_cv_folds(
                n_rows_train_val=len(train_val_df),
                n_splits=int(pv_cfg["n_splits"]),
                purge_bars=int(pv_cfg["purge_bars"]),
                embargo_bars=int(pv_cfg.get("embargo_bars", 0)),
            )
            if not cv_folds:
                logger.warning(
                    "purged_cv: no folds for symbol=%s attempt=%d train_val_rows=%d, skipping",
                    symbol, attempt, len(train_val_df),
                )
                continue
            logger.debug(
                "purged_cv symbol=%s attempt=%d seed=%d n_folds=%d train_val_rows=%d",
                symbol, attempt, seed, len(cv_folds), len(train_val_df),
            )

        model_payload: dict[str, Any] = {
            "classifiers": {},
            "regressors": {},
            "horizons": horizons,
            "thresholds": {},
            "feature_columns": feats,
        }
        per_horizon_acc: dict[str, float] = {}
        val_per_horizon_acc: dict[str, float] = {}
        per_horizon_strategy: list[tuple[float, float]] = []
        val_per_horizon_strategy: list[tuple[float, float]] = []
        label_balance_per_hz: dict[str, float] = {}

        # Phase 4: per-seed horizon-level accumulators for stability gate
        per_seed_horizon_accs: list[dict[str, float]] = [{} for _ in range(ensemble_seeds_count)]
        per_seed_horizon_strats: list[list[tuple[float, float]]] = [[] for _ in range(ensemble_seeds_count)]

        for hz, horizon_bars in horizons.items():
            up_col = f"up_{hz}"
            ret_col = f"ret_{hz}"
            y_holdout = holdout_df[up_col]
            y_holdout_ret = holdout_df[ret_col]

            if strategy == "purged_cv":
                # Run K purged CV folds to obtain stable val metrics.
                # Final model is fit on full train+val using the last fold for
                # early stopping (never touches holdout during selection).
                fold_accs: list[float] = []
                fold_strat: list[tuple[float, float]] = []
                final_threshold = 0.5

                for k_idx, fold in enumerate(cv_folds):
                    xf_tr = train_val_df.iloc[:fold.train_end][feats]
                    yf_tr = train_val_df.iloc[:fold.train_end][up_col]
                    yf_tr_ret = train_val_df.iloc[:fold.train_end][ret_col]
                    xf_val = train_val_df.iloc[fold.val_start:fold.val_end][feats]
                    yf_val = train_val_df.iloc[fold.val_start:fold.val_end][up_col]
                    yf_val_ret = train_val_df.iloc[fold.val_start:fold.val_end][ret_col]

                    f_clf, f_reg = _build_models(seed=seed, early_stopping_rounds=early_stopping_rounds)
                    f_clf.fit(xf_tr, yf_tr, eval_set=[(xf_val, yf_val)], verbose=False)
                    f_reg.fit(xf_tr, yf_tr_ret, eval_set=[(xf_val, yf_val_ret)], verbose=False)

                    proba_fold = f_clf.predict_proba(xf_val)[:, 1]
                    fold_threshold = tune_threshold_on_val(yf_val, proba_fold)
                    final_threshold = fold_threshold  # last fold threshold used on holdout

                    fold_acc = directional_accuracy_at_threshold(yf_val, proba_fold, fold_threshold)
                    sh_dd = strategy_metrics_from_proba(proba_fold, yf_val_ret, int(horizon_bars), fold_threshold)
                    fold_accs.append(fold_acc)
                    fold_strat.append(sh_dd)

                    logger.debug(
                        "cv_fold symbol=%s attempt=%d hz=%s fold=%d/%d "
                        "train=[0,%d) val=[%d,%d) threshold=%.2f acc=%.3f sharpe=%.3f",
                        symbol, attempt, hz, k_idx + 1, len(cv_folds),
                        fold.train_end, fold.val_start, fold.val_end,
                        fold_threshold, fold_acc, sh_dd[0],
                    )

                val_per_horizon_acc[hz] = float(np.mean(fold_accs))
                val_per_horizon_strategy.append((
                    float(np.mean([s[0] for s in fold_strat])),
                    float(np.max([s[1] for s in fold_strat])),
                ))

                # Fit final model on full train+val; last fold is the early-stopping eval set
                last_fold = cv_folds[-1]
                x_es = train_val_df.iloc[last_fold.val_start:last_fold.val_end][feats]
                y_es = train_val_df.iloc[last_fold.val_start:last_fold.val_end][up_col]
                y_es_ret = train_val_df.iloc[last_fold.val_start:last_fold.val_end][ret_col]

                # Train on rows before last_fold.val_start so eval set is truly held out.
                x_train_final = train_val_df.iloc[:last_fold.val_start][feats]
                y_train_final = train_val_df.iloc[:last_fold.val_start][up_col]
                y_train_final_ret = train_val_df.iloc[:last_fold.val_start][ret_col]
                logger.debug(
                    "purged_cv_final_fit symbol=%s attempt=%d hz=%s train=[0,%d) eval=[%d,%d)",
                    symbol, attempt, hz, last_fold.val_start, last_fold.val_start, last_fold.val_end,
                )
                final_clf, final_reg = _build_models(seed=seed, early_stopping_rounds=early_stopping_rounds)
                final_clf.fit(
                    x_train_final, y_train_final,
                    eval_set=[(x_es, y_es)], verbose=False,
                )
                final_reg.fit(
                    x_train_final, y_train_final_ret,
                    eval_set=[(x_es, y_es_ret)], verbose=False,
                )

                proba_holdout = final_clf.predict_proba(x_holdout)[:, 1]
                acc = directional_accuracy_at_threshold(y_holdout, proba_holdout, final_threshold)
                per_horizon_acc[hz] = acc
                per_horizon_strategy.append(
                    strategy_metrics_from_proba(proba_holdout, y_holdout_ret, int(horizon_bars), final_threshold)
                )

                label_balance_per_hz[hz] = float(train_val_df[up_col].mean())
                model_payload["thresholds"][hz] = final_threshold
                model_payload["classifiers"][hz] = final_clf
                model_payload["regressors"][hz] = final_reg

                logger.debug(
                    "horizon_train symbol=%s attempt=%d seed=%d horizon=%s strategy=purged_cv "
                    "n_folds=%d mean_val_acc=%.3f holdout_acc=%.3f threshold=%.2f label_balance=%.3f",
                    symbol, attempt, seed, hz, len(cv_folds),
                    val_per_horizon_acc[hz], acc, final_threshold, label_balance_per_hz[hz],
                )

            else:
                # seed_only / rolling_holdout: single train/val/holdout split
                x_train = train_df[feats]
                y_train = train_df[up_col]
                y_train_ret = train_df[ret_col]
                y_val = val_df[up_col]
                y_val_ret = val_df[ret_col]

                if use_multi_seed:
                    # Phase 4.1: train ensemble_seeds classifiers; aggregate probabilities.
                    # Per-seed holdout metrics are tracked separately for the stability gate.
                    # Primary seed's models are saved in the payload (Phase 6 adds full ensemble).
                    seed_clfs: list[XGBClassifier] = []
                    seed_val_probas_list: list[np.ndarray] = []
                    seed_holdout_probas_list: list[np.ndarray] = []
                    primary_reg: XGBRegressor | None = None

                    for s_idx in range(ensemble_seeds_count):
                        s_clf, s_reg = _build_models(seed=seed + s_idx, early_stopping_rounds=early_stopping_rounds)
                        s_clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
                        if s_idx == 0:
                            s_reg.fit(x_train, y_train_ret, eval_set=[(x_val, y_val_ret)], verbose=False)
                            primary_reg = s_reg
                        seed_clfs.append(s_clf)
                        seed_val_probas_list.append(s_clf.predict_proba(x_val)[:, 1])
                        seed_holdout_probas_list.append(s_clf.predict_proba(x_holdout)[:, 1])
                        # Track per-seed holdout metrics for the stability gate
                        s_thr = tune_threshold_on_val(y_val, seed_val_probas_list[-1])
                        s_acc = directional_accuracy_at_threshold(
                            y_holdout, seed_holdout_probas_list[-1], s_thr
                        )
                        s_strat = strategy_metrics_from_proba(
                            seed_holdout_probas_list[-1], y_holdout_ret, int(horizon_bars), s_thr
                        )
                        per_seed_horizon_accs[s_idx][hz] = s_acc
                        per_seed_horizon_strats[s_idx].append(s_strat)

                    proba_val = _agg_fn(np.stack(seed_val_probas_list, axis=0), axis=0)
                    proba_holdout = _agg_fn(np.stack(seed_holdout_probas_list, axis=0), axis=0)
                    clf, reg = seed_clfs[0], primary_reg
                else:
                    clf, reg = _build_models(seed=seed, early_stopping_rounds=early_stopping_rounds)
                    clf.fit(x_train, y_train, eval_set=[(x_val, y_val)], verbose=False)
                    reg.fit(x_train, y_train_ret, eval_set=[(x_val, y_val_ret)], verbose=False)
                    proba_val = clf.predict_proba(x_val)[:, 1]
                    proba_holdout = clf.predict_proba(x_holdout)[:, 1]

                # Tune threshold on val (keeps holdout blind during selection)
                threshold = tune_threshold_on_val(y_val, proba_val)

                val_acc = directional_accuracy_at_threshold(y_val, proba_val, threshold)
                val_per_horizon_acc[hz] = val_acc
                val_per_horizon_strategy.append(
                    strategy_metrics_from_proba(proba_val, y_val_ret, int(horizon_bars), threshold)
                )

                # Apply the same threshold on holdout for gate evaluation
                acc = directional_accuracy_at_threshold(y_holdout, proba_holdout, threshold)
                per_horizon_acc[hz] = acc
                per_horizon_strategy.append(
                    strategy_metrics_from_proba(proba_holdout, y_holdout_ret, int(horizon_bars), threshold)
                )

                label_balance_per_hz[hz] = float(y_train.mean())
                model_payload["thresholds"][hz] = threshold
                model_payload["classifiers"][hz] = clf
                model_payload["regressors"][hz] = reg

                logger.debug(
                    "horizon_train symbol=%s attempt=%d seed=%d horizon=%s threshold=%.2f "
                    "val_acc=%.3f holdout_acc=%.3f best_iteration=%s label_balance=%.3f",
                    symbol, attempt, seed, hz, threshold, val_acc, acc,
                    getattr(clf, "best_iteration", "?"), label_balance_per_hz[hz],
                )

        agg_acc = float(np.mean(list(per_horizon_acc.values())))
        val_agg_acc = float(np.mean(list(val_per_horizon_acc.values())))
        sharpe, max_dd = aggregate_strategy_metrics(per_horizon_strategy)
        val_sharpe, _ = aggregate_strategy_metrics(val_per_horizon_strategy)
        val_composite = _val_composite_score(val_agg_acc, val_sharpe, retry_cfg["val_score_weights"])

        metrics = {
            "attempt": attempt,
            "seed": seed,
            "strategy": strategy,
            "cold_start": is_cold_start,
            "tag": model_tag,
            "selection_reason": selection,
            "split_boundaries": {
                "train_end": split.train_end,
                "val_end": split.val_end,
                "holdout_end": split.holdout_end,
            },
            "bars_through_date": bars_through_date,
            "aggregate_accuracy": agg_acc,
            "val_aggregate_accuracy": val_agg_acc,
            "val_holdout_acc_gap": val_agg_acc - agg_acc,
            "per_horizon_accuracy": per_horizon_acc,
            "val_per_horizon_accuracy": val_per_horizon_acc,
            "per_horizon_thresholds": model_payload["thresholds"],
            "label_balance": label_balance_per_hz,
            "sharpe": sharpe,
            "val_sharpe": val_sharpe,
            "val_composite": val_composite,
            "max_drawdown": max_dd,
        }
        failed_gates = _failed_promotion_gates(metrics, per_horizon_acc, promo_cfg)

        # Phase 4.1: stability gate — require min_seeds_passing individual seeds to pass basic gates
        if use_multi_seed and min_seeds_passing > 0:
            seed_pass_list = []
            for s_idx in range(ensemble_seeds_count):
                s_agg = float(np.mean(list(per_seed_horizon_accs[s_idx].values())))
                s_sh, s_dd = aggregate_strategy_metrics(per_seed_horizon_strats[s_idx])
                s_m = {"aggregate_accuracy": s_agg, "sharpe": s_sh, "max_drawdown": s_dd}
                seed_pass_list.append(not _failed_basic_gates(s_m, per_seed_horizon_accs[s_idx], promo_cfg))
            seeds_passing = sum(seed_pass_list)
            if _stability_gate_failed(seed_pass_list, min_seeds_passing):
                failed_gates.append("stability")
                logger.debug(
                    "stability_gate symbol=%s attempt=%d seeds_passing=%d/%d min=%d FAILED",
                    symbol, attempt, seeds_passing, ensemble_seeds_count, min_seeds_passing,
                )
            else:
                logger.debug(
                    "stability_gate symbol=%s attempt=%d seeds_passing=%d/%d PASSED",
                    symbol, attempt, seeds_passing, ensemble_seeds_count,
                )

        pass_gate = not failed_gates

        logger.debug(
            "attempt_result symbol=%s attempt=%d seed=%d agg_acc=%.3f val_acc=%.3f "
            "val_sharpe=%.3f sharpe=%.3f max_dd=%.3f val_composite=%.3f promoted=%s failed_gates=%s",
            symbol, attempt, seed, agg_acc, val_agg_acc, val_sharpe, sharpe, max_dd,
            val_composite, pass_gate, failed_gates,
        )

        all_attempts.append({
            "attempt": attempt,
            "seed": seed,
            "val_agg_acc": val_agg_acc,
            "holdout_agg_acc": agg_acc,
            "val_holdout_acc_gap": val_agg_acc - agg_acc,
            "val_sharpe": val_sharpe,
            "sharpe": sharpe,
            "max_drawdown": max_dd,
            "val_composite": val_composite,
            "promoted": pass_gate,
        })

        version = datetime.now(tz=timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        artifact = save_models(model_dir, symbol, version, model_payload)
        store.write_model_registry(
            symbol=symbol, version=version, metrics=metrics, promoted=pass_gate, tag=model_tag
        )

        if debug:
            metrics["failed_gates"] = failed_gates
            from ml.debug_plots import show_train_plots
            show_train_plots(
                symbol=symbol,
                total_rows=split.holdout_end,
                train_end=split.train_end,
                val_end=split.val_end,
                model_payload=model_payload,
                feats=feats,
                metrics=metrics,
                per_horizon_acc=per_horizon_acc,
                promo_cfg=promo_cfg,
            )

        attempt_result = {
            "symbol": symbol,
            "version": version,
            "metrics": metrics,
            "artifact": str(artifact),
            "promoted": pass_gate,
            "failed_gates": failed_gates,
            "val_composite": val_composite,
        }
        all_attempt_results.append(attempt_result)

        # last_attempt: return on first passing attempt unless force_all_attempts is active
        if not force_all_attempts and selection == "last_attempt" and pass_gate:
            break

    # Phase 4.2: multi-window gate — require at least min_passing_windows rolling windows to pass
    if min_passing_windows > 0 and strategy == "rolling_holdout":
        windows_passed = sum(1 for r in all_attempt_results if r["promoted"])
        if windows_passed < min_passing_windows:
            logger.info(
                "multi_window_gate symbol=%s windows_passed=%d required=%d → promotion blocked",
                symbol, windows_passed, min_passing_windows,
            )
            for r in all_attempt_results:
                if r["promoted"]:
                    r["promoted"] = False
                    r["failed_gates"] = list(r.get("failed_gates", [])) + ["min_passing_windows"]

    if retry_cfg.get("log_attempt_comparison", True) and all_attempts:
        _log_attempt_summary(symbol, all_attempts)

    if debug and all_attempts:
        from ml.debug_plots import show_attempt_comparison
        show_attempt_comparison(symbol, all_attempts, promo_cfg)

    if not all_attempt_results:
        return {"symbol": symbol, "promoted": False}

    if selection == "best_val_score":
        # Select by val composite; holdout gates already computed for each attempt
        best = max(all_attempt_results, key=lambda r: r["val_composite"])
        best["metrics"]["selection_reason"] = "best_val_score"
        return best

    # last_attempt: if any attempt passed we broke early; otherwise return last
    return all_attempt_results[-1]
