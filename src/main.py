from __future__ import annotations

import argparse
import logging
import sys
from datetime import date, datetime, timezone
from typing import Any

import pandas as pd
from apscheduler.schedulers.blocking import BlockingScheduler

from config_loader import load_thresholds, load_yaml, market_reference_symbol
from data.ingest import ingest_news_for_symbol, ingest_symbol_bars
from data.store import DataStore
from ml.features import model_features_stale
from ml.predict import predict_symbol
from ml.registry import load_models
from ml.train import train_with_retries
from notify.base import format_alert
from notify.email import send_email_alert
from notify.push import send_ntfy_alert, send_telegram_alert
from notify.sms import send_sms_alert
from schwab_client.auth import build_client
from schwab_client.market_data import fetch_live_price
from sentiment.analyze import analyze_symbol_sentiment
from logging_config import setup_logging
from settings import Settings, ensure_runtime_dirs
from signals.fuse import fuse_signals

logger = logging.getLogger("stock-assistant")


def _load_symbols(settings: Settings) -> list[str]:
    cfg = load_yaml(settings.watchlist_path)
    symbols = cfg.get("symbols", [])
    if not symbols:
        raise ValueError("watchlist.yaml does not define any symbols")
    return [str(s).upper() for s in symbols]


def _load_thresholds(settings: Settings) -> dict[str, Any]:
    return load_thresholds(settings.thresholds_path)


def _load_reference_bars(
    store: DataStore,
    client: Any,
    thresholds: dict[str, Any],
) -> pd.DataFrame:
    ref_symbol = market_reference_symbol(thresholds)
    bars = store.load_bars(ref_symbol)
    if bars.empty:
        bars = ingest_symbol_bars(store, client, ref_symbol)
    return bars


def _warn_stale_bars(symbol: str, bars, max_age_days: int) -> None:
    if "datetime" in bars.columns and not bars.empty:
        last_bar_date = bars["datetime"].iloc[-1]
        try:
            last_bar_date = last_bar_date.date() if hasattr(last_bar_date, "date") else last_bar_date
        except Exception:
            return
        age_days = (date.today() - last_bar_date).days
        if age_days > max_age_days:
            logger.warning(
                "stale_bars symbol=%s last_bar=%s age_days=%d",
                symbol, last_bar_date, age_days,
            )


def retrain_symbol(settings: Settings, store: DataStore, symbol: str, debug: bool = False) -> dict:
    thresholds = _load_thresholds(settings)
    client = build_client(settings)
    spy_bars = _load_reference_bars(store, client, thresholds)
    bars = ingest_symbol_bars(store, client, symbol)
    _warn_stale_bars(symbol, bars, settings.max_bar_age_days)
    result = train_with_retries(
        store, symbol, bars, thresholds, settings.model_dir, debug=debug, spy_frame=spy_bars
    )
    logger.info("retrain result symbol=%s promoted=%s", symbol, result.get("promoted"))
    return result


def _apply_train_result(symbol: str, train_result: dict[str, Any]) -> tuple[str, dict] | None:
    version = train_result.get("version")
    if not version:
        logger.warning("training produced no model for symbol=%s", symbol)
        return None
    if not train_result.get("promoted"):
        failed = train_result.get("failed_gates", [])
        metrics = train_result.get("metrics", {})
        logger.warning(
            "training failed promotion gates for symbol=%s failed=%s metrics=%s; using latest model",
            symbol,
            failed,
            metrics,
        )
    return version, train_result.get("metrics", {})


def _train_symbol(
    store: DataStore,
    symbol: str,
    bars: pd.DataFrame,
    thresholds: dict[str, Any],
    model_dir,
    spy_bars: pd.DataFrame,
    debug: bool,
) -> tuple[str, dict] | None:
    try:
        train_result = train_with_retries(
            store, symbol, bars, thresholds, model_dir, debug=debug, spy_frame=spy_bars
        )
    except ValueError as exc:
        if "Not enough rows" in str(exc):
            logger.warning("skipping symbol=%s insufficient history: %s", symbol, exc)
            return None
        raise
    return _apply_train_result(symbol, train_result)


def run_symbol_cycle(
    settings: Settings,
    store: DataStore,
    symbol: str,
    thresholds: dict[str, Any],
    spy_bars: pd.DataFrame,
    debug: bool = False,
) -> None:
    logger.info("processing symbol=%s", symbol)
    signal_cfg = thresholds["signal"]
    client = build_client(settings)
    bars = ingest_symbol_bars(store, client, symbol)
    if bars.empty:
        logger.warning("no bars for symbol=%s", symbol)
        return

    _warn_stale_bars(symbol, bars, settings.max_bar_age_days)

    logger.debug(
        "bars_loaded symbol=%s rows=%d from=%s to=%s",
        symbol, len(bars),
        bars["datetime"].iloc[0] if "datetime" in bars.columns else "?",
        bars["datetime"].iloc[-1] if "datetime" in bars.columns else "?",
    )

    latest = store.get_latest_promoted_model(symbol)
    if not latest:
        existing = store.get_latest_model(symbol)
        if existing:
            version, metrics, promoted = existing
            if not promoted:
                logger.info("using existing non-promoted model for symbol=%s version=%s", symbol, version)
            latest = (version, metrics)
        else:
            latest = _train_symbol(
                store, symbol, bars, thresholds, settings.model_dir, spy_bars, debug
            )
            if not latest:
                return
    version, _metrics = latest
    if model_features_stale(load_models(settings.model_dir, symbol, version)):
        logger.info(
            "feature_set_changed symbol=%s version=%s retraining with current features",
            symbol,
            version,
        )
        latest = _train_symbol(
            store, symbol, bars, thresholds, settings.model_dir, spy_bars, debug
        )
        if not latest:
            return
        version, _metrics = latest
    logger.debug("model_version symbol=%s version=%s", symbol, version)

    pred = predict_symbol(
        settings.model_dir,
        symbol,
        version,
        bars,
        thresholds["horizons"],
        debug=debug,
        spy_frame=spy_bars,
        signal_config=signal_cfg,
    )
    articles = ingest_news_for_symbol(
        store,
        symbol,
        settings.newsapi_key,
        settings.alphavantage_key,
        newsapi_delay_seconds=settings.newsapi_delay_seconds,
        anthropic_api_key=settings.anthropic_api_key,
        anthropic_web_search_enabled=settings.anthropic_web_search_enabled,
        anthropic_model=settings.anthropic_model,
        anthropic_web_search_max_uses=settings.anthropic_web_search_max_uses,
        anthropic_delay_seconds=settings.anthropic_delay_seconds,
    )
    if len(articles) < 3:
        logger.warning(
            "sparse news symbol=%s article_count=%d (sentiment confidence may be low)",
            symbol,
            len(articles),
        )

    latest_close = float(bars["close"].iloc[-1])
    last_bar_date: date | None = None
    if "datetime" in bars.columns and not bars.empty:
        last_bar_ts = bars["datetime"].iloc[-1]
        try:
            last_bar_date = last_bar_ts.date() if hasattr(last_bar_ts, "date") else last_bar_ts
        except Exception:
            last_bar_date = None

    live_price = fetch_live_price(client, symbol)
    market_prices = f"last daily close ${latest_close:.2f}"
    if last_bar_date is not None:
        market_prices += f" ({last_bar_date.isoformat()})"
    if live_price is not None:
        market_prices += f", live Schwab quote ${live_price:.2f}"
    else:
        market_prices += ", live quote unavailable"

    hint = (
        f"ML best_horizon={pred.best_horizon}, ml_confidence={pred.ml_confidence:.2f}. "
        f"Authoritative market prices: {market_prices}."
    )
    window_key, sentiment = analyze_symbol_sentiment(settings, symbol, articles, hint)
    store.save_sentiment(symbol, window_key, sentiment)
    logger.debug(
        "prices symbol=%s close=%.4f last_bar=%s live=%s",
        symbol,
        latest_close,
        last_bar_date,
        f"{live_price:.4f}" if live_price is not None else "unavailable",
    )

    fused = fuse_signals(
        symbol=symbol,
        prediction=pred,
        sentiment=sentiment,
        ml_threshold=float(signal_cfg["ml_buy_threshold"]),
        sentiment_threshold=float(signal_cfg["sentiment_confidence_min"]),
        latest_close=latest_close,
        last_bar_date=last_bar_date,
        live_price=live_price,
    )
    store.save_signal(fused)
    if not fused.should_notify:
        logger.info("no alert symbol=%s reason=%s", symbol, fused.reason)
        return
    if store.should_suppress_alert(symbol, fused.signal_type, fused.confidence, settings.alert_dedup_hours):
        logger.info("suppressed duplicate alert symbol=%s", symbol)
        return

    msg = format_alert(symbol, fused, sentiment)
    if settings.dry_run:
        logger.info("DRY_RUN alert\n%s\n%s", msg.subject, msg.body)
    else:
        send_email_alert(settings, msg)
        send_sms_alert(settings, msg)
        send_telegram_alert(settings, msg)
        send_ntfy_alert(settings, msg)
    store.mark_alert_sent(symbol, fused.signal_type, fused.confidence)


def run_cycle(settings: Settings, symbols: list[str], debug: bool = False) -> None:
    store = DataStore(settings.sqlite_path, settings.parquet_dir)
    thresholds = _load_thresholds(settings)
    client = build_client(settings)
    spy_bars = _load_reference_bars(store, client, thresholds)
    for symbol in symbols:
        try:
            run_symbol_cycle(settings, store, symbol, thresholds, spy_bars, debug=debug)
        except Exception as exc:
            logger.exception("cycle failed symbol=%s err=%s", symbol, exc)


def run_daemon(settings: Settings, symbols: list[str]) -> None:
    scheduler = BlockingScheduler(timezone="UTC")
    scheduler.add_job(lambda: run_cycle(settings, symbols), "cron", minute="*/15")
    scheduler.add_job(lambda: [retrain_symbol(settings, DataStore(settings.sqlite_path, settings.parquet_dir), s) for s in symbols], "cron", hour=22, minute=30)
    scheduler.add_job(lambda: run_cycle(settings, symbols), "cron", hour="*/6", minute=5)
    logger.info("scheduler started at %s for symbols=%s", datetime.now(tz=timezone.utc).isoformat(), symbols)
    scheduler.start()


def run_backtest(settings: Settings, symbol: str, debug: bool = False) -> None:
    store = DataStore(settings.sqlite_path, settings.parquet_dir)
    thresholds = _load_thresholds(settings)
    client = build_client(settings)
    spy_bars = _load_reference_bars(store, client, thresholds)
    bars = store.load_bars(symbol)
    if bars.empty:
        bars = ingest_symbol_bars(store, client, symbol)
    result = train_with_retries(
        store, symbol, bars, thresholds, settings.model_dir, debug=debug, spy_frame=spy_bars
    )
    print(result)


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stock Decision Assistant")
    parser.add_argument(
        "--debug",
        action="store_true",
        default=False,
        help="Enable DEBUG logging and ML visualizations (requires matplotlib)",
    )
    parser.add_argument(
        "--debug-plots",
        action="store_true",
        default=False,
        help="Enable ML visualizations without enabling DEBUG log level",
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    sub.add_parser("run", help="Run one full cycle now")

    daemon = sub.add_parser("daemon", help="Run daemon scheduler")
    daemon.add_argument("--symbols", nargs="*", default=None, help="Override watchlist symbols")

    retrain = sub.add_parser("retrain", help="Retrain one symbol")
    retrain.add_argument("symbol", type=str)

    backtest = sub.add_parser("backtest", help="Train/evaluate one symbol with current settings")
    backtest.add_argument("symbol", type=str)

    sub.add_parser("schwab-login", help="Run Schwab OAuth login flow")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    debug: bool = args.debug
    plots: bool = debug or args.debug_plots
    settings = Settings()
    level = logging.DEBUG if debug else logging.INFO
    setup_logging(colors=None if settings.log_colors else False, level=level)
    ensure_runtime_dirs(settings)
    symbols = _load_symbols(settings)
    store = DataStore(settings.sqlite_path, settings.parquet_dir)

    if plots:
        from ml.debug_plots import setup_interactive
        setup_interactive()
        if debug:
            logger.debug("debug mode enabled — log level=DEBUG, ML plots=on")
        else:
            logger.info("debug-plots mode enabled — ML visualizations=on, log level=INFO")

    if args.cmd == "run":
        run_cycle(settings, symbols, debug=plots)
        if plots:
            from ml.debug_plots import wait_for_close
            wait_for_close()
        return 0
    if args.cmd == "daemon":
        if args.symbols:
            symbols = [s.upper() for s in args.symbols]
        run_daemon(settings, symbols)
        return 0
    if args.cmd == "retrain":
        retrain_symbol(settings, store, args.symbol.upper(), debug=plots)
        if plots:
            from ml.debug_plots import wait_for_close
            wait_for_close()
        return 0
    if args.cmd == "backtest":
        run_backtest(settings, args.symbol.upper(), debug=plots)
        if plots:
            from ml.debug_plots import wait_for_close
            wait_for_close()
        return 0
    if args.cmd == "schwab-login":
        build_client(settings)
        print("Schwab login flow completed. Token saved.")
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
