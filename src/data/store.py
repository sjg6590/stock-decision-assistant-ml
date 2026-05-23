from __future__ import annotations

import json
import sqlite3
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from sda_types import FusedSignal, SentimentResult


def _utc_now_iso() -> str:
    return datetime.now(tz=timezone.utc).isoformat()


class DataStore:
    def __init__(self, sqlite_path: Path, parquet_dir: Path) -> None:
        self.sqlite_path = sqlite_path
        self.parquet_dir = parquet_dir
        self.parquet_dir.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.sqlite_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS model_registry (
                    symbol TEXT NOT NULL,
                    version TEXT NOT NULL,
                    metrics_json TEXT NOT NULL,
                    promoted INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    bars_through_date TEXT,
                    tag TEXT,
                    PRIMARY KEY (symbol, version)
                )
                """
            )
            # Migration: add bars_through_date to pre-existing databases.
            try:
                conn.execute("ALTER TABLE model_registry ADD COLUMN bars_through_date TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
            try:
                conn.execute("ALTER TABLE model_registry ADD COLUMN tag TEXT")
            except sqlite3.OperationalError:
                pass  # column already exists
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sentiment_cache (
                    symbol TEXT NOT NULL,
                    window_key TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY (symbol, window_key)
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS signals (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    reason TEXT NOT NULL,
                    sell_guide TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS alerts_sent (
                    symbol TEXT NOT NULL,
                    signal_type TEXT NOT NULL,
                    confidence REAL NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS news_articles (
                    article_hash TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    source TEXT NOT NULL,
                    title TEXT NOT NULL,
                    url TEXT NOT NULL,
                    published_at TEXT NOT NULL,
                    content_json TEXT NOT NULL
                )
                """
            )

    def save_bars(self, symbol: str, frame: pd.DataFrame) -> None:
        path = self.parquet_dir / f"{symbol}.parquet"
        if path.exists():
            existing = pd.read_parquet(path)
            merged = pd.concat([existing, frame], ignore_index=True).drop_duplicates(subset=["datetime"])
        else:
            merged = frame
        merged = merged.sort_values("datetime")
        merged.to_parquet(path, index=False)

    def load_bars(self, symbol: str) -> pd.DataFrame:
        path = self.parquet_dir / f"{symbol}.parquet"
        if not path.exists():
            return pd.DataFrame()
        return pd.read_parquet(path).sort_values("datetime")

    def write_model_registry(
        self,
        symbol: str,
        version: str,
        metrics: dict[str, Any],
        promoted: bool,
        tag: str | None = None,
    ) -> None:
        bars_through_date = metrics.get("bars_through_date")
        registry_tag = tag if tag is not None else metrics.get("tag")
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO model_registry
                (symbol, version, metrics_json, promoted, created_at, bars_through_date, tag)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    symbol,
                    version,
                    json.dumps(metrics),
                    int(promoted),
                    _utc_now_iso(),
                    bars_through_date,
                    registry_tag,
                ),
            )

    def get_latest_promoted_model(self, symbol: str) -> tuple[str, dict[str, Any]] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT version, metrics_json
                FROM model_registry
                WHERE symbol = ? AND promoted = 1
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (symbol,),
            ).fetchone()
        if not row:
            return None
        return row[0], json.loads(row[1])

    def list_promoted_versions(self, symbol: str) -> list[tuple[str, str]]:
        """Return (version, created_at) for all promoted versions, oldest first."""
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT version, created_at
                FROM model_registry
                WHERE symbol = ? AND promoted = 1
                ORDER BY created_at ASC
                """,
                (symbol,),
            ).fetchall()
        return [(row[0], row[1]) for row in rows]

    def list_all_symbols(self) -> list[str]:
        """Return all distinct symbols in the model registry."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT DISTINCT symbol FROM model_registry ORDER BY symbol"
            ).fetchall()
        return [row[0] for row in rows]

    def get_latest_model(self, symbol: str) -> tuple[str, dict[str, Any], bool] | None:
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT version, metrics_json, promoted
                FROM model_registry
                WHERE symbol = ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (symbol,),
            ).fetchone()
        if not row:
            return None
        return row[0], json.loads(row[1]), bool(row[2])

    def save_sentiment(self, symbol: str, window_key: str, result: SentimentResult) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO sentiment_cache (symbol, window_key, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (symbol, window_key, json.dumps(asdict(result)), _utc_now_iso()),
            )

    def save_signal(self, signal: FusedSignal) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO signals (symbol, signal_type, confidence, reason, sell_guide, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    signal.symbol,
                    signal.signal_type,
                    signal.confidence,
                    signal.reason,
                    signal.sell_guide,
                    _utc_now_iso(),
                ),
            )

    def should_suppress_alert(self, symbol: str, signal_type: str, confidence: float, dedup_hours: int) -> bool:
        cutoff = (datetime.now(tz=timezone.utc) - timedelta(hours=dedup_hours)).isoformat()
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT confidence
                FROM alerts_sent
                WHERE symbol = ? AND signal_type = ? AND created_at >= ?
                ORDER BY created_at DESC
                LIMIT 1
                """,
                (symbol, signal_type, cutoff),
            ).fetchone()
        if not row:
            return False
        old_conf = float(row[0])
        return confidence <= (old_conf + 0.05)

    def mark_alert_sent(self, symbol: str, signal_type: str, confidence: float) -> None:
        with self._connect() as conn:
            conn.execute(
                """
                INSERT INTO alerts_sent (symbol, signal_type, confidence, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (symbol, signal_type, confidence, _utc_now_iso()),
            )

    def save_articles(self, symbol: str, source: str, articles: list[dict[str, Any]]) -> None:
        with self._connect() as conn:
            for article in articles:
                article_hash = article["article_hash"]
                conn.execute(
                    """
                    INSERT OR REPLACE INTO news_articles
                    (article_hash, symbol, source, title, url, published_at, content_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        article_hash,
                        symbol,
                        source,
                        article.get("title", ""),
                        article.get("url", ""),
                        article.get("published_at", _utc_now_iso()),
                        json.dumps(article),
                    ),
                )

    def load_recent_articles(self, symbol: str, limit: int = 30) -> list[dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT content_json
                FROM news_articles
                WHERE symbol = ?
                ORDER BY published_at DESC
                LIMIT ?
                """,
                (symbol, limit),
            ).fetchall()
        return [json.loads(r[0]) for r in rows]
