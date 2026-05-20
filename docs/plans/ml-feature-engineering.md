# Plan: richer feature engineering

**Project:** stock-decision-assistant
**Goal:** Expand the 10-feature baseline with technically-grounded indicators that give XGBoost more signal: volatility context, intrabar range, and cross-sectional market-relative return.
**Audience:** Claude Code / future implementation sessions
**Status:** Not started

---

## Background

The current feature set in `src/ml/features.py` has 10 columns:

| Feature | What it captures |
|---|---|
| `ret_1`, `ret_5`, `ret_20` | Recent momentum |
| `vol_20` | Realised volatility |
| `vol_z` | Volume relative to recent mean |
| `gap` | Overnight gap |
| `rsi_14` | Momentum oscillator |
| `macd`, `macd_signal` | Trend/momentum cross |
| `day_of_week` | Calendar seasonality |

**Gaps identified:**
- No measure of current price position within recent range (Bollinger Bands).
- No realised range-based volatility (ATR) — `vol_20` uses returns, not high/low.
- No cross-sectional context — the model sees a ticker in isolation with no awareness of what the market is doing.
- `day_of_week` is likely to be arbitraged away in liquid names; it can stay but carries low expected value.

---

## Phases

### Phase A — Intrabar volatility (ATR)

**Objective:** Replace/supplement `vol_20` (return-based) with ATR (range-based), which is more robust to gaps and is the standard measure quant systems use for position sizing and regime detection.

**Feature to add:** `atr_14` — 14-period Average True Range, normalised by close price.

```python
# in build_features()
from ta.volatility import AverageTrueRange
df["atr_14"] = AverageTrueRange(
    high=df["high"], low=df["low"], close=df["close"], window=14
).average_true_range() / (df["close"] + 1e-9)
```

**Requires:** `high` and `low` columns in the bars DataFrame. These are already fetched from Schwab (`src/data/ingest.py`) — confirm they are stored in SQLite/Parquet before using.

**Files:**
- `src/ml/features.py` — add `atr_14` computation and add to `feature_columns()`
- `tests/test_features.py` — new file; assert column present, no NaN after warmup, finite values

**Acceptance criteria:**
- `atr_14` present in output of `build_features()` and `feature_columns()`
- `pytest tests/test_features.py` passes
- `python -m main backtest AAPL` completes without error

---

### Phase B — Bollinger Band width and %B

**Objective:** Tell the model where the current price sits relative to recent volatility bounds. %B near 0 = near lower band (oversold); near 1 = near upper band (overbought). Width captures the volatility regime — squeeze precedes breakout.

**Features to add:**

| Feature | Formula | Intuition |
|---|---|---|
| `bb_width` | `(upper - lower) / middle` | Volatility regime (wide = high vol) |
| `bb_pct_b` | `(close - lower) / (upper - lower)` | Position within the band |

```python
from ta.volatility import BollingerBands
bb = BollingerBands(close=df["close"], window=20, window_dev=2)
df["bb_width"] = (bb.bollinger_hband() - bb.bollinger_lband()) / (bb.bollinger_mavg() + 1e-9)
df["bb_pct_b"] = bb.bollinger_pband()
```

**Files:**
- `src/ml/features.py`
- `tests/test_features.py` (extend Phase A test file)

**Acceptance criteria:**
- Both features present in `feature_columns()` output
- `bb_pct_b` clipped or handled when `upper == lower` (denominator guard)
- Backtest completes on AAPL

---

### Phase C — Market-relative return (cross-sectional context)

**Objective:** The model currently cannot distinguish a stock that rose 2% because the whole market rose 2% from one that rose 2% while the market fell. Excess return over SPY is the simplest cross-sectional signal.

**Feature to add:** `ret_5_vs_spy` — 5-day log return of the ticker minus the 5-day log return of SPY over the same window.

**Design choices:**
- SPY bars are fetched using the same Schwab market-data client and stored as a "reference ticker" in the same bar store. The reference symbol is configurable (`config/thresholds.yaml`).
- If SPY bars are missing or stale for a given date, the feature is `0.0` (neutral, not `NaN`) so the model degrades gracefully rather than dropping rows.
- Only `ret_5_vs_spy` to start. `ret_1_vs_spy` and `ret_20_vs_spy` can be added later if feature importance shows them useful.

**Config addition:**
```yaml
# config/thresholds.yaml
features:
  market_reference_symbol: "SPY"   # used for cross-sectional ret features
```

**Implementation sketch:**

```python
# src/ml/features.py
def build_features(frame: pd.DataFrame, spy_frame: pd.DataFrame | None = None) -> pd.DataFrame:
    ...
    if spy_frame is not None and not spy_frame.empty:
        spy = spy_frame.sort_values("datetime").set_index("datetime")["close"]
        ticker_ret5 = np.log(df["close"] / df["close"].shift(5))
        spy_aligned = spy.reindex(pd.to_datetime(df["datetime"])).ffill()
        spy_ret5 = np.log(spy_aligned / spy_aligned.shift(5))
        df["ret_5_vs_spy"] = (ticker_ret5 - spy_ret5.values).fillna(0.0)
    else:
        df["ret_5_vs_spy"] = 0.0
    ...
```

**Call-site change:** `train_with_retries` and `predict_symbol` receive SPY bars from the main cycle; gracefully skip if not available.

**Files:**
- `src/ml/features.py` — add `ret_5_vs_spy`; update `feature_columns()`
- `src/main.py` — fetch SPY bars once per cycle; pass into train and predict calls
- `config/thresholds.yaml` — `features.market_reference_symbol`
- `src/config_loader.py` — expose `features` block
- `tests/test_features.py` — test with and without `spy_frame`; test NaN-safety

**Acceptance criteria:**
- When `spy_frame` is `None`, feature is `0.0` for all rows (no errors).
- When `spy_frame` is provided, values are finite and non-trivially zero on AAPL vs SPY.
- Backtest with and without SPY data completes cleanly.

---

## Implementation order

```
Phase A (ATR)          ← lowest risk; self-contained; no new data dependency
     ↓
Phase B (Bollinger)    ← also self-contained; same data as A
     ↓
Phase C (SPY relative) ← new data fetch; touches main cycle; do last
```

---

## What to measure after each phase

Run `python -m main backtest AAPL --debug` before and after each phase and compare:

- Holdout aggregate accuracy (should trend up, not down)
- Holdout Sharpe (same)
- Promotion rate across the watchlist (should not collapse — if it does, a feature is introducing NaN or leakage)

Record results in the model registry. If a new feature consistently hurts metrics, drop it; XGBoost feature importance can diagnose which inputs are being used.

---

## Out of scope for this plan

- Fundamental data (P/E, EPS) — requires a separate data source
- Options flow / implied volatility — separate data source
- Sentiment features fed into the ML model (currently kept as a separate signal leg)
- Alternative model architectures (LSTM, transformers)
- Intraday bars

---

## Reference: files map

```
src/ml/features.py          # all changes land here
src/main.py                 # Phase C: SPY fetch wiring
config/thresholds.yaml      # Phase C: market_reference_symbol
src/config_loader.py        # Phase C: features block
tests/test_features.py      # new file covering all phases
```
