# Plan: data requirements & holdout sizing

**Project:** stock-decision-assistant
**Goal:** Raise the minimum training data from 600 bars (~2.4 years) to 1200 bars (~5 years) and double the holdout window so promotion gates are statistically meaningful. Handle the cold-start problem for tickers with limited history.
**Audience:** Claude Code / future implementation sessions
**Status:** Not started

---

## Background

The current configuration in `config/thresholds.yaml`:

```yaml
train:
  min_rows: 600       # ~2.4 years of daily bars
  val_size: 90        # ~4.5 months
  holdout_size: 90    # ~4.5 months
```

**Why these numbers are insufficient:**

| Problem | Detail |
|---|---|
| 600-bar minimum | Only ~420 bars of actual training data after val/holdout are reserved. That's ~1.7 years — one market regime. A model trained only on a bull market has no exposure to drawdowns, rising-rate environments, or liquidity crises. |
| 90-bar holdout | ~4.5 months is short enough that a single earnings season or macro shock can make the Sharpe look great or terrible by chance. The gate is noisy. |
| 90-bar val | Same issue: the F1-tuned threshold is fit on a short window and may over-adapt to one mini-regime. |

**Target configuration:**

```yaml
train:
  min_rows: 1200      # ~5 years of daily bars
  val_size: 126       # ~6 months (one half-year)
  holdout_size: 180   # ~9 months (covers multiple earnings cycles)
```

This yields ~894 training bars (~3.5 years) once val+holdout are reserved — enough to span at least one full bull/bear cycle for most tickers.

---

## Phases

### Phase D — Raise min_rows and holdout/val sizes

**Objective:** Single config change with a migration path for existing tickers.

**Config change:**
```yaml
train:
  min_rows: 1200
  val_size: 126
  holdout_size: 180
```

**Also update the purged_cv default:**
```yaml
train_retry:
  purged_cv:
    n_splits: 3
    purge_bars: 22
    embargo_bars: 5    # unchanged
```

The purge/embargo values are driven by max horizon bars (22 for 1mo), not by the total data size — no change needed there.

**Impact on rolling_holdout strategy:**
`rolling.step_bars: 30` shifts the holdout back 30 bars per attempt. With `holdout_size: 180`, attempts 1–5 span ~900 bars of out-of-sample time total (before `min_train_rows` clamps). This is more stable than the current 5 × 90 = 450-bar total.

**Migration:** Existing tickers with < 1200 bars in the store will raise `ValueError` from `train_with_retries`. The caller in `src/main.py` already catches and logs this — no code change needed. The operator should run a historical back-fill for those tickers (or accept they won't retrain until enough history accumulates).

**Files:**
- `config/thresholds.yaml` — update `min_rows`, `val_size`, `holdout_size`
- `tests/test_train.py` — update any fixtures/mocks that assume 600-row minimum or 90-bar split sizes
- `tests/test_splits.py` — same; re-verify purged CV fold sizes are valid under new split sizes

**Acceptance criteria:**
- `pytest tests/ -q` passes with no split-size assertion failures.
- `python -m main backtest AAPL` completes (AAPL has 5+ years of history via Schwab).
- Tickers with insufficient history log a clear `ValueError` at INFO level and skip gracefully.

---

### Phase E — Tiered cold-start handling

**Objective:** Don't silently skip recently-IPO'd tickers or tickers added to the watchlist mid-run. Give the operator visibility and a fallback.

**Problem:** Currently `train_with_retries` raises `ValueError: Not enough rows for {symbol}: {n} < {min_rows}` and the cycle moves on. There is no warning in the registry that this ticker needs more history, no ETA, and no lower-data-requirement fallback.

**Design: two-tier training config**

Add a `train_cold_start` block to `thresholds.yaml`:

```yaml
train_cold_start:
  enabled: true
  min_rows: 600           # allow training on 600–1199 bars
  val_size: 60            # shorter val
  holdout_size: 60        # shorter holdout
  promotion:              # stricter gates to compensate for noisier eval
    aggregate_accuracy_min: 0.57
    sharpe_min: 0.60
    max_drawdown_max: 0.12
  tag: "cold_start"       # written to model registry for filtering
```

**Behaviour:**
1. `train_with_retries` first attempts full training (1200-bar requirements).
2. If `len(bars) < min_rows` and `cold_start.enabled`, it falls back to cold-start config with tighter gates and tags the artifact `cold_start` in the registry.
3. Cold-start models are used for predictions but `fuse_signals` applies an additional confidence discount (`cold_start_confidence_discount: 0.1` off the fused score) so alerts from cold-start models are less likely to fire.

**Files:**
- `config/thresholds.yaml` — `train_cold_start` block
- `src/config_loader.py` — expose `train_cold_start`
- `src/ml/train.py` — two-tier fallback logic; write `tag` to metrics dict
- `src/data/store.py` — store `tag` column in `model_registry` table (migration)
- `src/signals/fuse.py` — apply confidence discount when prediction artifact is tagged `cold_start`
- `tests/test_train.py` — test cold-start fallback fires on 700-row input; test tag is written

**Acceptance criteria:**
- A ticker with 700 bars trains under cold-start config and logs `cold_start=True`.
- A ticker with 500 bars (below both thresholds) fails cleanly with an INFO log.
- Cold-start artifact tag visible in SQLite `model_registry.tag` column.
- Fused confidence for a cold-start artifact is lower than equivalent non-cold-start.

---

### Phase F — Switch default strategy to purged_cv

**Objective:** `seed_only` is documented as "for fast iteration" but it is the production default. `purged_cv` prevents label leakage across horizon windows and gives more stable val metrics. It should be the default.

**Config change:**
```yaml
train_retry:
  strategy: purged_cv   # was: seed_only
```

**Why now:** With the larger data requirements from Phase D, each split has enough rows to support 3 CV folds without cramped training windows. With 1200 bars and val_size=126, holdout_size=180, the train+val region is ~1020 bars — comfortable for 3 folds of ~300 bars each with a 27-bar purge+embargo gap.

**Verify fold sizes before switching** by running:
```bash
python -m main backtest AAPL --debug 2>&1 | grep "purged_cv\|cv_fold"
```
Confirm each fold has at least `min_rows / 3` training rows.

**Files:**
- `config/thresholds.yaml` — change `strategy: seed_only` → `strategy: purged_cv`
- `README.md` — update the "Default: seed_only" note in the validation strategies table

**Acceptance criteria:**
- Default backtest uses purged_cv without `--strategy` flag.
- Debug log shows `strategy=purged_cv` and `n_folds=3` for each attempt.
- Promotion rate on AAPL does not collapse (if it does, check fold row counts and purge gap).

---

## Implementation order

```
Phase D (raise min_rows + holdout)     ← config-only + test updates; do first
     ↓
Phase E (cold-start tier)              ← code change; builds on Phase D sizes
     ↓
Phase F (default strategy → purged_cv) ← config flip; do last; verify fold sizes
```

---

## Config reference: before and after

| Key | Before | After (Phase D) | Notes |
|---|---|---|---|
| `train.min_rows` | 600 | 1200 | ~5 years daily |
| `train.val_size` | 90 | 126 | ~6 months |
| `train.holdout_size` | 90 | 180 | ~9 months |
| `train_retry.strategy` | `seed_only` | `purged_cv` | Phase F |
| `train_cold_start.min_rows` | _(new)_ | 600 | Phase E |
| `train_cold_start.holdout_size` | _(new)_ | 60 | Phase E |

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Most watchlist tickers don't have 5 years of Schwab history | Run a one-time historical ingest; cold-start tier (Phase E) covers the gap |
| Larger splits make purged_cv folds very small on short-history tickers | Cold-start tier uses smaller splits; fold-size guard already in `purged_cv_folds` |
| Switching default strategy changes promotion behaviour | Audit promotion rate on watchlist before/after; rollback is a one-line config change |
| Holdout 180 bars + val 126 bars = 306 bars reserved; 1200 - 306 = 894 train | Acceptable; still 3.5+ years of training data |

---

## Reference: files map

```
config/thresholds.yaml      # Phases D and F
src/config_loader.py        # Phase E: cold_start block
src/ml/train.py             # Phase E: two-tier fallback
src/data/store.py           # Phase E: tag column migration
src/signals/fuse.py         # Phase E: cold-start confidence discount
tests/test_train.py         # Phases D and E
tests/test_splits.py        # Phase D: verify fold sizes under new split sizes
README.md                   # Phase F: update default strategy note
```
