# Long-term plan: robust ML training & retrain attempts

**Project:** stock-decision-assistant
**Goal:** Replace seed-only retries with time-aware validation, stable model selection, and evaluation that matches live inference.
**Audience:** Claude Code / future implementation sessions
**Status:** Phases 0–4, 7, and 8 complete. Optional work remains: Phase 5 (hyperparameter search), Phase 6 (ensemble inference), and wiring `refresh_bars_every_n_attempts`.

---

## Implementation status

| Phase | Title | Status |
|---|---|---|
| 0 | Foundations | ✅ Done |
| 1 | Honest evaluation & best-attempt selection | ✅ Done |
| 2 | Rolling holdout per attempt | ✅ Done |
| 3 | Purged time-series CV | ✅ Done |
| 4 | Stability gates & promotion v2 | ✅ Done |
| 5 | Hyperparameter search | ❌ Not started (optional) |
| 6 | Ensemble production models | ❌ Not started (optional) |
| 7 | Data freshness & operational hooks | ✅ Done (1 optional knob unwired — see phase) |
| 8 | Debug UX & matplotlib noise | ✅ Done |

**Next up (recommended order):** Phase 5 or 6 when you want more model diversity; wire `refresh_bars_every_n_attempts` if long retry runs need mid-run re-ingest.

---

## Executive summary

`train_with_retries()` in `src/ml/train.py` supports three validation strategies (`seed_only`, `rolling_holdout`, `purged_cv`), probability-based eval with per-horizon F1-tuned thresholds, `best_val_score` attempt selection, and Phase 4 stability gates (multi-seed, multi-window, overfit detector, horizon-weighted accuracy). Train/serve threshold parity is implemented in `predict.py` and `fuse.py`.

**Optional next steps:** hyperparameter search (Phase 5), full ensemble artifacts at inference (Phase 6), and mid-retry bar refresh (Phase 7 config knob exists but is not wired in code).

Original context (resolved): retries used to run on the same split with only the RNG seed changing; promotion used hard `predict()` at 0.5 while inference used `predict_proba()` + `ml_buy_threshold`; failed runs returned the last attempt, not the best; signal fusion ignored per-horizon thresholds.

---

## Current state (as-built)

| Area | Current behavior | File(s) |
|---|---|---|
| Retries | Up to 5 attempts; strategy configurable (`seed_only` \| `rolling_holdout` \| `purged_cv`) | `src/ml/train.py` |
| Split | `_build_splits` dispatches to `splits.py` based on strategy | `src/ml/splits.py`, `train.py` |
| Promotion gates | accuracy, per-horizon min, Sharpe, max DD; optional overfit gap, horizon-weighted acc, multi-seed stability, multi-window (rolling) | `config/thresholds.yaml`, `train.py` |
| Eval metrics | `predict_proba` + per-horizon F1-tuned threshold (val) | `src/ml/evaluate.py`, `train.py` |
| Live signals | `predict_proba`; per-horizon threshold in `HorizonPrediction.threshold`; fusion compares each horizon against its saved threshold (global `ml_buy_threshold` fallback for legacy artifacts) | `src/ml/predict.py`, `src/signals/fuse.py` |
| Multi-seed (Phase 4) | When `ensemble_seeds > 1`: median/mean aggregated proba for gate eval; per-seed stability check; **primary seed's model saved** (full ensemble list deferred to Phase 6) | `src/ml/train.py` |
| Artifacts | Every attempt saved; registry stores full metrics JSON + `bars_through_date` column | `src/ml/registry.py`, `src/data/store.py` |
| CLI | `backtest`, `retrain`, `run`, `daemon`; `--debug`, `--debug-plots` | `src/main.py` |
| Attempt selection | `last_attempt` (default) or `best_val_score` (composite val score) | `src/ml/train.py` |
| Logging | Attempt summary table at INFO; per-attempt DEBUG logs; matplotlib loggers capped at WARNING | `src/logging_config.py` |
| Data freshness | `bars_loaded` DEBUG log; `stale_bars` WARNING when last bar age > `settings.max_bar_age_days` | `src/main.py` |

Default strategy remains `seed_only` — all Phase 4 gates are disabled by default (`ensemble_seeds: 1`, `min_seeds_passing_gates: 0`, `min_passing_windows: 0`, `max_val_holdout_sharpe_gap: 0`).

---

## Design principles

1. **Val tunes, holdout decides** — never tune thresholds or hyperparams on holdout.
2. **Time order is sacred** — no random shuffles; purged gaps for multi-horizon labels.
3. **Stability over peak** — prefer median performance across folds/seeds over one lucky run.
4. **Train eval = prod eval** — same probability threshold logic in training gates and `predict_symbol` / `fuse_signals`.
5. **Incremental PRs** — each phase ships with tests and backward-compatible config defaults.

---

## Phase 0 — Foundations ✅ DONE

**Objective:** Make observability and config hooks ready without changing promotion behavior.

### As-built
- `train_retry` section in `config/thresholds.yaml` with strategy, selection, seeds, `log_attempt_comparison`
- Structured attempt logging and INFO summary table in `train_with_retries`
- Registry metadata: attempt, seed, selection_reason, strategy, split_boundaries, bars_through_date
- Tests in `tests/test_train.py`

### Files
- `config/thresholds.yaml`
- `src/ml/train.py`
- `src/data/store.py`
- `tests/test_train.py`

---

## Phase 1 — Honest evaluation & best-attempt selection ✅ DONE

**Objective:** Align metrics with production; stop returning a worse last attempt.

### As-built
- `directional_accuracy_at_threshold`, `tune_threshold_on_val`, `strategy_metrics_from_proba` in `evaluate.py`
- Per-horizon thresholds in model payload (`model_payload["thresholds"][hz]`)
- `predict_symbol` loads saved thresholds; legacy artifacts fall back to `ml_buy_threshold_fallback` (default 0.60)
- `fuse_signals` compares each horizon's `probability_up` against `HorizonPrediction.threshold` (global `ml_threshold` when `threshold == 0`)
- `best_val_score` selection (composite `val_agg_acc + val_sharpe` via `val_score_weights`)
- Tests in `tests/test_evaluate.py`, `tests/test_train.py`, `tests/test_predict_threshold.py`, `tests/test_signal_fusion.py`

### Files
- `src/ml/evaluate.py`
- `src/ml/train.py`
- `src/ml/predict.py`
- `src/signals/fuse.py`
- `src/main.py`

---

## Phase 2 — Rolling / expanding holdout per attempt ✅ DONE

**Objective:** Each attempt sees a different time window, not just a different seed.

### As-built
- `src/ml/splits.py`: `fixed_tail_split`, `rolling_holdout_splits`
- Config: `train_retry.rolling.step_bars`, `min_train_rows`
- Wired in `train_with_retries`; holdout date range logged per attempt
- Promotion on final holdout per attempt (Option A); multi-window gate in Phase 4 (Option B)
- Tests in `tests/test_splits.py`

### Files
- `src/ml/splits.py`
- `src/ml/train.py`
- `config/thresholds.yaml`
- `tests/test_splits.py`

---

## Phase 3 — Purged time-series CV for validation ✅ DONE

**Objective:** Stable val estimates; reduce optimism from overlapping horizon labels.

### As-built
- `purged_cv_folds` in `splits.py` (Lopez de Prado purge + embargo documented in code)
- K-fold val metrics aggregated (mean); final model fit on full train+val with last fold for early stopping
- Holdout evaluated once per attempt with last-fold threshold
- Config: `train_retry.purged_cv.n_splits`, `purge_bars`, `embargo_bars`
- README validation-strategies section documents purging vs rolling
- Tests in `tests/test_splits.py`, `tests/test_train.py`

### Files
- `src/ml/splits.py`
- `src/ml/train.py`
- `README.md`
- `tests/test_splits.py`, `tests/test_train.py`

---

## Phase 4 — Stability gates & promotion v2 ✅ DONE

**Objective:** Don't promote on one lucky seed or one lucky window.

### As-built

**4.1 Multi-seed stability** (`train_retry.ensemble_seeds`, `train_retry.stability`):
- Trains N classifiers per horizon; aggregates val/holdout proba via median or mean
- Requires `min_seeds_passing_gates` of N individual seeds to pass basic gates (disabled when 0)
- Saves primary seed's classifier/regressor in payload (Phase 6 will persist full list)
- Skipped when `strategy: purged_cv` (fold variance already provides stability signal)

**4.2 Multi-window promotion** (`promotion.min_passing_windows`):
- After all rolling_holdout attempts, blocks promotion unless enough windows passed
- Forces all attempts to run when active (even with `selection: last_attempt`)

**4.3 Overfit detector** (`promotion.max_val_holdout_sharpe_gap`):
- Fails if `val_sharpe - holdout_sharpe > max_gap` (0 = disabled)

**4.4 Horizon-weighted gates** (`promotion.horizon_weights`, `weighted_accuracy_min`):
- Weighted accuracy gate when `horizon_weights` non-empty; per-horizon floor still enforced

**4.5 Tests** in `tests/test_train_promotion.py` (unit + integration smoke tests)

### Acceptance criteria (met)
- Promotion can require 2/3 seeds passing OR 2/N windows passing (config-driven, disabled by default)
- Failed promotion logs `failed_gates` including `stability`, `val_holdout_sharpe_gap`, `weighted_accuracy`, `min_passing_windows`

### Files
- `src/ml/train.py`
- `config/thresholds.yaml`
- `tests/test_train_promotion.py`

---

## Phase 5 — Hyperparameter search ❌ NOT STARTED (optional, ~3–5 days)

**Objective:** Attempt diversity beyond RNG.

### Tasks

**5.1** Small search space (config-driven grid or Optuna):
```
max_depth: [3, 4, 5]
learning_rate: [0.03, 0.05, 0.08]
subsample, colsample_bytree: [0.8, 0.9]
n_estimators cap 300; keep early stopping.
```

**5.2** Search on val only (or inner purged CV from Phase 3).

**5.3** Persist winning params in model payload and registry.

**5.4** Dependency: `optuna` optional extra in `pyproject.toml` `[project.optional-dependencies]`.

**5.5** CLI flag `--no-hparam-search` for fast backtests.

### Acceptance criteria
- Backtest with search completes; best params in `models.pkl` metadata.
- Without Optuna installed, grid fallback works.

### Files
- `src/ml/hparam_search.py` (new)
- `src/ml/train.py`
- `pyproject.toml`
- `tests/test_hparam_search.py`

---

## Phase 6 — Ensemble production models ❌ NOT STARTED (optional, ~2 days)

**Objective:** Deploy ensemble, not single seed winner.

**Note:** Phase 4 already trains multiple seeds for stability gating when `ensemble_seeds > 1`, but only the primary seed's models are saved and used at inference.

### Tasks

**6.1** Train N models per horizon (seeds or bagging); save list in payload:
```python
model_payload["classifiers"][hz] = [clf1, clf2, clf3]
```

**6.2** Inference: average `predict_proba`; average regressor output.

**6.3** Backward compat: `load_models` accepts single model or list.

**6.4** Promotion: gates on ensemble holdout metrics only.

### Acceptance criteria
- Old artifacts still load and predict.
- New artifacts use ensemble by default when `train_retry.ensemble_seeds > 1`.

### Files
- `src/ml/train.py`
- `src/ml/predict.py`
- `src/ml/registry.py`
- `tests/test_predict_ensemble.py`

---

## Phase 7 — Data freshness & operational hooks ✅ DONE (1 optional knob unwired)

**Objective:** Retries aren't pointless re-fits on stale bars.

### As-built
- ✅ `bars_loaded ... from=... to=...` DEBUG log in `run_symbol_cycle` (`src/main.py`)
- ✅ `retrain` semantics documented in README (single ingest before retry loop)
- ✅ `stale_bars` WARNING in `_warn_stale_bars`, called from `run_symbol_cycle` and `retrain_symbol`
- ✅ `bars_through_date` in registry metrics JSON and dedicated SQLite column (`src/data/store.py` migration)
- ✅ Config key `train_retry.refresh_bars_every_n_attempts: 0` documented in README and `thresholds.yaml`

### Remaining (optional)

**7.1** Wire `refresh_bars_every_n_attempts` in `train_with_retries` or `retrain_symbol`:
- When `> 0`, re-call `ingest_symbol_bars` every N attempts inside the retry loop
- Requires passing ingest callback or client into training (currently bars are loaded once upstream)

### Files
- `src/main.py` (wire refresh knob)
- `src/ml/train.py` (optional callback hook)

---

## Phase 8 — Debug UX & matplotlib noise ✅ DONE

**Objective:** `--debug` useful without 900KB font logs.

### As-built
- ✅ `debug_plots.py`: `show_train_plots`, `show_predict_plots`, `show_attempt_comparison`
- ✅ `setup_interactive()` / `wait_for_close()` wired into CLI commands
- ✅ Matplotlib loggers set to WARNING in `setup_logging` (`src/logging_config.py`)
- ✅ `--debug-plots` flag: plots without DEBUG log flood (`src/main.py`)
- ✅ Attempt comparison panel called from `train_with_retries` when `debug=True`

### Files
- `src/logging_config.py`
- `src/ml/debug_plots.py`
- `src/main.py`

---

## Testing strategy (ongoing)

| Layer | What to test |
|---|---|
| Unit | splits, thresholds, metrics, promotion rules |
| Integration | `train_with_retries` on synthetic OHLCV fixture in `tests/fixtures/` |
| Regression | Golden metrics bounds on fixed fixture (not live AAPL — flaky) |
| Manual | `python -m main backtest AAPL` with venv; compare promotion rate over time |

**Fixture:** 800–1200 rows generated pandas DataFrame with datetime, open, high, low, close, volume — committed or built in conftest.

**Existing test files:**

| File | Covers |
|---|---|
| `tests/test_train.py` | retry loop, config defaults, attempt logging |
| `tests/test_splits.py` | rolling, purged CV, leakage guards |
| `tests/test_evaluate.py` | threshold tuning, proba metrics |
| `tests/test_train_promotion.py` | Phase 4 gates, multi-seed, multi-window |
| `tests/test_predict_threshold.py` | legacy artifact fallback |
| `tests/test_signal_fusion.py` | per-horizon threshold in fusion |

---

## Config evolution (single source of truth)

All keys below exist in `config/thresholds.yaml`. Phase 4 and Phase 7 keys default to **disabled** (0 or empty) so behavior matches pre-Phase-4 unless explicitly enabled.

```yaml
horizons: { ... }              # ✅

train:
  min_rows: 600                # ✅
  val_size: 90                 # ✅
  holdout_size: 90             # ✅
  early_stopping_rounds: 20    # ✅
  max_retrain_attempts: 5      # ✅

train_retry:
  strategy: seed_only          # ✅
  selection: last_attempt      # ✅
  seeds: [43, 44, 45, 46, 47]  # ✅
  log_attempt_comparison: true # ✅
  val_score_weights:           # ✅
    accuracy: 0.5
    sharpe: 0.5
  rolling:                     # ✅
    step_bars: 30
    min_train_rows: 600
  purged_cv:                   # ✅
    n_splits: 3
    purge_bars: 22
    embargo_bars: 5
  refresh_bars_every_n_attempts: 0  # ✅ config only; not wired in code
  ensemble_seeds: 1            # ✅ Phase 4 (1 = disabled)
  stability:                   # ✅ Phase 4
    min_seeds_passing_gates: 0
    metric: median

promotion:
  aggregate_accuracy_min: 0.55 # ✅
  per_horizon_accuracy_min: 0.50  # ✅
  sharpe_min: 0.50             # ✅
  max_drawdown_max: 0.15       # ✅
  max_val_holdout_sharpe_gap: 0.0  # ✅ Phase 4 (0 = disabled)
  min_passing_windows: 0       # ✅ Phase 4 (0 = disabled)
  window_metric: median_sharpe  # ✅ Phase 4
  horizon_weights: {}          # ✅ Phase 4 (empty = disabled)
  weighted_accuracy_min: 0.52  # ✅ Phase 4

signal:
  ml_buy_threshold: 0.60       # ✅ — fallback for legacy artifacts without saved thresholds
  sentiment_confidence_min: 0.55  # ✅
```

### Example: enable Phase 4 stability on AAPL backtests

```yaml
train_retry:
  strategy: rolling_holdout
  ensemble_seeds: 3
  stability:
    min_seeds_passing_gates: 2
    metric: median

promotion:
  max_val_holdout_sharpe_gap: 0.75
  min_passing_windows: 2
  horizon_weights:
    "1d": 1.0
    "1mo": 0.5
  weighted_accuracy_min: 0.52
```

---

## Suggested implementation order for Claude Code

**Core plan (Phases 0–4, 7, 8) is complete.** Remaining optional work:

```
Phase 7.1 (wire refresh_bars_every_n_attempts)     ← ~2 hours, only if needed
     ↓
Phase 5 (hyperparameter search)                    ← ~3–5 days, optional
     ↓
Phase 6 (ensemble inference artifacts)             ← ~2 days, optional; builds on Phase 4 multi-seed training
```

**Session prompts for remaining work:**

**Session H — Phase 7.1 (optional)**
> Wire `train_retry.refresh_bars_every_n_attempts` from `config/thresholds.yaml` into the retrain flow: when > 0, re-ingest bars every N attempts inside `train_with_retries` or via a callback from `retrain_symbol`. Default 0 must preserve current single-ingest behavior. Add a test with a mock ingest callback.

**Session I — Phase 5**
> Implement Phase 5 from `docs/plans/ml-training-robustness.md`: config-driven hyperparameter search on val (or inner purged CV), persist winning params in model payload and registry, optional Optuna with grid fallback, `--no-hparam-search` CLI flag.

**Session J — Phase 6**
> Implement Phase 6 from `docs/plans/ml-training-robustness.md`: save list of classifiers/regressors per horizon when `ensemble_seeds > 1`, average `predict_proba` at inference, backward-compatible `load_models`. Build on existing Phase 4 multi-seed training loop in `train.py`.

---

## PR checklist template (paste into each Claude Code session)

```
## PR: [Phase X.Y - title]
### Scope
- [ ] Files listed in plan only
- [ ] Config defaults preserve old behavior
- [ ] Tests added/updated
### Verification
- [ ] `pytest tests/ -q`
- [ ] `source .venv/bin/activate && python -m main backtest AAPL` (or fixture symbol)
- [ ] Logs show new fields without errors
### Docs
- [ ] README updated if CLI/config behavior changed
- [ ] thresholds.yaml commented
```

---

## Risks & mitigations

| Risk | Mitigation |
|---|---|
| Holdout peeking when picking "best of 5" on holdout | Select on val only; holdout once (implemented) |
| Slower backtests (CV × seeds) | `--fast` mode: `seed_only` + 1 fold; `--no-hparam-search` when Phase 5 exists |
| Breaking old `models.pkl` | Version field in payload; loaders handle v1/v2 |
| Schwab rate limits | No extra ingest per attempt unless `refresh_bars_every_n_attempts` wired |
| Stricter gates → never promote | Log near-miss; optional `promotion.relaxed_mode` for dev only |
| Phase 4 multi-seed saves only primary model | Phase 6 adds full ensemble persistence |

---

## Success metrics (3–6 months)

- Holdout Sharpe variance across seeds ↓ (track in registry).
- Promotion rate stable but not zero on watchlist backtests.
- Val/holdout Sharpe gap ↓ (less overfit) — use `max_val_holdout_sharpe_gap` when tuning.
- Live run cycle uses same threshold logic as training (achieved in Phase 1).
- Debug backtest output < 200 lines without losing signal (achieved in Phase 8).

---

## Out of scope (explicit)

- New features / alternative models (LSTM, etc.)
- Sentiment pipeline changes
- Alerting changes
- Cloud training / GPU
- Auto-trading execution

---

## Reference: key files map

```
config/thresholds.yaml     # all knobs
src/ml/train.py            # orchestration, Phase 4 gates
src/ml/splits.py           # rolling + purged CV
src/ml/evaluate.py         # metrics & thresholds
src/ml/predict.py          # inference parity
src/ml/registry.py         # artifact schema
src/ml/debug_plots.py      # Phase 8 visualizations
src/ml/hparam_search.py    # Phase 5 (not yet created)
src/main.py                # CLI, stale bars, debug flags
src/data/store.py          # registry + bars_through_date
src/signals/fuse.py        # per-horizon threshold fusion
src/logging_config.py      # matplotlib log suppression
tests/test_train.py
tests/test_splits.py
tests/test_evaluate.py
tests/test_train_promotion.py
tests/test_predict_threshold.py
tests/test_signal_fusion.py
README.md                  # validation strategies, debug flags, retrain semantics
```
