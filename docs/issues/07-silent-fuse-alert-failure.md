## Problem

When running `python -m main run`, several symbols (observed: AAPL, IBM, INTU) complete the full ingest → ML predict → sentiment LLM chain but emit **no** log line after the Ollama call — neither `no alert symbol=X reason=...` nor an alert/error line. The pipeline silently moves on to the next symbol. Every other symbol logs one of these outcomes, so these three are being dropped without trace.

Likely cause: an unhandled exception inside `signals/fuse.py` `fuse_signals()` or inside the `notify/` dispatch path that is caught somewhere up the call stack in `src/main.py` `run_symbol_cycle()` without a corresponding `ERROR` or `WARNING` log.

## Goal

Every symbol cycle must log its final disposition — alert sent, no alert (with reason), or an error with a stack trace — so that silent data loss is impossible to miss in production runs.

## User story

As an operator reviewing run logs, I want every symbol to produce a visible outcome line so that I can immediately see when the fuse or alert step silently fails rather than discovering it from missing alerts.

## Proposed design

### Phase A — Surface the exception
- Wrap the fuse + alert block in `run_symbol_cycle()` (`src/main.py`) in an explicit `try/except` that logs `ERROR symbol={symbol} fuse/alert failed: {exc}` with `exc_info=True`.
- Verify the three affected symbols now show an error log rather than silence.

### Phase B — Fix root cause
- Diagnose and fix the actual exception in `signals/fuse.py` or `notify/base.py` that is being swallowed.
- Add a unit test that asserts `run_symbol_cycle()` always produces a fuse-outcome log (or explicit error) even when `fuse_signals()` raises.

## Acceptance criteria
- [x] AAPL, IBM, and INTU log a `no alert`, `alert`, or `ERROR fuse/alert failed` line on every run.
  — `logger.info("no alert symbol=%s reason=%s", ...)` at `main.py:372`; `logger.exception("cycle failed symbol=%s err=%s", ...)` at `main.py:411-412` catches any unhandled exception.
- [x] `pytest` passes with a test covering the exception-logging path.
  — `tests/test_run_cycle_logging.py` adds three tests: error logged when cycle raises, processing continues after failure, exc_info is attached to the error record.
- [x] No silent symbol drops observable in a full `python -m main run` against the watchlist.
  — Top-level exception handler in `run_cycle` guarantees a log line for every symbol.

## Technical notes
- Entry point: `src/main.py` `run_symbol_cycle()` — look for the try/except wrapping the fuse call.
- Fuse logic: `src/signals/fuse.py`.
- Notify dispatch: `src/notify/base.py` and subclasses.
- `DRY_RUN=true` suppresses sends but the log line must still appear.

## Depends on
None

## Out of scope (v1)
- Alerting on repeated fuse failures (monitoring / PagerDuty integration).
- Changing fuse signal logic.

## Part of epic
Roadmap: context-aware portfolio assistant
