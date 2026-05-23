## Problem

`src/sentiment/llm_provider.py` `SYSTEM_PROMPT` contained the instruction:

> "Default most outputs to 0.3–0.65; reserve 0.75+ for exceptional clarity."

Ollama (and other small LLMs) interpreted this literally and consistently output `confidence: 0.45` for any stock with mixed or inconclusive news. Because `sentiment_confidence_min` in `thresholds.yaml` is `0.58`, a neutral confidence of `0.45` always fails the gate in `fuse_signals()`, silently blocking BUY alerts even for stocks with 5–6 bullish ML horizons (GOOGL, NVDA, MSFT, META observed in run logs 2026-05-23).

The instruction was intended to prevent overconfidence, but instead produced a hard ceiling that made the neutral band (0.3–0.65) entirely useless — every neutral stock was stuck at 0.45, every bullish signal was blocked by sentiment gating.

## Goal

Remove the prescriptive default and replace it with calibrated bands that let the model use the full 0.0–1.0 range proportionate to actual evidence strength, so that clearly-neutral stocks reach 0.5–0.64 (above the 0.58 gate) and only truly ambiguous/sparse-news cases stay below 0.5.

## User story

As a trader, I want the sentiment gate to block signals only when news is genuinely uncertain, so that stocks with strong ML signals and neutral (but not bearish) news still generate alerts.

## Proposed design

### Phase A — Fix SYSTEM_PROMPT confidence bands (shipped)

Remove "Default most outputs to 0.3–0.65" from `SYSTEM_PROMPT` in `src/sentiment/llm_provider.py`.
Replace with explicit bands:
- 0.0–0.2: no relevant articles
- 0.3–0.49: sparse/ambiguous/contradictory evidence
- 0.5–0.64: balanced news — neutral stocks with present-but-inconclusive evidence
- 0.65–0.79: moderate conviction, leans one direction
- 0.8–1.0: strong consistent signal
Add "Calibrate to actual evidence strength; do not artificially compress scores."

### Phase B — Optional: add confidence logging at INFO level

Log `confidence=X.XX` alongside `sentiment_label` in `analyze_symbol_sentiment()` so future regressions are immediately visible in run output without needing debug mode.

## Acceptance criteria

- [ ] `SYSTEM_PROMPT` no longer contains "Default most outputs to 0.3–0.65"
- [ ] Neutral stocks with 10+ news articles produce confidence in 0.5–0.65 range in a live run
- [ ] Stocks with sparse news (< 5 articles) stay below 0.5
- [ ] Tests: `pytest tests/test_llm_provider.py` passes (161 total pass)
- [ ] No regression in `pytest` full suite

## Technical notes

Root cause is entirely in the SYSTEM_PROMPT string constant, not in parsing or fuse logic.
`sentiment_confidence_min: 0.58` in `thresholds.yaml` is intentional — the fix is in the prompt, not the threshold.

## Depends on

None — standalone prompt fix.

## Out of scope (v1)

- Changing `sentiment_confidence_min` threshold
- Per-provider prompt tuning

## Part of epic

bugs-runtime-reliability-epic
