# Stock Decision Assistant

Local Python daemon that combines Schwab market data, ML predictions, news sentiment, and multi-channel alerts.

**This project is for education and research only. It is not investment advice.**

## Features

- Walk-forward train / validation / holdout flow per ticker with promotion gates (accuracy, Sharpe, drawdown).
- Prediction horizons: `1d`, `3d`, `5d`, `2w`, `3w`, `1mo`.
- Validation strategies: `seed_only`, `rolling_holdout`, `purged_cv` (default).
- News + LLM sentiment (yfinance, Alpha Vantage, NewsAPI, optional Anthropic web search).
- Provider-agnostic LLM layer: `openai`, `anthropic`, or `ollama`.
- Alert channels: email, SMS (Twilio), Telegram, ntfy — with deduplication and confidence bump logic.
- Default `DRY_RUN=true` so alerts are logged but not sent until you opt in.

## Requirements

- Python **3.11+**
- [Schwab developer app](https://developer.schwab.com/) (API key, secret, OAuth callback URL)
- Optional: NewsAPI, Alpha Vantage, and an LLM provider API key (or local Ollama)
- Optional: SMTP / Twilio / Telegram / ntfy for live alerts

## Quick start

```bash
git clone https://github.com/sjg6590/stock-decision-assistant.git
cd stock-decision-assistant

python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"

cp .env.example .env
cp config/watchlist.example.yaml config/watchlist.yaml
cp config/thresholds.example.yaml config/thresholds.yaml
# Edit .env with your API keys and paths
```

### Schwab OAuth (one-time)

```bash
python -m main schwab-login
# or: python scripts/schwab_login.py
```

Tokens are written to `SCHWAB_TOKEN_PATH` (default `./secrets/token.json`). That path is gitignored.

### Run

```bash
python -m main run          # one full watchlist cycle
python -m main daemon       # scheduled daemon
python -m main retrain AAPL
python -m main backtest AAPL
```

## Configuration

| File | Purpose |
|------|---------|
| `.env` | API keys, paths, `DRY_RUN`, alert channels — copy from `.env.example` |
| `config/watchlist.yaml` | Ticker symbols — copy from `config/watchlist.example.yaml` |
| `config/thresholds.yaml` | Training, promotion, fusion — copy from `config/thresholds.example.yaml` |

`config/watchlist.yaml` and `config/thresholds.yaml` are **not** tracked in git so you can tune locally without merge noise. Example files document every knob.

Runtime data:

- SQLite + Parquet under `data/` (gitignored except `data/.gitkeep`)
- Model artifacts under `data/models/{SYMBOL}/{VERSION}/`

## Environment variables

See [`.env.example`](.env.example) for the full list. Minimum for core workflow:

| Variable | Required |
|----------|----------|
| `SCHWAB_API_KEY`, `SCHWAB_APP_SECRET`, `SCHWAB_CALLBACK_URL` | Yes |
| `NEWSAPI_KEY`, `ALPHAVANTAGE_KEY` | Optional (improves news coverage) |
| `OPENAI_API_KEY` / `ANTHROPIC_API_KEY` / Ollama | Per `LLM_PROVIDER` |
| Alert channel vars | Only if `DRY_RUN=false` |

Keep the watchlist small on free Alpha Vantage and NewsAPI tiers to avoid HTTP 429 rate limits.

## Project layout

```
stock-decision-assistant/
  config/           # watchlist + thresholds (local copies from *.example.yaml)
  scripts/          # schwab_login.py
  src/              # application code
  tests/
  docs/plans/       # design notes
```

## Retrain behavior

`retrain` ingests the latest bars **once** before the retry loop, then runs up to `train.max_retrain_attempts` attempts on those bars (avoids repeated Schwab calls). To re-ingest during a long retry run, set `train_retry.refresh_bars_every_n_attempts` in `config/thresholds.yaml` (default `0` = disabled).

## Validation strategies

Set `train_retry.strategy` in `config/thresholds.yaml`. Full parameter reference: `config/thresholds.example.yaml`.

### `seed_only`

Same fixed train / val / holdout split; each attempt uses a different random seed. Fast iteration.

```
[──────── train ────────][── val ──][── holdout ──]
```

### `rolling_holdout`

Holdout window shifts back by `rolling.step_bars` each attempt — checks calendar stability.

### `purged_cv` (default)

K walk-forward folds inside train+val; holdout is untouched until final promotion. Purge gap follows Lopez de Prado style label isolation (`purge_bars` ≥ max horizon).

| Goal | Strategy |
|------|----------|
| Production default; reduce val variance | `purged_cv` |
| Fast baseline | `seed_only` |
| Calendar stability check | `rolling_holdout` |

## Debug flags

| Flag | Effect |
|------|--------|
| `--debug` | DEBUG logging + ML plots (requires `matplotlib`) |
| `--debug-plots` | Plots only; log level stays INFO |

Works with `run`, `retrain`, and `backtest`. Plot windows block exit until closed.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check src tests
```

CI runs `pytest` on Python 3.11–3.13 (see `.github/workflows/ci.yml`).

## Security

- **Never commit** `.env`, `secrets/`, SQLite databases, or trained models.
- Rotate API keys if they were ever exposed.
- Use `DRY_RUN=true` until alert channels are configured and tested.

## License

[MIT](LICENSE) — see file for terms. No warranty; not financial advice.
