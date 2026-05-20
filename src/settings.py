from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    schwab_api_key: str = Field(default="", alias="SCHWAB_API_KEY")
    schwab_app_secret: str = Field(default="", alias="SCHWAB_APP_SECRET")
    schwab_callback_url: str = Field(default="https://127.0.0.1", alias="SCHWAB_CALLBACK_URL")
    schwab_token_path: Path = Field(default=Path("./secrets/token.json"), alias="SCHWAB_TOKEN_PATH")

    newsapi_key: str = Field(default="", alias="NEWSAPI_KEY")
    alphavantage_key: str = Field(default="", alias="ALPHAVANTAGE_KEY")
    # Pause between NewsAPI calls when processing a watchlist (helps avoid HTTP 429).
    newsapi_delay_seconds: float = Field(default=0.0, alias="NEWSAPI_DELAY_SECONDS")

    llm_provider: str = Field(default="openai", alias="LLM_PROVIDER")
    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    anthropic_api_key: str = Field(default="", alias="ANTHROPIC_API_KEY")
    anthropic_model: str = Field(default="claude-haiku-4-5-20251001", alias="ANTHROPIC_MODEL")
    # Anthropic server-side web search (extra API cost; enable in Claude Console privacy settings).
    anthropic_web_search_enabled: bool = Field(default=False, alias="ANTHROPIC_WEB_SEARCH_ENABLED")
    # When true, also attach web_search during sentiment (doubles Anthropic calls per symbol).
    anthropic_web_search_on_sentiment: bool = Field(default=False, alias="ANTHROPIC_WEB_SEARCH_ON_SENTIMENT")
    anthropic_web_search_max_uses: int = Field(default=2, alias="ANTHROPIC_WEB_SEARCH_MAX_USES")
    anthropic_delay_seconds: float = Field(default=2.0, alias="ANTHROPIC_DELAY_SECONDS")
    ollama_base_url: str = Field(default="http://localhost:11434", alias="OLLAMA_BASE_URL")
    ollama_model: str = Field(default="llama3.1", alias="OLLAMA_MODEL")
    llm_timeout_seconds: int = Field(default=180, alias="LLM_TIMEOUT_SECONDS")
    llm_article_limit: int = Field(default=25, alias="LLM_ARTICLE_LIMIT")

    watchlist_path: Path = Field(default=Path("./config/watchlist.yaml"), alias="WATCHLIST_PATH")
    thresholds_path: Path = Field(default=Path("./config/thresholds.yaml"), alias="THRESHOLDS_PATH")
    sqlite_path: Path = Field(default=Path("./data/stock_assistant.sqlite"), alias="SQLITE_PATH")
    parquet_dir: Path = Field(default=Path("./data/parquet"), alias="PARQUET_DIR")
    model_dir: Path = Field(default=Path("./data/models"), alias="MODEL_DIR")

    dry_run: bool = Field(default=True, alias="DRY_RUN")
    alert_dedup_hours: int = Field(default=24, alias="ALERT_DEDUP_HOURS")
    log_colors: bool = Field(default=True, alias="LOG_COLORS")
    max_bar_age_days: int = Field(default=3, alias="MAX_BAR_AGE_DAYS")

    smtp_host: str = Field(default="", alias="SMTP_HOST")
    smtp_port: int = Field(default=587, alias="SMTP_PORT")
    smtp_user: str = Field(default="", alias="SMTP_USER")
    smtp_password: str = Field(default="", alias="SMTP_PASSWORD")
    smtp_from: str = Field(default="", alias="SMTP_FROM")
    alert_email_to: str = Field(default="", alias="ALERT_EMAIL_TO")

    twilio_account_sid: str = Field(default="", alias="TWILIO_ACCOUNT_SID")
    twilio_auth_token: str = Field(default="", alias="TWILIO_AUTH_TOKEN")
    twilio_from_number: str = Field(default="", alias="TWILIO_FROM_NUMBER")
    alert_sms_to: str = Field(default="", alias="ALERT_SMS_TO")

    telegram_bot_token: str = Field(default="", alias="TELEGRAM_BOT_TOKEN")
    telegram_chat_id: str = Field(default="", alias="TELEGRAM_CHAT_ID")
    ntfy_topic: str = Field(default="", alias="NTFY_TOPIC")


def ensure_runtime_dirs(settings: Settings) -> None:
    settings.sqlite_path.parent.mkdir(parents=True, exist_ok=True)
    settings.parquet_dir.mkdir(parents=True, exist_ok=True)
    settings.model_dir.mkdir(parents=True, exist_ok=True)
    settings.schwab_token_path.parent.mkdir(parents=True, exist_ok=True)
