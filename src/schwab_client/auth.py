from __future__ import annotations

from typing import Any

from schwab.auth import easy_client

from settings import Settings


def build_client(settings: Settings, use_async: bool = False) -> Any:
    if not settings.schwab_api_key or not settings.schwab_app_secret:
        raise ValueError("Missing Schwab credentials in environment.")
    return easy_client(
        api_key=settings.schwab_api_key,
        app_secret=settings.schwab_app_secret,
        callback_url=settings.schwab_callback_url,
        token_path=str(settings.schwab_token_path),
        asyncio=use_async,
    )
