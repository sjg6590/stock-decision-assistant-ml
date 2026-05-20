from __future__ import annotations

import httpx

from notify.base import AlertMessage
from settings import Settings


def send_telegram_alert(settings: Settings, msg: AlertMessage) -> None:
    if not settings.telegram_bot_token or not settings.telegram_chat_id:
        return
    url = f"https://api.telegram.org/bot{settings.telegram_bot_token}/sendMessage"
    payload = {"chat_id": settings.telegram_chat_id, "text": f"{msg.subject}\n{msg.body}"}
    with httpx.Client(timeout=20) as client:
        client.post(url, json=payload).raise_for_status()


def send_ntfy_alert(settings: Settings, msg: AlertMessage) -> None:
    if not settings.ntfy_topic:
        return
    url = f"https://ntfy.sh/{settings.ntfy_topic}"
    with httpx.Client(timeout=20) as client:
        client.post(url, content=msg.body.encode("utf-8"), headers={"Title": msg.subject}).raise_for_status()
