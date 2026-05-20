from __future__ import annotations

import smtplib
from email.message import EmailMessage

from notify.base import AlertMessage
from settings import Settings


def send_email_alert(settings: Settings, msg: AlertMessage) -> None:
    if not settings.alert_email_to or not settings.smtp_host:
        return
    message = EmailMessage()
    message["Subject"] = msg.subject
    message["From"] = settings.smtp_from or settings.smtp_user
    message["To"] = settings.alert_email_to
    message.set_content(msg.body)
    with smtplib.SMTP(settings.smtp_host, settings.smtp_port, timeout=20) as server:
        server.starttls()
        if settings.smtp_user:
            server.login(settings.smtp_user, settings.smtp_password)
        server.send_message(message)
