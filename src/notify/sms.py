from __future__ import annotations

from twilio.rest import Client

from notify.base import AlertMessage
from settings import Settings


def send_sms_alert(settings: Settings, msg: AlertMessage) -> None:
    if not settings.alert_sms_to or not settings.twilio_account_sid:
        return
    client = Client(settings.twilio_account_sid, settings.twilio_auth_token)
    client.messages.create(
        body=f"{msg.subject}\n{msg.body}",
        from_=settings.twilio_from_number,
        to=settings.alert_sms_to,
    )
