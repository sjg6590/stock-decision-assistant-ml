from __future__ import annotations

from typing import Any


def get_account_numbers(client: Any) -> list[dict]:
    response = client.get_account_numbers()
    response.raise_for_status()
    return response.json()


def get_linked_accounts(client: Any) -> list[dict]:
    response = client.get_accounts()
    response.raise_for_status()
    return response.json()
