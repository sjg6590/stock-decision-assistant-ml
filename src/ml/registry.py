from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any


def model_path(model_dir: Path, symbol: str, version: str) -> Path:
    path = model_dir / symbol / version
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_models(model_dir: Path, symbol: str, version: str, payload: dict[str, Any]) -> Path:
    path = model_path(model_dir, symbol, version) / "models.pkl"
    with path.open("wb") as fh:
        pickle.dump(payload, fh)
    return path


def load_models(model_dir: Path, symbol: str, version: str) -> dict[str, Any]:
    path = model_path(model_dir, symbol, version) / "models.pkl"
    with path.open("rb") as fh:
        return pickle.load(fh)
