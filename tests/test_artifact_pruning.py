"""Tests for non-promoted artifact pruning in train_with_retries."""
from __future__ import annotations

from pathlib import Path

import pytest

from ml.train import prune_excess_promoted_artifacts


class _StubStore:
    def __init__(self, promoted_versions: list[tuple[str, str]]):
        self._versions = promoted_versions

    def list_promoted_versions(self, symbol: str) -> list[tuple[str, str]]:
        return list(self._versions)


def test_prune_non_promoted_files_are_deleted(tmp_path):
    """Non-promoted artifact files should be deleted, promoted ones kept."""
    symbol = "AAPL"

    # Create artifact files for 3 promoted versions
    for ver in ["v1", "v2", "v3"]:
        p = tmp_path / symbol / ver
        p.mkdir(parents=True)
        (p / "models.pkl").write_bytes(b"payload")

    store = _StubStore(
        [
            ("v1", "2024-01-01T00:00:00"),
            ("v2", "2024-01-02T00:00:00"),
            ("v3", "2024-01-03T00:00:00"),
        ]
    )
    freed = prune_excess_promoted_artifacts(store, tmp_path, symbol, max_versions=2)

    # v1 should be deleted (oldest, beyond max_versions=2)
    assert not (tmp_path / symbol / "v1" / "models.pkl").exists()
    assert (tmp_path / symbol / "v2" / "models.pkl").exists()
    assert (tmp_path / symbol / "v3" / "models.pkl").exists()
    assert freed > 0


def test_no_pruning_when_within_limit(tmp_path):
    symbol = "TSLA"
    for ver in ["v1", "v2"]:
        p = tmp_path / symbol / ver
        p.mkdir(parents=True)
        (p / "models.pkl").write_bytes(b"x")

    store = _StubStore(
        [("v1", "2024-01-01T00:00:00"), ("v2", "2024-01-02T00:00:00")]
    )
    freed = prune_excess_promoted_artifacts(store, tmp_path, symbol, max_versions=5)
    assert freed == 0
    assert (tmp_path / symbol / "v1" / "models.pkl").exists()
    assert (tmp_path / symbol / "v2" / "models.pkl").exists()


def test_missing_artifact_file_does_not_raise(tmp_path):
    symbol = "MSFT"
    store = _StubStore(
        [
            ("v1", "2024-01-01T00:00:00"),
            ("v2", "2024-01-02T00:00:00"),
        ]
    )
    # Don't create any actual files
    freed = prune_excess_promoted_artifacts(store, tmp_path, symbol, max_versions=1)
    assert freed == 0


def test_prune_multiple_excess_versions(tmp_path):
    symbol = "GOOG"
    versions = [f"v{i}" for i in range(1, 6)]
    for ver in versions:
        p = tmp_path / symbol / ver
        p.mkdir(parents=True)
        (p / "models.pkl").write_bytes(b"data" * 100)

    promoted = [(ver, f"2024-01-0{i+1}T00:00:00") for i, ver in enumerate(versions)]
    store = _StubStore(promoted)
    freed = prune_excess_promoted_artifacts(store, tmp_path, symbol, max_versions=2)

    # v1, v2, v3 should be deleted; v4, v5 kept
    for ver in ["v1", "v2", "v3"]:
        assert not (tmp_path / symbol / ver / "models.pkl").exists()
    for ver in ["v4", "v5"]:
        assert (tmp_path / symbol / ver / "models.pkl").exists()
    assert freed > 0
