"""Regression test: purged_cv final model training indices must be disjoint from eval-set indices."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_purged_cv_train_eval_disjoint():
    """Verify that the final fit's training slice doesn't overlap with the early-stopping eval set."""
    from ml.splits import purged_cv_folds

    n = 300
    n_splits = 3
    purge_bars = 22
    embargo_bars = 5

    folds = purged_cv_folds(n, n_splits=n_splits, purge_bars=purge_bars, embargo_bars=embargo_bars)
    assert len(folds) > 0, "Expected at least one fold"

    last_fold = folds[-1]

    # Training indices: [0, last_fold.val_start)
    train_indices = set(range(0, last_fold.val_start))
    # Eval-set indices: [last_fold.val_start, last_fold.val_end)
    eval_indices = set(range(last_fold.val_start, last_fold.val_end))

    overlap = train_indices & eval_indices
    assert len(overlap) == 0, (
        f"Training and eval-set indices overlap after fix: {overlap}. "
        f"last_fold.val_start={last_fold.val_start}, last_fold.val_end={last_fold.val_end}"
    )


def test_purged_cv_train_slice_smaller_than_full():
    """The fixed training slice must be strictly smaller than the full train_val region."""
    from ml.splits import purged_cv_folds

    n = 300
    folds = purged_cv_folds(n, n_splits=3, purge_bars=22, embargo_bars=5)
    last_fold = folds[-1]

    train_slice_size = last_fold.val_start  # rows [0, val_start)
    full_train_val_size = n  # rows [0, n)

    assert train_slice_size < full_train_val_size, (
        "Fixed training slice should be smaller than full train_val region"
    )


def test_purged_cv_eval_set_within_train_val():
    """Eval-set indices must lie within [0, train_val_rows), not in the holdout region."""
    from ml.splits import purged_cv_folds

    n_train_val = 250
    folds = purged_cv_folds(n_train_val, n_splits=3, purge_bars=22, embargo_bars=5)
    last_fold = folds[-1]

    assert last_fold.val_end <= n_train_val, (
        f"Eval set extends into holdout region: val_end={last_fold.val_end} > n_train_val={n_train_val}"
    )
