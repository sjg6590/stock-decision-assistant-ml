from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class CvFold:
    train_end: int   # exclusive; fold train = df.iloc[:train_end]
    val_start: int   # inclusive
    val_end: int     # exclusive


def purged_cv_folds(
    n_rows_train_val: int,
    n_splits: int,
    purge_bars: int,
    embargo_bars: int = 0,
) -> list[CvFold]:
    """Walk-forward purged folds within a train+val region (holdout excluded).

    Divides n_rows_train_val into (n_splits + 1) equal blocks. Block 0 seeds
    the initial training window; blocks 1..n_splits are sequential val windows.

    Purge: each fold's train ends at val_start - purge_bars, ensuring no
    training label (which spans up to purge_bars bars forward) overlaps the
    val window.  This is the Lopez de Prado purge idea adapted for a
    sequential (walk-forward) setting rather than combinatorial k-fold.

    Embargo (optional): train_end is shifted back by an additional embargo_bars
    as an isolation buffer after the previous val period.

    Fewer than n_splits folds are returned when the data is too short or
    train_end would fall below 1.
    """
    if n_splits < 1 or n_rows_train_val < 1:
        return []

    block = n_rows_train_val // (n_splits + 1)
    if block < 1:
        return []

    gap = purge_bars + embargo_bars
    folds: list[CvFold] = []
    for k in range(n_splits):
        val_start = block * (k + 1)
        val_end = min(block * (k + 2), n_rows_train_val)
        train_end = val_start - gap
        if train_end < 1 or val_end <= val_start:
            continue
        folds.append(CvFold(train_end=train_end, val_start=val_start, val_end=val_end))
    return folds


@dataclass(frozen=True)
class SplitIndices:
    train_end: int    # exclusive; train  = df.iloc[:train_end]
    val_end: int      # exclusive; val    = df.iloc[train_end:val_end]
    holdout_end: int  # exclusive; holdout = df.iloc[val_end:holdout_end]


def fixed_tail_split(df: pd.DataFrame, val_size: int, holdout_size: int) -> SplitIndices:
    train_end = len(df) - (val_size + holdout_size)
    val_end = len(df) - holdout_size
    return SplitIndices(train_end=train_end, val_end=val_end, holdout_end=len(df))


def rolling_holdout_splits(
    df: pd.DataFrame,
    val_size: int,
    holdout_size: int,
    n_attempts: int,
    step_bars: int,
    min_train_rows: int = 200,
) -> list[SplitIndices]:
    """Return up to n_attempts splits where each holdout window shifts back by step_bars.

    Attempt 0 uses the most recent holdout (identical to fixed_tail_split).
    Attempt k shifts the [val | holdout] window back by k * step_bars bars.
    Stops early when train_end would fall below min_train_rows.
    """
    splits: list[SplitIndices] = []
    for k in range(n_attempts):
        shift = k * step_bars
        holdout_end = len(df) - shift
        val_end = holdout_end - holdout_size
        train_end = val_end - val_size
        if train_end < min_train_rows:
            break
        if val_end <= train_end or holdout_end <= val_end:
            break
        splits.append(SplitIndices(train_end=train_end, val_end=val_end, holdout_end=holdout_end))
    return splits
