from __future__ import annotations

import numpy as np
import pandas as pd

from ml.splits import CvFold, SplitIndices, fixed_tail_split, purged_cv_folds, rolling_holdout_splits


def _make_df(n: int = 1200) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    dt = pd.date_range("2020-01-01", periods=n, freq="B")
    close = 100.0 + np.cumsum(rng.normal(0, 1, n))
    return pd.DataFrame({"datetime": dt, "close": close})


class TestFixedTailSplit:
    def test_indices(self) -> None:
        df = _make_df(1000)
        s = fixed_tail_split(df, val_size=90, holdout_size=90)
        assert s.train_end == 820
        assert s.val_end == 910
        assert s.holdout_end == 1000

    def test_no_leakage(self) -> None:
        df = _make_df(500)
        s = fixed_tail_split(df, val_size=60, holdout_size=60)
        assert 0 < s.train_end < s.val_end < s.holdout_end

    def test_holdout_end_equals_len(self) -> None:
        df = _make_df(600)
        s = fixed_tail_split(df, val_size=90, holdout_size=90)
        assert s.holdout_end == len(df)

    def test_split_sizes(self) -> None:
        df = _make_df(600)
        s = fixed_tail_split(df, val_size=90, holdout_size=90)
        assert s.val_end - s.train_end == 90
        assert s.holdout_end - s.val_end == 90


class TestRollingHoldoutSplits:
    def test_returns_up_to_n_attempts(self) -> None:
        df = _make_df(1200)
        splits = rolling_holdout_splits(df, val_size=90, holdout_size=90, n_attempts=5, step_bars=30)
        assert len(splits) == 5

    def test_first_attempt_matches_fixed_tail(self) -> None:
        df = _make_df(1200)
        splits = rolling_holdout_splits(df, val_size=90, holdout_size=90, n_attempts=5, step_bars=30)
        fixed = fixed_tail_split(df, val_size=90, holdout_size=90)
        assert splits[0] == fixed

    def test_different_holdout_windows(self) -> None:
        df = _make_df(1200)
        splits = rolling_holdout_splits(df, val_size=90, holdout_size=90, n_attempts=5, step_bars=30)
        val_ends = [s.val_end for s in splits]
        assert len(set(val_ends)) == len(splits)

    def test_holdout_windows_shift_back_by_step(self) -> None:
        df = _make_df(1200)
        step = 30
        splits = rolling_holdout_splits(df, val_size=90, holdout_size=90, n_attempts=5, step_bars=step)
        for i in range(len(splits) - 1):
            assert splits[i].val_end - splits[i + 1].val_end == step
            assert splits[i].holdout_end - splits[i + 1].holdout_end == step

    def test_no_index_leakage(self) -> None:
        df = _make_df(1200)
        splits = rolling_holdout_splits(df, val_size=90, holdout_size=90, n_attempts=5, step_bars=30)
        for s in splits:
            assert 0 < s.train_end < s.val_end < s.holdout_end

    def test_holdout_size_constant_across_attempts(self) -> None:
        df = _make_df(1200)
        splits = rolling_holdout_splits(df, val_size=90, holdout_size=90, n_attempts=5, step_bars=30)
        for s in splits:
            assert s.holdout_end - s.val_end == 90

    def test_val_size_constant_across_attempts(self) -> None:
        df = _make_df(1200)
        splits = rolling_holdout_splits(df, val_size=90, holdout_size=90, n_attempts=5, step_bars=30)
        for s in splits:
            assert s.val_end - s.train_end == 90

    def test_min_train_rows_limits_attempts(self) -> None:
        # len=1200, val=90, holdout=90, step=200, min_train=600
        # k=0: train_end = 1200-0*200-90-90 = 1020 >= 600 -> included
        # k=1: train_end = 1200-1*200-90-90 =  820 >= 600 -> included
        # k=2: train_end = 1200-2*200-90-90 =  620 >= 600 -> included
        # k=3: train_end = 1200-3*200-90-90 =  420 < 600  -> stop
        df = _make_df(1200)
        splits = rolling_holdout_splits(
            df, val_size=90, holdout_size=90, n_attempts=5, step_bars=200, min_train_rows=600
        )
        assert len(splits) == 3

    def test_empty_when_insufficient_data(self) -> None:
        df = _make_df(200)
        splits = rolling_holdout_splits(
            df, val_size=90, holdout_size=90, n_attempts=5, step_bars=30, min_train_rows=600
        )
        assert splits == []

    def test_returns_split_indices_instances(self) -> None:
        df = _make_df(1200)
        splits = rolling_holdout_splits(df, val_size=90, holdout_size=90, n_attempts=3, step_bars=30)
        assert all(isinstance(s, SplitIndices) for s in splits)


class TestPurgedCvFolds:
    def test_returns_correct_fold_count(self) -> None:
        folds = purged_cv_folds(n_rows_train_val=600, n_splits=3, purge_bars=22, embargo_bars=0)
        assert len(folds) == 3

    def test_returns_cv_fold_instances(self) -> None:
        folds = purged_cv_folds(n_rows_train_val=600, n_splits=3, purge_bars=22, embargo_bars=0)
        assert all(isinstance(f, CvFold) for f in folds)

    def test_purge_gap_maintained(self) -> None:
        purge = 22
        folds = purged_cv_folds(n_rows_train_val=600, n_splits=3, purge_bars=purge, embargo_bars=0)
        for fold in folds:
            assert fold.val_start - fold.train_end >= purge

    def test_embargo_included_in_gap(self) -> None:
        purge, embargo = 22, 5
        folds = purged_cv_folds(n_rows_train_val=600, n_splits=3, purge_bars=purge, embargo_bars=embargo)
        for fold in folds:
            assert fold.val_start - fold.train_end >= purge + embargo

    def test_no_train_val_overlap(self) -> None:
        folds = purged_cv_folds(n_rows_train_val=600, n_splits=3, purge_bars=22, embargo_bars=0)
        for fold in folds:
            assert fold.train_end <= fold.val_start

    def test_sequential_val_windows(self) -> None:
        folds = purged_cv_folds(n_rows_train_val=600, n_splits=3, purge_bars=22, embargo_bars=0)
        for i in range(len(folds) - 1):
            assert folds[i].val_end <= folds[i + 1].val_start

    def test_val_end_within_bounds(self) -> None:
        n = 600
        folds = purged_cv_folds(n_rows_train_val=n, n_splits=3, purge_bars=22, embargo_bars=0)
        for fold in folds:
            assert fold.val_end <= n

    def test_fewer_folds_when_purge_is_large(self) -> None:
        # block=80, purge=200; first two folds get train_end < 1 and are skipped
        folds = purged_cv_folds(n_rows_train_val=400, n_splits=4, purge_bars=200, embargo_bars=0)
        assert len(folds) < 4

    def test_returns_empty_when_data_too_short(self) -> None:
        folds = purged_cv_folds(n_rows_train_val=10, n_splits=3, purge_bars=22, embargo_bars=0)
        assert folds == []

    def test_returns_empty_for_zero_splits(self) -> None:
        folds = purged_cv_folds(n_rows_train_val=600, n_splits=0, purge_bars=22, embargo_bars=0)
        assert folds == []

    def test_expanding_train_window(self) -> None:
        # Each successive fold's train_end should be larger (expanding window)
        folds = purged_cv_folds(n_rows_train_val=600, n_splits=3, purge_bars=22, embargo_bars=0)
        train_ends = [f.train_end for f in folds]
        assert train_ends == sorted(train_ends)


# Phase D production split sizes (config/thresholds.yaml train block)
VAL_SIZE = 126
HOLDOUT_SIZE = 180
MIN_ROWS = 1200


class TestPhaseDSplitSizes:
    def test_fixed_tail_split_production_indices(self) -> None:
        df = _make_df(MIN_ROWS)
        s = fixed_tail_split(df, val_size=VAL_SIZE, holdout_size=HOLDOUT_SIZE)
        assert s.train_end == MIN_ROWS - (VAL_SIZE + HOLDOUT_SIZE)
        assert s.val_end == MIN_ROWS - HOLDOUT_SIZE
        assert s.holdout_end == MIN_ROWS
        assert s.val_end - s.train_end == VAL_SIZE
        assert s.holdout_end - s.val_end == HOLDOUT_SIZE

    def test_rolling_holdout_production_min_train_rows(self) -> None:
        # len=1200, val=126, holdout=180, step=200, min_train=600
        # k=0: train_end=894; k=1: 694; k=2: 494 < 600 -> 2 attempts
        df = _make_df(MIN_ROWS)
        splits = rolling_holdout_splits(
            df,
            val_size=VAL_SIZE,
            holdout_size=HOLDOUT_SIZE,
            n_attempts=5,
            step_bars=200,
            min_train_rows=600,
        )
        assert len(splits) == 2

    def test_purged_cv_folds_valid_under_production_train_val_region(self) -> None:
        # train+val region = len - holdout = 1020 bars
        n_rows_train_val = MIN_ROWS - HOLDOUT_SIZE
        folds = purged_cv_folds(
            n_rows_train_val=n_rows_train_val,
            n_splits=3,
            purge_bars=22,
            embargo_bars=5,
        )
        assert len(folds) == 3
        for fold in folds:
            assert fold.val_start - fold.train_end >= 22 + 5
            assert fold.val_end <= n_rows_train_val
