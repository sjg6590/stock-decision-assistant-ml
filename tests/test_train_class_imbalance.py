"""Tests for scale_pos_weight class-imbalance correction in training."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ml.train import _build_models, _compute_scale_pos_weight


def test_compute_scale_pos_weight_imbalanced():
    # 70% positive -> n_neg=30, n_pos=70 -> weight ≈ 0.4286
    y = pd.Series([1] * 70 + [0] * 30)
    spw = _compute_scale_pos_weight(y)
    assert abs(spw - 30 / 70) < 1e-9


def test_compute_scale_pos_weight_balanced():
    y = pd.Series([1] * 50 + [0] * 50)
    spw = _compute_scale_pos_weight(y)
    assert abs(spw - 1.0) < 1e-9


def test_compute_scale_pos_weight_all_positive_safe():
    y = pd.Series([1, 1, 1])
    spw = _compute_scale_pos_weight(y)
    assert spw == 1.0


def test_build_models_passes_scale_pos_weight():
    clf, reg = _build_models(seed=42, early_stopping_rounds=10, scale_pos_weight=2.5)
    assert clf.get_params()["scale_pos_weight"] == pytest.approx(2.5)


def test_build_models_default_weight_is_one():
    clf, _ = _build_models(seed=42, early_stopping_rounds=10)
    assert clf.get_params()["scale_pos_weight"] == pytest.approx(1.0)


def test_regressor_has_no_scale_pos_weight():
    _, reg = _build_models(seed=42, early_stopping_rounds=10, scale_pos_weight=3.0)
    # XGBRegressor doesn't use scale_pos_weight; confirm it's not set
    params = reg.get_params()
    assert params.get("scale_pos_weight", None) is None or params.get("scale_pos_weight", 1.0) == 1.0
