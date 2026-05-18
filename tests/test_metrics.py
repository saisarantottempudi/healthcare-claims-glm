"""Tests for evaluation metrics."""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.evaluation.metrics import (
    calibration_by_decile,
    gamma_deviance,
    gini_coefficient,
    lorenz_curve,
    mae,
    mape,
    pearson_residuals,
    poisson_deviance,
    rmse,
)


class TestPoissonDeviance:
    def test_perfect_prediction_zero(self):
        y = np.array([2.0, 3.0, 1.0])
        assert poisson_deviance(y, y) == pytest.approx(0.0, abs=1e-9)

    def test_positive_values(self):
        y_true = np.array([2.0, 3.0, 1.0])
        y_pred = np.array([1.5, 3.5, 0.8])
        assert poisson_deviance(y_true, y_pred) > 0

    def test_zeros_handled(self):
        y_true = np.array([0.0, 1.0, 2.0])
        y_pred = np.array([0.5, 1.0, 2.0])
        result = poisson_deviance(y_true, y_pred)
        assert np.isfinite(result)


class TestGammaDeviance:
    def test_perfect_prediction_zero(self):
        y = np.array([100.0, 200.0, 150.0])
        assert gamma_deviance(y, y) == pytest.approx(0.0, abs=1e-9)

    def test_positive_values(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([80.0, 220.0])
        assert gamma_deviance(y_true, y_pred) > 0


class TestBasicMetrics:
    def test_mae_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert mae(y, y) == pytest.approx(0.0)

    def test_rmse_perfect(self):
        y = np.array([1.0, 2.0, 3.0])
        assert rmse(y, y) == pytest.approx(0.0)

    def test_mape_positive(self):
        y_true = np.array([100.0, 200.0])
        y_pred = np.array([110.0, 180.0])
        assert mape(y_true, y_pred) > 0

    def test_rmse_greater_than_mae(self):
        y_true = np.array([1.0, 1.0, 10.0])
        y_pred = np.array([1.0, 1.0, 1.0])
        assert rmse(y_true, y_pred) >= mae(y_true, y_pred)


class TestGini:
    def test_perfect_model_gini_high(self):
        rng = np.random.default_rng(42)
        n = 1_000
        # Perfect model: score exactly equals the outcome
        y_true  = rng.poisson(1.0, n).astype(float)
        y_score = y_true.copy()
        gini = gini_coefficient(y_true, y_score)
        assert gini >= 0.3

    def test_random_model_gini_near_zero(self):
        rng = np.random.default_rng(42)
        n = 5_000
        y_true = rng.poisson(0.5, n).astype(float)
        y_score = rng.uniform(0, 1, n)
        gini = gini_coefficient(y_true, y_score)
        assert abs(gini) < 0.1

    def test_gini_range(self):
        rng = np.random.default_rng(99)
        y_true  = rng.poisson(1.0, 1_000).astype(float)
        y_score = rng.exponential(1.0, 1_000)
        gini = gini_coefficient(y_true, y_score)
        assert -1.0 <= gini <= 1.0


class TestLorenzCurve:
    def test_output_starts_at_zero_ends_at_one(self):
        y_true  = np.array([1.0, 2.0, 3.0, 4.0])
        y_score = np.array([0.1, 0.2, 0.3, 0.4])
        x, y = lorenz_curve(y_true, y_score)
        assert x[0] == pytest.approx(0.0)
        assert y[-1] == pytest.approx(1.0)

    def test_monotone_increasing(self):
        rng = np.random.default_rng(0)
        y_true  = rng.poisson(1.0, 100).astype(float)
        y_score = rng.exponential(1.0, 100)
        _, y = lorenz_curve(y_true, y_score)
        assert np.all(np.diff(y) >= -1e-10)


class TestCalibration:
    def test_calibration_returns_dataframe(self):
        y_true = np.random.poisson(0.5, 500).astype(float)
        y_pred = np.random.exponential(0.5, 500)
        cal = calibration_by_decile(y_true, y_pred, n_bins=10)
        assert "actual_mean" in cal.columns
        assert "pred_mean" in cal.columns
        assert "ratio" in cal.columns

    def test_calibration_ratio_near_one_for_perfect_model(self):
        y = np.random.exponential(2.0, 500)
        cal = calibration_by_decile(y, y, n_bins=5)
        assert cal["ratio"].between(0.99, 1.01).all()


class TestPearsonResiduals:
    def test_zero_for_perfect_prediction(self):
        y = np.array([1.0, 2.0, 3.0])
        r = pearson_residuals(y, y, family="poisson")
        np.testing.assert_allclose(r, 0.0, atol=1e-9)

    def test_gamma_residuals_shape(self):
        y_true = np.array([100.0, 200.0, 300.0])
        y_pred = np.array([90.0, 210.0, 290.0])
        r = pearson_residuals(y_true, y_pred, family="gamma")
        assert r.shape == y_true.shape

    def test_invalid_family_raises(self):
        with pytest.raises(ValueError):
            pearson_residuals(np.array([1.0]), np.array([1.0]), family="binomial")
