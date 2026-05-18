"""Tests for Poisson GLM, Gamma GLM, and PurePremiumModel."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_data import generate_claims_dataset
from src.data.loader import split_data
from src.data.preprocessor import preprocess
from src.evaluation.metrics import (
    gini_coefficient,
    poisson_deviance,
    gamma_deviance,
)
from src.models.frequency_model import PoissonFrequencyModel
from src.models.severity_model import GammaSeverityModel
from src.models.pure_premium import PurePremiumModel


@pytest.fixture(scope="module")
def train_test():
    df = preprocess(generate_claims_dataset(n_samples=5_000, seed=1))
    return split_data(df, test_size=0.2, random_state=1)


@pytest.fixture(scope="module")
def freq_model(train_test):
    train, _ = train_test
    m = PoissonFrequencyModel()
    m.fit(train)
    return m


@pytest.fixture(scope="module")
def sev_model(train_test):
    train, _ = train_test
    m = GammaSeverityModel()
    m.fit(train)
    return m


@pytest.fixture(scope="module")
def pure_premium_model(freq_model, sev_model):
    return PurePremiumModel(freq_model, sev_model)


# ── Frequency model ───────────────────────────────────────────────────────────

class TestPoissonFrequencyModel:
    def test_fit_sets_result(self, freq_model):
        assert freq_model.result_ is not None

    def test_aic_finite(self, freq_model):
        assert np.isfinite(freq_model.aic)

    def test_deviance_ratio_positive(self, freq_model):
        assert 0 < freq_model.deviance_ratio < 1

    def test_predict_returns_positive(self, freq_model, train_test):
        _, test = train_test
        preds = freq_model.predict(test)
        assert (preds > 0).all()

    def test_predict_length(self, freq_model, train_test):
        _, test = train_test
        preds = freq_model.predict(test)
        assert len(preds) == len(test)

    def test_coefficient_table_shape(self, freq_model):
        ct = freq_model.coefficient_table()
        assert "feature" in ct.columns
        assert "exp_coef" in ct.columns
        assert len(ct) > 5

    def test_smoker_coefficient_positive(self, freq_model):
        ct = freq_model.coefficient_table()
        smoker_row = ct[ct["feature"].str.contains("smoker", case=False)]
        assert len(smoker_row) > 0
        assert smoker_row["coef"].values[0] > 0, "Smokers should have higher claim rate"

    def test_chronic_coefficient_positive(self, freq_model):
        ct = freq_model.coefficient_table()
        chronic_row = ct[ct["feature"].str.contains("chronic", case=False)]
        assert len(chronic_row) > 0
        assert chronic_row["coef"].values[0] > 0


# ── Severity model ────────────────────────────────────────────────────────────

class TestGammaSeverityModel:
    def test_fit_sets_result(self, sev_model):
        assert sev_model.result_ is not None

    def test_aic_finite(self, sev_model):
        assert np.isfinite(sev_model.aic)

    def test_dispersion_positive(self, sev_model):
        assert sev_model.dispersion_ > 0

    def test_predict_returns_positive(self, sev_model, train_test):
        _, test = train_test
        preds = sev_model.predict(test)
        assert (preds > 0).all()

    def test_smoker_premium_higher(self, sev_model, train_test):
        _, test = train_test
        smokers     = test[test["smoker"]].head(100)
        non_smokers = test[~test["smoker"]].head(100)
        assert sev_model.predict(smokers).mean() > sev_model.predict(non_smokers).mean()


# ── Pure premium model ────────────────────────────────────────────────────────

class TestPurePremiumModel:
    def test_predict_returns_three_columns(self, pure_premium_model, train_test):
        _, test = train_test
        preds = pure_premium_model.predict(test)
        assert set(preds.columns) == {"pred_frequency", "pred_severity", "pred_pure_premium"}

    def test_pure_premium_equals_freq_times_sev(self, pure_premium_model, train_test):
        _, test = train_test
        preds = pure_premium_model.predict(test)
        expected = preds["pred_frequency"] * preds["pred_severity"]
        np.testing.assert_allclose(preds["pred_pure_premium"], expected, rtol=1e-6)

    def test_london_higher_than_wales(self, pure_premium_model, train_test):
        _, test = train_test
        london = test[test["region"] == "London"].head(200)
        wales  = test[test["region"] == "Wales"].head(200)
        if len(london) > 10 and len(wales) > 10:
            assert (
                pure_premium_model.predict(london)["pred_pure_premium"].mean()
                > pure_premium_model.predict(wales)["pred_pure_premium"].mean()
            )

    def test_platinum_higher_than_bronze(self, pure_premium_model, train_test):
        _, test = train_test
        plat   = test[test["plan_type"] == "Platinum"].head(200)
        bronze = test[test["plan_type"] == "Bronze"].head(200)
        if len(plat) > 10 and len(bronze) > 10:
            assert (
                pure_premium_model.predict(plat)["pred_pure_premium"].mean()
                > pure_premium_model.predict(bronze)["pred_pure_premium"].mean()
            )

    def test_gini_above_zero(self, pure_premium_model, train_test):
        _, test = train_test
        preds = pure_premium_model.predict(test)
        gini = gini_coefficient(test["pure_premium"].values, preds["pred_pure_premium"].values)
        assert gini > 0.1, f"Expected positive Gini, got {gini:.4f}"


# ── Serialisation roundtrip ───────────────────────────────────────────────────

class TestSerialisation:
    def test_save_load_frequency(self, freq_model, train_test, tmp_path):
        path = tmp_path / "freq.pkl"
        freq_model.save(path)
        loaded = PoissonFrequencyModel.load(path)
        _, test = train_test
        np.testing.assert_allclose(
            freq_model.predict(test),
            loaded.predict(test),
            rtol=1e-6,
        )

    def test_save_load_severity(self, sev_model, train_test, tmp_path):
        path = tmp_path / "sev.pkl"
        sev_model.save(path)
        loaded = GammaSeverityModel.load(path)
        _, test = train_test
        np.testing.assert_allclose(
            sev_model.predict(test),
            loaded.predict(test),
            rtol=1e-6,
        )
