"""Tests for data loading, schema validation, and preprocessing."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.generate_data import generate_claims_dataset
from src.data.loader import validate_schema, split_data
from src.data.preprocessor import preprocess


@pytest.fixture(scope="module")
def raw_df():
    return generate_claims_dataset(n_samples=1_000, seed=0)


@pytest.fixture(scope="module")
def processed_df(raw_df):
    return preprocess(raw_df)


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestSchemaValidation:
    def test_required_columns_present(self, raw_df):
        validate_schema(raw_df)  # should not raise

    def test_missing_column_raises(self, raw_df):
        with pytest.raises(ValueError, match="Missing columns"):
            validate_schema(raw_df.drop(columns=["claim_count"]))

    def test_negative_claim_count_raises(self, raw_df):
        bad = raw_df.copy()
        bad.loc[0, "claim_count"] = -1
        with pytest.raises(ValueError, match="negative"):
            validate_schema(bad)

    def test_negative_claim_amount_raises(self, raw_df):
        bad = raw_df.copy()
        bad.loc[0, "claim_amount"] = -100
        with pytest.raises(ValueError, match="negative"):
            validate_schema(bad)


# ── Distribution sanity tests ─────────────────────────────────────────────────

class TestDataDistributions:
    def test_claim_count_non_negative(self, raw_df):
        assert (raw_df["claim_count"] >= 0).all()

    def test_claim_amount_non_negative(self, raw_df):
        assert (raw_df["claim_amount"] >= 0).all()

    def test_zero_claim_rate_reasonable(self, raw_df):
        zero_rate = (raw_df["claim_count"] == 0).mean()
        assert 0.4 < zero_rate < 0.9, f"Unexpected zero-claim rate: {zero_rate:.2%}"

    def test_exposure_years_in_range(self, raw_df):
        assert raw_df["exposure_years"].between(0.01, 1.0).all()

    def test_age_range(self, raw_df):
        assert raw_df["age"].between(18, 100).all()

    def test_plan_type_values(self, raw_df):
        valid = {"Bronze", "Silver", "Gold", "Platinum"}
        assert set(raw_df["plan_type"].unique()).issubset(valid)

    def test_region_values(self, raw_df):
        valid = {"London", "South East", "Midlands", "North England", "Scotland", "Wales"}
        assert set(raw_df["region"].unique()).issubset(valid)


# ── Preprocessing tests ───────────────────────────────────────────────────────

class TestPreprocessing:
    def test_bmi_category_created(self, processed_df):
        assert "bmi_category" in processed_df.columns

    def test_age_band_created(self, processed_df):
        assert "age_band" in processed_df.columns

    def test_bmi_excess_non_negative(self, processed_df):
        assert (processed_df["bmi_excess"] >= 0).all()

    def test_log_exposure_is_log_of_exposure(self, processed_df):
        expected = np.log(processed_df["exposure_years"])
        np.testing.assert_allclose(processed_df["log_exposure"], expected, rtol=1e-6)

    def test_pure_premium_computed(self, processed_df):
        assert "pure_premium" in processed_df.columns
        assert (processed_df["pure_premium"] >= 0).all()

    def test_no_nulls_in_key_columns(self, processed_df):
        key_cols = ["age", "bmi", "claim_count", "claim_amount", "exposure_years"]
        assert processed_df[key_cols].isnull().sum().sum() == 0


# ── Train/test split ──────────────────────────────────────────────────────────

class TestSplit:
    def test_split_sizes(self, processed_df):
        train, test = split_data(processed_df, test_size=0.2, random_state=42)
        total = len(train) + len(test)
        assert total == len(processed_df)
        assert abs(len(test) / total - 0.2) < 0.05

    def test_no_overlap(self, processed_df):
        train, test = split_data(processed_df, test_size=0.2, random_state=42)
        train_ids = set(train["policy_id"].tolist())
        test_ids  = set(test["policy_id"].tolist())
        assert len(train_ids & test_ids) == 0
