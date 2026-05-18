"""
FastAPI endpoint tests using httpx TestClient.

These tests require a fitted model to be present at models/.
Run `make train` (or `python scripts/train.py`) before running the full suite.
"""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from httpx import AsyncClient
from fastapi.testclient import TestClient

from src.api.main import app

MODEL_DIR = ROOT / "models"

pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "frequency_model.pkl").exists(),
    reason="Trained models not found. Run `python scripts/train.py` first.",
)


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


VALID_PAYLOAD = {
    "age": 45,
    "gender": "Female",
    "bmi": 27.5,
    "smoker": False,
    "chronic_conditions": True,
    "num_dependants": 2,
    "region": "London",
    "plan_type": "Gold",
    "years_as_customer": 5,
    "exposure_years": 1.0,
}


class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        r = client.get("/health")
        assert r.status_code == 200

    def test_health_model_loaded(self, client):
        r = client.get("/health")
        assert r.json()["model_loaded"] is True

    def test_health_status_ok(self, client):
        r = client.get("/health")
        assert r.json()["status"] == "ok"


class TestPredictEndpoint:
    def test_returns_200(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.status_code == 200

    def test_response_has_required_fields(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        data = r.json()
        assert "pred_frequency" in data
        assert "pred_severity" in data
        assert "pred_pure_premium" in data
        assert "risk_band" in data

    def test_predictions_are_positive(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        data = r.json()
        assert data["pred_frequency"] > 0
        assert data["pred_severity"] > 0
        assert data["pred_pure_premium"] > 0

    def test_pure_premium_equals_freq_times_sev(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        data = r.json()
        expected = data["pred_frequency"] * data["pred_severity"]
        # Values are rounded to 2 dp on serialisation; allow rounding error
        assert abs(data["pred_pure_premium"] - expected) < 1.0

    def test_risk_band_valid(self, client):
        r = client.post("/predict", json=VALID_PAYLOAD)
        assert r.json()["risk_band"] in {"LOW", "MEDIUM", "HIGH", "VERY_HIGH"}

    def test_smoker_has_higher_premium(self, client):
        smoker_payload = {**VALID_PAYLOAD, "smoker": True}
        non_smoker_payload = {**VALID_PAYLOAD, "smoker": False}
        r_smoker     = client.post("/predict", json=smoker_payload)
        r_non_smoker = client.post("/predict", json=non_smoker_payload)
        assert (
            r_smoker.json()["pred_pure_premium"]
            > r_non_smoker.json()["pred_pure_premium"]
        )

    def test_chronic_condition_raises_premium(self, client):
        chronic = {**VALID_PAYLOAD, "chronic_conditions": True}
        healthy = {**VALID_PAYLOAD, "chronic_conditions": False}
        r_chronic = client.post("/predict", json=chronic)
        r_healthy = client.post("/predict", json=healthy)
        assert (
            r_chronic.json()["pred_pure_premium"]
            > r_healthy.json()["pred_pure_premium"]
        )

    def test_invalid_region_returns_422(self, client):
        bad = {**VALID_PAYLOAD, "region": "Mars"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_invalid_plan_type_returns_422(self, client):
        bad = {**VALID_PAYLOAD, "plan_type": "Diamond"}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422

    def test_age_out_of_range_returns_422(self, client):
        bad = {**VALID_PAYLOAD, "age": 150}
        r = client.post("/predict", json=bad)
        assert r.status_code == 422


class TestBatchPredictEndpoint:
    def test_batch_returns_200(self, client):
        payload = {"policies": [VALID_PAYLOAD, {**VALID_PAYLOAD, "age": 30}]}
        r = client.post("/predict/batch", json=payload)
        assert r.status_code == 200

    def test_batch_count_matches_input(self, client):
        n = 5
        payload = {"policies": [VALID_PAYLOAD] * n}
        r = client.post("/predict/batch", json=payload)
        assert r.json()["count"] == n

    def test_batch_empty_returns_422(self, client):
        r = client.post("/predict/batch", json={"policies": []})
        assert r.status_code == 422


class TestModelInfoEndpoint:
    def test_returns_200(self, client):
        r = client.get("/model/info")
        assert r.status_code == 200

    def test_contains_aic(self, client):
        r = client.get("/model/info")
        data = r.json()
        assert "aic" in data["frequency_model"]
        assert "aic" in data["severity_model"]

    def test_contains_features(self, client):
        r = client.get("/model/info")
        data = r.json()
        assert len(data["features"]) > 0
