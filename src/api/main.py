"""
FastAPI prediction service for the Healthcare Claims GLM.

Endpoints
---------
GET  /health            — liveness / readiness check
POST /predict           — single-policy pure premium prediction
POST /predict/batch     — batch prediction (up to 1000 policies)
GET  /model/info        — model metadata and training metrics
GET  /docs              — Swagger UI (auto-generated)

The model is loaded once at startup and kept in memory.
In production, use model registry URIs (MLflow / SageMaker) instead of
loading from local disk.
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from loguru import logger

from src.api.schemas import (
    BatchPredictionRequest,
    BatchPredictionResponse,
    HealthResponse,
    PredictionRequest,
    PredictionResponse,
)
from src.data.preprocessor import preprocess
from src.models.pure_premium import PurePremiumModel

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
VERSION = "1.0.0"

# Global model holder
_model: PurePremiumModel | None = None


def _load_model() -> PurePremiumModel:
    global _model
    if _model is None:
        logger.info(f"Loading PurePremiumModel from {MODEL_DIR} …")
        _model = PurePremiumModel.load(MODEL_DIR)
        logger.success("Model loaded successfully.")
    return _model


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup: load model into memory
    try:
        _load_model()
    except Exception as e:
        logger.error(f"Failed to load model at startup: {e}")
    yield
    # Shutdown: nothing to clean up (stateless)


app = FastAPI(
    title="Healthcare Claims GLM API",
    description=(
        "Predicts expected annual claim frequency, severity, and pure premium "
        "for health insurance policyholders using Poisson (frequency) and "
        "Gamma (severity) GLMs."
    ),
    version=VERSION,
    lifespan=lifespan,
)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _risk_band(pure_premium: float) -> str:
    if pure_premium < 500:
        return "LOW"
    elif pure_premium < 2_000:
        return "MEDIUM"
    elif pure_premium < 5_000:
        return "HIGH"
    return "VERY_HIGH"


def _request_to_df(req: PredictionRequest) -> pd.DataFrame:
    """Convert a single Pydantic request to a preprocessed DataFrame row."""
    raw = pd.DataFrame([req.model_dump()])
    return preprocess(raw)


def _batch_to_df(requests: list[PredictionRequest]) -> pd.DataFrame:
    raw = pd.DataFrame([r.model_dump() for r in requests])
    return preprocess(raw)


def _make_response(row: pd.Series) -> PredictionResponse:
    return PredictionResponse(
        pred_frequency=round(float(row["pred_frequency"]), 4),
        pred_severity=round(float(row["pred_severity"]), 2),
        pred_pure_premium=round(float(row["pred_pure_premium"]), 2),
        risk_band=_risk_band(float(row["pred_pure_premium"])),
        model_version=VERSION,
    )


# ── Routes ────────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse, tags=["infrastructure"])
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=_model is not None,
        version=VERSION,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["prediction"])
def predict(request: PredictionRequest) -> PredictionResponse:
    """
    Predict claim frequency, severity, and pure premium for a single policy.

    Returns:
    - **pred_frequency**: expected annual claim count (λ)
    - **pred_severity**: expected cost per claim in GBP (μ_sev)
    - **pred_pure_premium**: expected annual claims cost = λ × μ_sev
    - **risk_band**: LOW / MEDIUM / HIGH / VERY_HIGH tier
    """
    model = _load_model()
    df = _request_to_df(request)

    try:
        preds = model.predict(df)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    row = preds.iloc[0]
    return _make_response(row)


@app.post("/predict/batch", response_model=BatchPredictionResponse, tags=["prediction"])
def predict_batch(request: BatchPredictionRequest) -> BatchPredictionResponse:
    """
    Batch prediction for up to 1,000 policies in a single call.

    Policies are processed together for efficiency; order is preserved.
    """
    model = _load_model()
    df = _batch_to_df(request.policies)

    try:
        preds = model.predict(df)
    except Exception as e:
        logger.error(f"Batch prediction failed: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

    responses = [_make_response(preds.iloc[i]) for i in range(len(preds))]
    return BatchPredictionResponse(predictions=responses, count=len(responses))


@app.get("/model/info", tags=["model"])
def model_info() -> dict[str, Any]:
    """Return model metadata and key training statistics."""
    model = _load_model()
    return {
        "version":         VERSION,
        "frequency_model": {
            "family":          "Poisson",
            "link":            "log",
            "aic":             round(model.frequency_model.aic, 1),
            "deviance_ratio":  round(model.frequency_model.deviance_ratio, 4),
        },
        "severity_model": {
            "family":          "Gamma",
            "link":            "log",
            "aic":             round(model.severity_model.aic, 1),
            "deviance_ratio":  round(model.severity_model.deviance_ratio, 4),
            "dispersion":      round(model.severity_model.dispersion_, 4),
        },
        "n_features": len(model.frequency_model.feature_names_),
        "features":   model.frequency_model.feature_names_,
    }
