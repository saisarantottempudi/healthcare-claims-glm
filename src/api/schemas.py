"""Pydantic request/response schemas for the prediction API."""

from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class PredictionRequest(BaseModel):
    """
    Single policyholder prediction request.

    All fields mirror the raw data schema so the API can be called
    directly from the front-end quoting system.
    """
    age: int = Field(..., ge=18, le=100, description="Policyholder age in years")
    gender: str = Field(..., description="Male | Female | Non-binary")
    bmi: float = Field(..., ge=10.0, le=70.0, description="Body Mass Index")
    smoker: bool = Field(..., description="Current smoker status")
    chronic_conditions: bool = Field(..., description="Presence of chronic health conditions")
    num_dependants: int = Field(..., ge=0, le=10, description="Number of dependants on policy")
    region: str = Field(..., description="UK region of residence")
    plan_type: str = Field(..., description="Bronze | Silver | Gold | Platinum")
    years_as_customer: int = Field(..., ge=0, le=50)
    exposure_years: float = Field(1.0, ge=0.01, le=1.0, description="Policy exposure in years")

    @field_validator("gender")
    @classmethod
    def validate_gender(cls, v):
        valid = {"Male", "Female", "Non-binary"}
        if v not in valid:
            raise ValueError(f"gender must be one of {valid}")
        return v

    @field_validator("region")
    @classmethod
    def validate_region(cls, v):
        valid = {"London", "South East", "Midlands", "North England", "Scotland", "Wales"}
        if v not in valid:
            raise ValueError(f"region must be one of {valid}")
        return v

    @field_validator("plan_type")
    @classmethod
    def validate_plan_type(cls, v):
        valid = {"Bronze", "Silver", "Gold", "Platinum"}
        if v not in valid:
            raise ValueError(f"plan_type must be one of {valid}")
        return v

    model_config = {
        "json_schema_extra": {
            "example": {
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
        }
    }


class PredictionResponse(BaseModel):
    """Structured prediction output."""
    pred_frequency:    float = Field(..., description="Expected annual claim count")
    pred_severity:     float = Field(..., description="Expected cost per claim (£)")
    pred_pure_premium: float = Field(..., description="Expected annual claim cost (£)")
    risk_band:         str   = Field(..., description="LOW | MEDIUM | HIGH | VERY_HIGH")
    model_version:     str   = Field(..., description="Model version tag")


class BatchPredictionRequest(BaseModel):
    policies: list[PredictionRequest] = Field(..., min_length=1, max_length=1000)


class BatchPredictionResponse(BaseModel):
    predictions: list[PredictionResponse]
    count: int


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    version: str
