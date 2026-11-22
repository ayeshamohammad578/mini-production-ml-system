"""Pydantic input/output contracts for the REST API.

Validation rules mirror what the model can safely accept:
  * required credit-history fields must be present and non-negative;
  * monthly_income and dependents may be missing (None) - they are missing for
    ~20% / ~3% of real applicants and the pipeline imputes them;
  * physically impossible values are rejected with HTTP 422, while large-but-
    possible outliers are accepted and capped by the trained pipeline.
"""
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field


class Applicant(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={"example": {
        "applicant_id": "A-10023", "revol_util": 0.45, "age": 41, "late_30_59": 0, "late_60_89": 0, "late_90": 0,
        "debt_ratio": 0.35, "monthly_income": 6200, "open_lines": 8, "real_estate_lines": 1, "dependents": 2}})

    applicant_id: Optional[str] = Field(None, max_length=64, description="Caller's record ID, echoed back")
    revol_util: float = Field(..., ge=0, le=100_000, description="Revolving balance / credit limit")
    age: int = Field(..., ge=18, le=110, description="Age in years")
    late_30_59: int = Field(..., ge=0, le=98, description="Times 30-59 days late (96/98 = bureau code)")
    late_60_89: int = Field(..., ge=0, le=98, description="Times 60-89 days late")
    late_90: int = Field(..., ge=0, le=98, description="Times 90+ days late")
    debt_ratio: float = Field(..., ge=0, le=1_000_000, description="Monthly debt / gross monthly income")
    monthly_income: Optional[float] = Field(None, ge=0, le=10_000_000, description="Gross monthly income; may be missing")
    open_lines: int = Field(..., ge=0, le=100, description="Open credit lines and loans")
    real_estate_lines: int = Field(..., ge=0, le=100, description="Mortgage / real-estate lines")
    dependents: Optional[int] = Field(None, ge=0, le=30, description="Number of dependents; may be missing")


class BatchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    records: list[Applicant] = Field(..., min_length=1, max_length=1000)


class Prediction(BaseModel):
    applicant_id: Optional[str]
    probability_of_default: float
    decision: Literal["APPROVE", "DECLINE_OR_REVIEW"]
    risk_band: Literal["LOW", "MEDIUM", "HIGH", "VERY_HIGH"]
    threshold: float
    model_version: str
    fallback_used: bool


class BatchResponse(BaseModel):
    predictions: list[Prediction]
    n_records: int
    latency_ms: float


class Health(BaseModel):
    status: Literal["ok", "degraded"]
    model_loaded: bool
    model_version: str
    detail: Optional[str] = None


class ErrorResponse(BaseModel):
    error: str
    detail: list | str
    request_id: str
