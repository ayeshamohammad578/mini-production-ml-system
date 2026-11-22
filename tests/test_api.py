"""API contract: health, prediction schema, validation errors, fallback behaviour."""
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from serving import app as app_module
from serving.model_service import ModelService
from serving.schemas import Applicant


@pytest.fixture(scope="module")
def client():
    with TestClient(app_module.app) as c:
        yield c


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["model_loaded"] is True


def test_predict_valid(client, valid_record):
    r = client.post("/predict", json=valid_record)
    assert r.status_code == 200
    body = r.json()
    assert 0 <= body["probability_of_default"] <= 1
    assert body["decision"] in {"APPROVE", "DECLINE_OR_REVIEW"}
    assert body["applicant_id"] == "T-1" and body["fallback_used"] is False
    assert "X-Request-ID" in r.headers and r.headers["X-Fallback"] == "false"


def test_high_risk_profile_scores_higher(client, valid_record):
    risky = valid_record | {"late_90": 3, "late_60_89": 2, "revol_util": 1.4, "age": 23}
    p_safe = client.post("/predict", json=valid_record).json()["probability_of_default"]
    p_risky = client.post("/predict", json=risky).json()["probability_of_default"]
    assert p_risky > p_safe


@pytest.mark.parametrize("patch", [{"monthly_income": None, "dependents": None}, {"revol_util": 50000.0},
                                   {"late_90": 98, "late_60_89": 98, "late_30_59": 98}])
def test_edge_cases_are_scored(client, valid_record, patch):
    assert client.post("/predict", json=valid_record | patch).status_code == 200


@pytest.mark.parametrize("patch", [{"age": 7}, {"debt_ratio": -1.0}, {"open_lines": "five"}, {"credit_score": 700}])
def test_invalid_payloads_rejected(client, valid_record, patch):
    r = client.post("/predict", json=valid_record | patch)
    assert r.status_code == 422 and r.json()["error"] == "invalid_payload"


def test_missing_required_field_rejected(client, valid_record):
    body = {k: v for k, v in valid_record.items() if k != "age"}
    r = client.post("/predict", json=body)
    assert r.status_code == 422 and r.json()["detail"][0]["field"] == "age"


def test_batch_endpoint(client, valid_record):
    r = client.post("/predict/batch", json={"records": [valid_record] * 5})
    assert r.status_code == 200 and r.json()["n_records"] == 5
    assert client.post("/predict/batch", json={"records": []}).status_code == 422


def test_schema_unit(valid_record):
    Applicant(**valid_record)
    with pytest.raises(ValidationError):
        Applicant(**(valid_record | {"late_90": -1}))


def test_fallback_when_artifacts_missing(tmp_path, valid_record, synthetic_frame):
    svc = ModelService(artifact_dir=tmp_path).load()          # empty dir -> load fails gracefully
    assert not svc.ready and svc.load_error
    from src.config import RAW_FEATURES
    p, flag, fb = svc.score_frame(synthetic_frame[RAW_FEATURES])
    assert fb is True and ((p >= 0) & (p <= 1)).all()


def test_health_reports_degraded_in_fallback(tmp_path, valid_record):
    original = app_module.service
    app_module.service = ModelService(artifact_dir=tmp_path).load()
    try:
        c = TestClient(app_module.app)
        assert c.get("/health").status_code == 503
        r = c.post("/predict", json=valid_record)
        assert r.status_code == 200 and r.json()["fallback_used"] is True and r.headers["X-Fallback"] == "true"
    finally:
        app_module.service = original
