"""FastAPI online-scoring service.

Run:   uvicorn serving.app:app --host 0.0.0.0 --port 8000
Docs:  http://localhost:8000/docs

Endpoints
  GET  /health          liveness + readiness (model loaded?)
  GET  /model-info      version, threshold, features, validation metrics
  POST /predict         score one applicant
  POST /predict/batch   score up to 1,000 applicants in one call
  GET  /metrics         in-process operational counters and rolling latency percentiles
Error contract: 422 for invalid payloads (field-level detail), 500 for unexpected errors,
every response carries an X-Request-ID header. If the model cannot score, a transparent
heuristic fallback is served and flagged (fallback_used=true, X-Fallback: true).
"""
import logging
import time
import uuid
from collections import deque
from contextlib import asynccontextmanager

import numpy as np
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from serving.model_service import ModelService
from serving.schemas import Applicant, BatchRequest, BatchResponse, Prediction, Health

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("api")

service = ModelService()
STATS = {"requests": 0, "errors_4xx": 0, "errors_5xx": 0, "fallbacks": 0, "records_scored": 0}
LATENCIES = deque(maxlen=5000)


@asynccontextmanager
async def lifespan(app: FastAPI):
    service.load()                       # load artifacts once, not per request
    yield


app = FastAPI(title="Credit Default Risk Scoring API", version="1.0.0", lifespan=lifespan)


@app.middleware("http")
async def telemetry(request: Request, call_next):
    rid = request.headers.get("X-Request-ID", uuid.uuid4().hex[:12])
    request.state.rid = rid
    t0 = time.perf_counter()
    response = await call_next(request)
    ms = (time.perf_counter() - t0) * 1000
    STATS["requests"] += 1
    if 400 <= response.status_code < 500: STATS["errors_4xx"] += 1
    if response.status_code >= 500: STATS["errors_5xx"] += 1
    if request.url.path.startswith("/predict"): LATENCIES.append(ms)
    response.headers["X-Request-ID"] = rid
    response.headers["X-Latency-ms"] = f"{ms:.2f}"
    return response


@app.exception_handler(RequestValidationError)
async def validation_error(request: Request, exc: RequestValidationError):
    detail = [{"field": ".".join(str(x) for x in e["loc"][1:]), "message": e["msg"], "input": e.get("input")}
              for e in exc.errors()]
    log.warning("Rejected payload %s: %s", getattr(request.state, "rid", "-"), detail)
    return JSONResponse(status_code=422, content={"error": "invalid_payload", "detail": detail,
                                                  "request_id": getattr(request.state, "rid", "-")})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("Unhandled error")
    return JSONResponse(status_code=500, content={"error": "internal_error", "detail": "unexpected server error",
                                                  "request_id": getattr(request.state, "rid", "-")})


def _score(records: list[Applicant]) -> tuple[list[Prediction], bool]:
    X = pd.DataFrame([r.model_dump(exclude={"applicant_id"}) for r in records])
    p, flag, fb = service.score_frame(X)
    if fb: STATS["fallbacks"] += len(records)
    STATS["records_scored"] += len(records)
    thr = service.threshold if not fb else float(service.meta.get("fallback_rule_threshold", 0.21))
    preds = [Prediction(applicant_id=r.applicant_id, probability_of_default=round(float(pi), 6),
                        decision=service.decision_label(bool(fi)), risk_band=service.risk_band(float(pi)),
                        threshold=thr, model_version=service.version if not fb else "heuristic-fallback",
                        fallback_used=fb) for r, pi, fi in zip(records, p, flag)]
    return preds, fb


@app.get("/health", response_model=Health)
def health():
    if service.ready:
        return Health(status="ok", model_loaded=True, model_version=service.version)
    return JSONResponse(status_code=503, content=Health(status="degraded", model_loaded=False, model_version="heuristic-fallback",
                                                        detail=service.load_error).model_dump())


@app.get("/model-info")
def model_info():
    keep = ["model_version", "trained_at_utc", "selected_model", "decision_threshold", "input_features",
            "cost_fn", "cost_fp", "train_rows", "train_default_rate", "selected_params"]
    return {k: service.meta.get(k) for k in keep} | {"ready": service.ready}


@app.post("/predict", response_model=Prediction)
def predict(applicant: Applicant):
    preds, fb = _score([applicant])
    return JSONResponse(content=preds[0].model_dump(), headers={"X-Fallback": str(fb).lower()})


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(req: BatchRequest):
    t0 = time.perf_counter()
    preds, fb = _score(req.records)
    body = BatchResponse(predictions=preds, n_records=len(preds), latency_ms=round((time.perf_counter() - t0) * 1000, 3))
    return JSONResponse(content=body.model_dump(), headers={"X-Fallback": str(fb).lower()})


@app.get("/metrics")
def metrics():
    lat = np.array(LATENCIES) if LATENCIES else np.array([np.nan])
    return STATS | {"latency_ms_p50": float(np.nanpercentile(lat, 50)), "latency_ms_p95": float(np.nanpercentile(lat, 95)),
                    "latency_ms_p99": float(np.nanpercentile(lat, 99)), "window": len(LATENCIES),
                    "model_version": service.version, "model_loaded": service.ready}
