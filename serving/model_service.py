"""Model loading and scoring, decoupled from the web framework.

The FastAPI app, the batch CLI and the benchmark all use this one class, so the
online and batch paths score identically. Artifacts are loaded once at start-up.
"""
import json
import logging
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, RAW_FEATURES
from src.train import rule_score

log = logging.getLogger("model_service")


class ModelService:
    def __init__(self, artifact_dir: Path = ARTIFACT_DIR):
        self.artifact_dir = Path(artifact_dir)
        self.model = None
        self.meta: dict = {}
        self.load_error: str | None = None
        self.loaded_at: float | None = None
        self._clf = None
        self._threads = None

    # ---------------------------------------------------------------- lifecycle
    def load(self) -> "ModelService":
        try:
            self.meta = json.loads((self.artifact_dir / "model_metadata.json").read_text())
            self.model = joblib.load(self.artifact_dir / "model.joblib")
            # Online scoring is dominated by per-call overhead, not compute: one thread per call
            # avoids LightGBM's thread-pool start-up and lets the server parallelise across requests.
            self._clf = getattr(self.model, "named_steps", {}).get("clf")
            self._threads = None
            self._set_threads(1)
            # smoke-test the artifact on a canonical record before declaring ready
            self.model.predict_proba(pd.DataFrame([dict.fromkeys(RAW_FEATURES, 1.0)]))
            self.loaded_at = time.time(); self.load_error = None
            log.info("Loaded model %s (threshold %.2f)", self.version, self.threshold)
        except Exception as exc:  # artifact missing/corrupt -> service stays up in fallback mode
            self.model = None; self.load_error = f"{type(exc).__name__}: {exc}"
            log.exception("Model load failed; serving heuristic fallback")
        return self

    @property
    def ready(self) -> bool:
        return self.model is not None

    @property
    def version(self) -> str:
        return self.meta.get("model_version", "unavailable")

    @property
    def threshold(self) -> float:
        return float(self.meta.get("decision_threshold", 0.16))

    def _set_threads(self, n: int):
        """1 thread for small online calls (no pool start-up cost); all cores for large batch files."""
        if self._clf is not None and self._threads != n:
            self._clf.set_params(n_jobs=n); self._threads = n

    # ---------------------------------------------------------------- scoring
    def score_frame(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, bool]:
        """Return (probability_of_default, decision_flag, used_fallback) for a batch of records."""
        X = X.reindex(columns=RAW_FEATURES).astype(float)
        if self.ready:
            try:
                self._set_threads(1 if len(X) < 5000 else -1)
                p = self.model.predict_proba(X)[:, 1]
                if np.isfinite(p).all():
                    return p, p >= self.threshold, False
                raise ValueError("non-finite model output")
            except Exception:
                log.exception("Model scoring failed; using heuristic fallback")
        p = rule_score(X)
        thr = float(self.meta.get("fallback_rule_threshold", 0.21))
        return p, p >= thr, True

    @staticmethod
    def decision_label(flag: bool) -> str:
        return "DECLINE_OR_REVIEW" if flag else "APPROVE"

    @staticmethod
    def risk_band(p: float) -> str:
        return "LOW" if p < 0.05 else "MEDIUM" if p < 0.16 else "HIGH" if p < 0.40 else "VERY_HIGH"
