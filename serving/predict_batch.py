"""Batch scoring CLI: score a CSV of unseen applicants.

Usage:
  python -m serving.predict_batch --input data/test.csv --output reports/scored_test.csv --id-col applicant_id

Behaviour
  * Validates every row with the same rules as the API; invalid rows are not
    scored but written with status=REJECTED and a reason (the job never crashes
    on bad rows).
  * Scores valid rows in vectorised chunks and logs throughput (records/sec).
  * Output keeps the record ID: id, probability_of_default, decision, risk_band,
    model_version, fallback_used, status, reason.
"""
import argparse
import logging
import sys
import time

import numpy as np
import pandas as pd

from serving.model_service import ModelService
from src.config import RAW_FEATURES

log = logging.getLogger("predict_batch")

RULES = {  # column: (min, max, required)
    "revol_util": (0, 100_000, True), "age": (18, 110, True), "late_30_59": (0, 98, True),
    "late_60_89": (0, 98, True), "late_90": (0, 98, True), "debt_ratio": (0, 1_000_000, True),
    "monthly_income": (0, 10_000_000, False), "open_lines": (0, 100, True),
    "real_estate_lines": (0, 100, True), "dependents": (0, 30, False),
}


def validate(df: pd.DataFrame) -> pd.Series:
    """Vectorised row validation; returns '' for valid rows or a reason string."""
    reason = pd.Series("", index=df.index, dtype=object)
    for col, (lo, hi, required) in RULES.items():
        if col not in df.columns:
            if required:
                reason[:] = reason + f"missing column {col};"
            continue
        v = pd.to_numeric(df[col], errors="coerce")
        non_numeric = v.isna() & df[col].notna()
        reason[non_numeric] += f"{col} not numeric;"
        if required:
            reason[df[col].isna()] += f"{col} missing;"
        reason[(v < lo) | (v > hi)] += f"{col} out of range [{lo},{hi}];"
    return reason


def run(input_path, output_path, id_col, chunk_size=50_000):
    t_start = time.perf_counter()
    service = ModelService().load()
    df = pd.read_csv(input_path)
    if id_col not in df.columns:
        df[id_col] = np.arange(len(df))
        log.warning("ID column '%s' not found; using row numbers", id_col)
    reason = validate(df)
    valid = reason == ""
    out = pd.DataFrame({id_col: df[id_col], "probability_of_default": np.nan, "decision": None, "risk_band": None,
                        "model_version": service.version, "fallback_used": False,
                        "status": np.where(valid, "SCORED", "REJECTED"), "reason": reason.str.rstrip(";")})
    t_score = time.perf_counter(); fb_any = False
    X = df.loc[valid, RAW_FEATURES].apply(pd.to_numeric, errors="coerce")
    for start in range(0, len(X), chunk_size):
        part = X.iloc[start:start + chunk_size]
        p, flag, fb = service.score_frame(part)
        fb_any |= fb
        out.loc[part.index, "probability_of_default"] = np.round(p, 6)
        out.loc[part.index, "decision"] = np.where(flag, "DECLINE_OR_REVIEW", "APPROVE")
        out.loc[part.index, "risk_band"] = [service.risk_band(x) for x in p]
        out.loc[part.index, "fallback_used"] = fb
    score_s = time.perf_counter() - t_score
    out.to_csv(output_path, index=False)
    total_s = time.perf_counter() - t_start
    n_ok = int(valid.sum())
    log.info("Rows read %d | scored %d | rejected %d | fallback=%s", len(df), n_ok, len(df) - n_ok, fb_any)
    log.info("Scoring throughput: %.0f records/sec (%.3fs); end-to-end incl. I/O: %.0f records/sec (%.3fs)",
             n_ok / max(score_s, 1e-9), score_s, len(df) / max(total_s, 1e-9), total_s)
    return {"rows": len(df), "scored": n_ok, "rejected": len(df) - n_ok, "score_seconds": score_s,
            "total_seconds": total_s, "records_per_sec_scoring": n_ok / max(score_s, 1e-9),
            "records_per_sec_end_to_end": len(df) / max(total_s, 1e-9), "fallback_used": fb_any}


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser(description="Batch credit-default scoring")
    ap.add_argument("--input", required=True); ap.add_argument("--output", required=True)
    ap.add_argument("--id-col", default="applicant_id"); ap.add_argument("--chunk-size", type=int, default=50_000)
    a = ap.parse_args(argv)
    try:
        run(a.input, a.output, a.id_col, a.chunk_size)
    except FileNotFoundError as e:
        log.error("Input not found: %s", e); return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
