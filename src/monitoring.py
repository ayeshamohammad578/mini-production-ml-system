"""Drift monitoring: PSI and two-sample KS against the training reference profile.

Run:  python -m src.monitoring --input data/test.csv                 # healthy check
      python -m src.monitoring --input data/test.csv --simulate-drift  # demo: stressed economy

Library use: compute_drift(df) -> DataFrame with PSI, KS p-value and status per feature.
"""
import argparse
import json
import logging

import joblib
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from src.config import ARTIFACT_DIR, REPORT_DIR, RAW_FEATURES, PSI_WARN, PSI_ALERT, KS_PVALUE_ALERT, KS_STAT_MIN
from src.features import apply_rules

log = logging.getLogger("monitoring")


def psi(expected_sample, actual, edges, eps=1e-4):
    """Population Stability Index over the training-decile bins (open-ended outer bins)."""
    e = np.asarray(expected_sample, dtype=float); a = np.asarray(actual, dtype=float)
    a = a[~np.isnan(a)]
    inner = np.asarray(edges[1:-1], dtype=float)
    e_pct = np.bincount(np.searchsorted(inner, e, side="right"), minlength=len(inner) + 1) / max(len(e), 1)
    a_pct = np.bincount(np.searchsorted(inner, a, side="right"), minlength=len(inner) + 1) / max(len(a), 1)
    e_pct, a_pct = np.clip(e_pct, eps, None), np.clip(a_pct, eps, None)
    return float(np.sum((a_pct - e_pct) * np.log(a_pct / e_pct)))


def status(psi_value, ks_p, ks_stat):
    """ALERT: PSI >= 0.25 (major shift). WARN: PSI >= 0.10, or a KS shift that is both
    statistically significant AND material (D >= KS_STAT_MIN)."""
    if psi_value >= PSI_ALERT:
        return "ALERT"
    if psi_value >= PSI_WARN or (ks_p < KS_PVALUE_ALERT and ks_stat >= KS_STAT_MIN):
        return "WARN"
    return "OK"


def compute_drift(df: pd.DataFrame, scores: np.ndarray | None = None) -> pd.DataFrame:
    ref = json.loads((ARTIFACT_DIR / "reference_profile.json").read_text())
    d = apply_rules(df[RAW_FEATURES])
    rows = []
    for c in RAW_FEATURES + ["total_late"]:
        cur = d[c].to_numpy(dtype=float).round(6)   # same precision as the stored reference sample
        p = psi(ref[c]["sample"], cur, ref[c]["edges"])
        ks = ks_2samp(ref[c]["sample"], cur[~np.isnan(cur)])
        rows.append({"feature": c, "psi": p, "ks_stat": ks.statistic, "ks_pvalue": ks.pvalue,
                     "missing_rate_ref": ref[c]["missing_rate"], "missing_rate_now": float(np.isnan(cur).mean()),
                     "status": status(p, ks.pvalue, ks.statistic)})
    if scores is not None:
        scores = np.round(scores, 6)
        p = psi(ref["_score"]["sample"], scores, ref["_score"]["edges"])
        ks = ks_2samp(ref["_score"]["sample"], scores)
        rows.append({"feature": "MODEL SCORE (prediction drift)", "psi": p, "ks_stat": ks.statistic, "ks_pvalue": ks.pvalue,
                     "missing_rate_ref": 0.0, "missing_rate_now": 0.0, "status": status(p, ks.pvalue, ks.statistic)})
    return pd.DataFrame(rows)


SCENARIOS = {  # income multiplier, utilisation multiplier, share of borrowers with one extra 30-59 day delinquency
    "moderate": (0.85, 1.30, 0.10),   # mild slowdown
    "severe": (0.60, 1.80, 0.30),     # recession-style stress
}


def simulate_stress(df: pd.DataFrame, severity: str = "severe", seed: int = 7) -> pd.DataFrame:
    """Synthetic economic stress used to demonstrate that the monitor escalates correctly."""
    inc, util, late = SCENARIOS[severity]
    rng = np.random.default_rng(seed); s = df.copy()
    s["monthly_income"] = s["monthly_income"] * inc
    s["revol_util"] = s["revol_util"] * util
    bump = rng.random(len(s)) < late
    s.loc[bump, "late_30_59"] = s.loc[bump, "late_30_59"].fillna(0) + 1
    return s


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--simulate-drift", choices=list(SCENARIOS), default=None, help="apply a synthetic stress scenario")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    df = pd.read_csv(a.input)
    if a.simulate_drift:
        df = simulate_stress(df, a.simulate_drift)
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    rep = compute_drift(df, model.predict_proba(df[RAW_FEATURES])[:, 1])
    out = a.out or REPORT_DIR / (f"drift_{a.simulate_drift}.csv" if a.simulate_drift else "drift_baseline.csv")
    rep.to_csv(out, index=False)
    log.info("\n%s", rep[["feature", "psi", "ks_pvalue", "status"]].round(4).to_string(index=False))
    n_alert = (rep.status == "ALERT").sum()
    log.info("Features in ALERT: %d -> %s", n_alert, "TRIGGER RETRAINING REVIEW" if n_alert else "no action")


if __name__ == "__main__":
    main()
