"""Train baselines + LightGBM, handle imbalance, pick the decision threshold, export artifacts.

Run:  python -m src.train          (after python -m src.data_pipeline)

Model selection and threshold tuning use the VALIDATION partition only.
The TEST partition is never read here (see src/evaluate.py).
"""
import json
import logging
import time
from datetime import datetime, timezone

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score, f1_score, brier_score_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (ARTIFACT_DIR, DATA_DIR, REPORT_DIR, RAW_FEATURES, TARGET, SEED, COST_FN, COST_FP)
from src.features import CreditFeatureBuilder, apply_rules, MODEL_FEATURES

log = logging.getLogger("train")
MODEL_VERSION = "credit-default-lgbm-1.0.0"


def rule_score(X: pd.DataFrame) -> np.ndarray:
    """Interpretable heuristic baseline (also the serving fallback).

    Risk points: any 90+ day delinquency (3), any 60-89 (2), any 30-59 (1),
    utilisation above 100% (2), above 75% (1), age under 30 (1). Scaled to 0-1.
    """
    d = apply_rules(X)
    pts = (3 * (d["late_90"].fillna(0) > 0) + 2 * (d["late_60_89"].fillna(0) > 0) + 1 * (d["late_30_59"].fillna(0) > 0)
           + 2 * (d["revol_util"].fillna(0) > 1) + 1 * (d["revol_util"].fillna(0) > 0.75) + 1 * (d["age"].fillna(50) < 30)
           + 3 * d["late_code_flag"])
    return (pts / 10).clip(0, 1).to_numpy(dtype=float)


def expected_cost(y, p, thr):
    pred = p >= thr
    fn = ((~pred) & (y == 1)).sum()
    fp = (pred & (y == 0)).sum()
    return (COST_FN * fn + COST_FP * fp) / len(y)


def best_threshold(y, p):
    grid = np.round(np.arange(0.01, 0.91, 0.01), 2)
    costs = np.array([expected_cost(y, p, t) for t in grid])
    return float(grid[costs.argmin()]), float(costs.min())


def scores(y, p, thr):
    pred = (p >= thr).astype(int)
    return {"pr_auc": average_precision_score(y, p), "roc_auc": roc_auc_score(y, p),
            "f1_macro": f1_score(y, pred, average="macro"), "brier": brier_score_loss(y, np.clip(p, 0, 1)),
            "threshold": thr, "cost_per_applicant": expected_cost(y, p, thr)}


def reference_profile(train_X: pd.DataFrame, train_scores: np.ndarray) -> dict:
    """Training-time distribution snapshot stored with the model for drift monitoring (PSI/KS)."""
    d = apply_rules(train_X)
    prof = {}
    for c in RAW_FEATURES + ["total_late"]:
        s = d[c].dropna()
        edges = np.unique(np.quantile(s, np.linspace(0, 1, 11)))
        prof[c] = {"edges": edges.tolist(), "missing_rate": float(d[c].isna().mean()),
                   "sample": s.sample(min(5000, len(s)), random_state=SEED).round(6).tolist()}
    prof["_score"] = {"edges": np.unique(np.quantile(train_scores, np.linspace(0, 1, 11))).tolist(),
                      "sample": pd.Series(train_scores).sample(5000, random_state=SEED).round(6).tolist()}
    return prof


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    train = pd.read_csv(DATA_DIR / "train.csv")
    val = pd.read_csv(DATA_DIR / "val.csv")
    Xtr, ytr = train[RAW_FEATURES], train[TARGET].to_numpy()
    Xva, yva = val[RAW_FEATURES], val[TARGET].to_numpy()
    pos_weight = (ytr == 0).sum() / (ytr == 1).sum()
    log.info("Train %d rows (%.2f%% default) | Val %d rows | neg/pos = %.1f", len(ytr), 100 * ytr.mean(), len(yva), pos_weight)

    results, candidates = {}, {}

    # ---- Baseline 0: prior (no skill) --------------------------------------------------------
    dummy = DummyClassifier(strategy="prior").fit(Xtr, ytr)
    p = dummy.predict_proba(Xva)[:, 1]
    results["B0 Prior (dummy)"] = scores(yva, p, 0.5) | {"fit_s": 0.0}

    # ---- Baseline 1: interpretable rule (also the production fallback) ----------------------
    p = rule_score(Xva); thr, _ = best_threshold(yva, p)
    results["B1 Heuristic rule"] = scores(yva, p, thr) | {"fit_s": 0.0}

    # ---- Baseline 2: logistic regression on the same leak-free features ---------------------
    lr = Pipeline([("features", CreditFeatureBuilder()), ("scale", StandardScaler()),
                   ("clf", LogisticRegression(max_iter=2000))])
    t0 = time.time(); lr.fit(Xtr, ytr); fit = time.time() - t0
    p = lr.predict_proba(Xva)[:, 1]; thr, _ = best_threshold(yva, p)
    results["B2 Logistic regression"] = scores(yva, p, thr) | {"fit_s": fit}
    candidates["B2 Logistic regression"] = (lr, thr)

    # ---- LightGBM: imbalance handled two ways, small validation grid ------------------------
    grid = [dict(num_leaves=nl, min_child_samples=mc, learning_rate=0.03, n_estimators=400,
                 subsample=0.85, subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0)
            for nl in (15, 31) for mc in (50, 200)]
    for strategy, weighted in (("threshold tuning (unweighted)", False), ("class-weighted loss", True)):
        best = None
        for params in grid:
            clf = LGBMClassifier(random_state=SEED, verbose=-1, n_jobs=-1,
                                 scale_pos_weight=pos_weight if weighted else 1.0, **params)
            pipe = Pipeline([("features", CreditFeatureBuilder()), ("clf", clf)])
            t0 = time.time(); pipe.fit(Xtr, ytr); fit = time.time() - t0
            p = pipe.predict_proba(Xva)[:, 1]; thr, _ = best_threshold(yva, p)
            s = scores(yva, p, thr) | {"fit_s": fit, "params": params}
            if best is None or s["pr_auc"] > best[1]["pr_auc"]:
                best = (pipe, s, thr)
        name = f"LightGBM - {strategy}"
        results[name] = best[1]; candidates[name] = (best[0], best[2])
        log.info("%s: PR-AUC %.4f, ROC-AUC %.4f, cost %.4f @thr %.2f", name, best[1]["pr_auc"], best[1]["roc_auc"],
                 best[1]["cost_per_applicant"], best[2])

    # ---- Select the production model ---------------------------------------------------------
    # Highest validation PR-AUC wins, but differences below PR_AUC_TIE are treated as a tie and
    # the better-calibrated model (lower Brier) is preferred: the API returns a probability of
    # default that downstream pricing/limits consume, so it must be a usable probability.
    PR_AUC_TIE = 0.005
    lgbm_names = [n for n in candidates if n.startswith("LightGBM")]
    top = max(results[n]["pr_auc"] for n in lgbm_names)
    near = [n for n in lgbm_names if top - results[n]["pr_auc"] < PR_AUC_TIE]
    chosen = min(near, key=lambda n: results[n]["brier"])
    model, thr = candidates[chosen]
    log.info("Selected: %s (threshold %.2f)", chosen, thr)

    # ---- Export artifacts ------------------------------------------------------------------------
    ARTIFACT_DIR.mkdir(exist_ok=True)
    joblib.dump(model, ARTIFACT_DIR / "model.joblib")
    joblib.dump(candidates["B2 Logistic regression"][0], ARTIFACT_DIR / "baseline_logreg.joblib")
    train_scores = model.predict_proba(Xtr)[:, 1]
    meta = {
        "model_version": MODEL_VERSION,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "selected_model": chosen,
        "decision_threshold": thr,
        "baseline_logreg_threshold": candidates["B2 Logistic regression"][1],
        "fallback_rule_threshold": results["B1 Heuristic rule"]["threshold"],
        "input_features": RAW_FEATURES,
        "model_features": MODEL_FEATURES,
        "cost_fn": COST_FN, "cost_fp": COST_FP,
        "train_rows": int(len(ytr)), "train_default_rate": float(ytr.mean()),
        "validation_metrics": {k: {m: v for m, v in r.items() if m != "params"} for k, r in results.items()},
        "selected_params": results[chosen].get("params"),
    }
    (ARTIFACT_DIR / "model_metadata.json").write_text(json.dumps(meta, indent=2))
    (ARTIFACT_DIR / "reference_profile.json").write_text(json.dumps(reference_profile(Xtr, train_scores)))
    REPORT_DIR.mkdir(exist_ok=True)
    pd.DataFrame(results).T.drop(columns="params", errors="ignore").to_csv(REPORT_DIR / "validation_results.csv")
    log.info("Artifacts written to %s", ARTIFACT_DIR)


if __name__ == "__main__":
    main()
