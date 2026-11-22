"""One-time evaluation of every model on the untouched TEST partition.

Run:  python -m src.evaluate
Writes reports/test_metrics.json and figures in reports/figures/.
"""
import json
import logging

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import (average_precision_score, roc_auc_score, f1_score, precision_recall_curve,
                             confusion_matrix, brier_score_loss, precision_score, recall_score)

from src.config import ARTIFACT_DIR, DATA_DIR, REPORT_DIR, FIG_DIR, RAW_FEATURES, TARGET, COST_FN, COST_FP
from src.train import rule_score, expected_cost

log = logging.getLogger("evaluate")


def recall_at_precision(y, p, min_precision):
    prec, rec, _ = precision_recall_curve(y, p)
    ok = prec >= min_precision
    return float(rec[ok].max()) if ok.any() else 0.0


def metric_row(y, p, thr):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    return {"PR-AUC": average_precision_score(y, p), "ROC-AUC": roc_auc_score(y, p),
            "F1-macro": f1_score(y, pred, average="macro"), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred), "Recall@Precision>=0.50": recall_at_precision(y, p, 0.50),
            "Brier": brier_score_loss(y, np.clip(p, 0, 1)), "Threshold": thr,
            "Cost per 1,000 applicants": 1000 * expected_cost(y, p, thr), "Flag rate": float(pred.mean()),
            "TP": int(tp), "FP": int(fp), "FN": int(fn), "TN": int(tn)}


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    test = pd.read_csv(DATA_DIR / "test.csv")
    X, y = test[RAW_FEATURES], test[TARGET].to_numpy()
    meta = json.loads((ARTIFACT_DIR / "model_metadata.json").read_text())
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    lr = joblib.load(ARTIFACT_DIR / "baseline_logreg.joblib")

    probs = {"B0 Prior (dummy)": np.full(len(y), meta["train_default_rate"]),
             "B1 Heuristic rule": rule_score(X),
             "B2 Logistic regression": lr.predict_proba(X)[:, 1],
             f"LightGBM (production: {meta['selected_model'].split(' - ')[1]})": model.predict_proba(X)[:, 1]}
    thrs = [0.5, meta["fallback_rule_threshold"], meta["baseline_logreg_threshold"], meta["decision_threshold"]]
    rows = {n: metric_row(y, p, t) for (n, p), t in zip(probs.items(), thrs)}
    prod = list(probs)[-1]
    # Same production scores at the naive 0.5 threshold, to show the value of cost-based thresholding
    rows["LightGBM @ naive 0.50 threshold"] = metric_row(y, probs[prod], 0.5)
    res = pd.DataFrame(rows).T
    res.to_csv(REPORT_DIR / "test_metrics.csv")
    (REPORT_DIR / "test_metrics.json").write_text(json.dumps({"n_test": int(len(y)), "default_rate": float(y.mean()),
                                                              "cost_fn": COST_FN, "cost_fp": COST_FP,
                                                              "results": rows}, indent=2, default=float))
    log.info("\n%s", res[["PR-AUC", "ROC-AUC", "F1-macro", "Recall", "Precision", "Cost per 1,000 applicants"]].round(4))

    # --- Figures ----------------------------------------------------------------------------------
    cols = ["#9E9E9E", "#FF8F00", "#1F77B4", "#2E7D32"]
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
    for (n, p), c in zip(probs.items(), cols):
        if n.startswith("B0"):
            ax[0].axhline(y.mean(), color=c, ls="--", label=f"Prior / no skill ({y.mean():.3f})"); continue
        pr, rc, _ = precision_recall_curve(y, p)
        ax[0].plot(rc, pr, color=c, label=f"{n.split(' (')[0]} ({average_precision_score(y, p):.3f})")
    ax[0].set(title="Precision-recall curve (test)", xlabel="Recall", ylabel="Precision"); ax[0].legend(fontsize=8)
    grid = np.round(np.arange(0.02, 0.81, 0.01), 2)
    for (n, p), c in zip(list(probs.items())[1:], cols[1:]):
        ax[1].plot(grid, [1000 * expected_cost(y, p, t) for t in grid], color=c, label=n.split(" (")[0])
    ax[1].axvline(meta["decision_threshold"], color="#2E7D32", ls=":", lw=1)
    ax[1].set(title=f"Expected cost per 1,000 applicants (FN={COST_FN:g}, FP={COST_FP:g})", xlabel="Decision threshold",
              ylabel="Cost units"); ax[1].legend(fontsize=8)
    for (n, p), c in zip(list(probs.items())[2:], cols[2:]):
        fr, mp = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax[2].plot(mp, fr, "o-", color=c, ms=4, label=f"{n.split(' (')[0]} (Brier {brier_score_loss(y, p):.4f})")
    ax[2].plot([0, 1], [0, 1], "k--", lw=0.8); ax[2].set(title="Calibration (test)", xlabel="Predicted PD", ylabel="Observed default rate",
                                                     xlim=(0, 0.8), ylim=(0, 0.8)); ax[2].legend(fontsize=8)
    plt.tight_layout(); plt.savefig(FIG_DIR / "test_curves.png", dpi=160); plt.close()

    r = rows[prod]
    cm = np.array([[r["TN"], r["FP"]], [r["FN"], r["TP"]]])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    ax.imshow(cm, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, f"{cm[i, j]:,}", ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=12)
    ax.set_xticks([0, 1], ["Approve", "Decline / review"]); ax.set_yticks([0, 1], ["Repaid", "Defaulted"])
    ax.set_title(f"Production model, test set @ threshold {meta['decision_threshold']:.2f}", fontsize=10)
    plt.tight_layout(); plt.savefig(FIG_DIR / "confusion_matrix.png", dpi=160); plt.close()

    imp = pd.Series(model.named_steps["clf"].booster_.feature_importance("gain"),
                    index=model.named_steps["features"].get_feature_names_out()).sort_values()
    fig, ax = plt.subplots(figsize=(6, 4.6))
    (imp / imp.sum() * 100).plot.barh(ax=ax, color="#1F4E79"); ax.set_title("Feature importance (% of total gain)")
    plt.tight_layout(); plt.savefig(FIG_DIR / "feature_importance.png", dpi=160); plt.close()
    log.info("Figures written to %s", FIG_DIR)


if __name__ == "__main__":
    main()
