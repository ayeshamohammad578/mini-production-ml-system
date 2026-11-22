"""Batch CLI behaviour and drift-metric sanity checks."""
import numpy as np
import pandas as pd

from serving.predict_batch import run
from src.monitoring import psi, status


def test_batch_cli_scores_and_rejects(tmp_path, synthetic_frame):
    df = synthetic_frame.copy()
    df.loc[3, "age"] = 7                      # invalid -> REJECTED
    df.loc[4, "open_lines"] = None            # required missing -> REJECTED
    inp, out = tmp_path / "in.csv", tmp_path / "out.csv"
    df.to_csv(inp, index=False)
    stats = run(inp, out, "applicant_id")
    res = pd.read_csv(out)
    assert len(res) == len(df) and list(res["applicant_id"]) == list(df["applicant_id"])
    assert set(res.loc[[3, 4], "status"]) == {"REJECTED"}
    scored = res[res.status == "SCORED"]
    assert scored["probability_of_default"].between(0, 1).all()
    assert stats["scored"] + stats["rejected"] == len(df)


def test_psi_zero_for_identical_and_large_for_shift():
    rng = np.random.default_rng(0)
    ref = rng.normal(0, 1, 5000)
    edges = np.quantile(ref, np.linspace(0, 1, 11))
    assert psi(ref, ref, edges) < 1e-6
    assert psi(ref, rng.normal(1.0, 1, 5000), edges) > 0.25


def test_status_levels():
    assert status(0.30, 0.5, 0.01) == "ALERT"
    assert status(0.12, 0.5, 0.01) == "WARN"
    assert status(0.01, 0.001, 0.10) == "WARN"
    assert status(0.01, 0.001, 0.01) == "OK"      # significant but immaterial KS shift is ignored
