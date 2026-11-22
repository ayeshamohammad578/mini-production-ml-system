import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import RAW_FEATURES  # noqa: E402


@pytest.fixture
def valid_record():
    return {"applicant_id": "T-1", "revol_util": 0.45, "age": 41, "late_30_59": 0, "late_60_89": 0, "late_90": 0,
            "debt_ratio": 0.35, "monthly_income": 6200.0, "open_lines": 8, "real_estate_lines": 1, "dependents": 2}


@pytest.fixture
def synthetic_frame():
    """Synthetic applicants covering normal values and every edge case the pipeline must survive."""
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({
        "revol_util": rng.gamma(1.0, 0.4, n), "age": rng.integers(18, 95, n).astype(float),
        "late_30_59": rng.poisson(0.3, n).astype(float), "late_60_89": rng.poisson(0.1, n).astype(float),
        "late_90": rng.poisson(0.1, n).astype(float), "debt_ratio": rng.gamma(1.0, 0.5, n),
        "monthly_income": rng.normal(6000, 2500, n).clip(0), "open_lines": rng.integers(0, 30, n).astype(float),
        "real_estate_lines": rng.integers(0, 4, n).astype(float), "dependents": rng.integers(0, 5, n).astype(float),
    })
    df.loc[::7, "monthly_income"] = np.nan            # missing income
    df.loc[::11, "dependents"] = np.nan               # missing dependents
    df.loc[5, ["late_30_59", "late_60_89", "late_90"]] = 98   # bureau code
    df.loc[6, "revol_util"] = 50_000                  # extreme outlier
    df.loc[7, "debt_ratio"] = 300_000
    df.loc[8, "age"] = 0                              # entry error
    df.loc[9, RAW_FEATURES] = np.nan                  # entirely empty row
    df["default"] = (rng.random(n) < 0.07).astype(int)
    df.insert(0, "applicant_id", np.arange(n))
    return df
