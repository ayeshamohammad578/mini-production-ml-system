"""Leak-free feature engineering as a scikit-learn transformer.

`CreditFeatureBuilder` is placed *inside* the model Pipeline, so:
  * everything it learns (outlier caps, imputation medians) is fitted on the
    training partition only (Pipeline.fit), and
  * the identical, already-fitted transformation is serialized with the model
    and re-applied at validation, test and inference time (Pipeline.predict).
"""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.config import RAW_FEATURES, LATE_COLS, CAP_COLS

ENGINEERED = ["late_code_flag", "income_missing", "total_late", "ever_90_late",
              "monthly_debt", "income_per_member", "non_re_lines", "util_over_1"]
MODEL_FEATURES = RAW_FEATURES + ENGINEERED


def apply_rules(df: pd.DataFrame) -> pd.DataFrame:
    """Stateless cleaning rules (nothing learned from data, safe anywhere).

    * Late-payment counts of 96/98 are system codes, not counts -> NaN + flag.
    * Age below 18 is a data-entry error -> NaN (imputed later).
    * Negative values are impossible for every input -> NaN.
    """
    d = df.reindex(columns=RAW_FEATURES).astype(float).copy()
    d[d < 0] = np.nan
    codes = d[LATE_COLS] >= 96
    d["late_code_flag"] = codes.any(axis=1).astype(float)
    d[LATE_COLS] = d[LATE_COLS].mask(codes)
    d.loc[d["age"] < 18, "age"] = np.nan
    d["income_missing"] = d["monthly_income"].isna().astype(float)
    d["total_late"] = d[LATE_COLS].sum(axis=1, min_count=1)
    d["ever_90_late"] = (d["late_90"].fillna(0) > 0).astype(float)
    d["monthly_debt"] = d["debt_ratio"] * d["monthly_income"]
    d["income_per_member"] = d["monthly_income"] / (d["dependents"].fillna(0) + 1)
    d["non_re_lines"] = d["open_lines"] - d["real_estate_lines"]
    d["util_over_1"] = (d["revol_util"] > 1).astype(float)
    return d[MODEL_FEATURES]


class CreditFeatureBuilder(BaseEstimator, TransformerMixin):
    """Rules + train-fitted 99th-percentile caps + train-fitted median imputation."""

    def __init__(self, cap_quantile: float = 0.99):
        self.cap_quantile = cap_quantile

    def fit(self, X, y=None):
        d = apply_rules(X)
        self.caps_ = d[CAP_COLS].quantile(self.cap_quantile)
        d[CAP_COLS] = d[CAP_COLS].clip(upper=self.caps_, axis=1)
        self.medians_ = d.median().fillna(0.0)
        self.feature_names_out_ = list(d.columns)
        return self

    def transform(self, X):
        """NumPy implementation of apply_rules + caps + imputation (identical output, ~50x faster
        on single records than the pandas version; equality is asserted in tests/test_pipeline.py)."""
        A = (X[RAW_FEATURES] if isinstance(X, pd.DataFrame) else pd.DataFrame(X, columns=RAW_FEATURES)).to_numpy(dtype=float, copy=True)
        A[A < 0] = np.nan
        i = {c: k for k, c in enumerate(RAW_FEATURES)}
        late = A[:, [i[c] for c in LATE_COLS]]
        codes = late >= 96
        flag = codes.any(axis=1).astype(float)
        late[codes] = np.nan
        A[:, [i[c] for c in LATE_COLS]] = late
        age = A[:, i["age"]]; age[age < 18] = np.nan
        inc = A[:, i["monthly_income"]]
        all_nan = np.isnan(late).all(axis=1)
        total = np.where(all_nan, np.nan, np.nansum(late, axis=1))
        ever90 = (np.nan_to_num(A[:, i["late_90"]], nan=0.0) > 0).astype(float)
        mdebt = A[:, i["debt_ratio"]] * inc
        ipm = inc / (np.nan_to_num(A[:, i["dependents"]], nan=0.0) + 1)
        nonre = A[:, i["open_lines"]] - A[:, i["real_estate_lines"]]
        with np.errstate(invalid="ignore"):
            over1 = (A[:, i["revol_util"]] > 1).astype(float)
        out = np.column_stack([A, flag, np.isnan(inc).astype(float), total, ever90, mdebt, ipm, nonre, over1])
        if not hasattr(self, "_cap_idx"):
            self._cap_idx = np.array([MODEL_FEATURES.index(c) for c in CAP_COLS])
            self._cap_val = self.caps_[CAP_COLS].to_numpy(dtype=float)
            self._med = self.medians_[MODEL_FEATURES].to_numpy(dtype=float)
        sub = out[:, self._cap_idx]
        out[:, self._cap_idx] = np.where(sub > self._cap_val, self._cap_val, sub)
        nan = np.isnan(out)
        out[nan] = np.take(self._med, np.nonzero(nan)[1])
        return pd.DataFrame(out, columns=MODEL_FEATURES, index=getattr(X, "index", None))

    def transform_reference(self, X):
        """Original pandas implementation, kept as the readable specification / test oracle."""
        d = apply_rules(X)
        d[CAP_COLS] = d[CAP_COLS].clip(upper=self.caps_, axis=1)
        return d.fillna(self.medians_)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.feature_names_out_)
