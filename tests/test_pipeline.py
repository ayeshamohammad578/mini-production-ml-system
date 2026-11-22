"""Leakage safeguards and inference integrity of the serialized pipeline."""
import json

import joblib
import numpy as np
import pandas as pd

from src.config import ARTIFACT_DIR, RAW_FEATURES
from src.data_pipeline import split
from src.features import CreditFeatureBuilder, MODEL_FEATURES


def test_split_is_disjoint_and_stratified(synthetic_frame):
    big = pd.concat([synthetic_frame] * 10, ignore_index=True)
    big["applicant_id"] = np.arange(len(big))
    big[RAW_FEATURES] = big[RAW_FEATURES] + np.random.default_rng(1).random((len(big), len(RAW_FEATURES))) * 1e-3
    tr, va, te = split(big)
    ids = [set(x["applicant_id"]) for x in (tr, va, te)]
    assert not (ids[0] & ids[1]) and not (ids[0] & ids[2]) and not (ids[1] & ids[2])
    rates = [x["default"].mean() for x in (tr, va, te)]
    assert max(rates) - min(rates) < 0.02


def test_feature_builder_learns_only_from_fit_data(synthetic_frame):
    X = synthetic_frame[RAW_FEATURES]
    fb = CreditFeatureBuilder().fit(X.iloc[:200])
    caps_before = fb.caps_.copy()
    fb.transform(X.iloc[200:])                       # transforming new data must not refit anything
    pd.testing.assert_series_equal(caps_before, fb.caps_)


def test_fast_transform_matches_reference(synthetic_frame):
    X = synthetic_frame[RAW_FEATURES]
    fb = CreditFeatureBuilder().fit(X)
    fast, ref = fb.transform(X), fb.transform_reference(X)
    assert list(fast.columns) == MODEL_FEATURES
    np.testing.assert_allclose(fast.to_numpy(), ref[MODEL_FEATURES].to_numpy(), rtol=0, atol=1e-9)
    assert not fast.isna().any().any()


def test_serialized_model_scores_unseen_and_edge_records(synthetic_frame):
    model = joblib.load(ARTIFACT_DIR / "model.joblib")
    p = model.predict_proba(synthetic_frame[RAW_FEATURES])[:, 1]
    assert p.shape == (len(synthetic_frame),)
    assert np.isfinite(p).all() and (p >= 0).all() and (p <= 1).all()


def test_metadata_contract():
    meta = json.loads((ARTIFACT_DIR / "model_metadata.json").read_text())
    assert meta["input_features"] == RAW_FEATURES
    assert 0 < meta["decision_threshold"] < 1
    assert "LightGBM" in meta["selected_model"]
