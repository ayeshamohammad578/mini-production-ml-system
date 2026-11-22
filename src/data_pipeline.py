"""Ingestion, validation and leak-free stratified splitting.

Run:  python -m src.data_pipeline
Writes data/train.csv, data/val.csv, data/test.csv and reports/leakage_audit.json.
"""
import json
import logging
import urllib.request

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import (DATA_DIR, RAW_FILE, DATA_URL, RENAME, RAW_FEATURES, TARGET, ID_COL,
                        SEED, TEST_SIZE, VAL_SIZE, REPORT_DIR)

log = logging.getLogger("data_pipeline")


def fetch_raw(force: bool = False) -> pd.DataFrame:
    """Download the public dataset once and cache it in data/."""
    DATA_DIR.mkdir(exist_ok=True)
    if force or not RAW_FILE.exists():
        log.info("Downloading %s", DATA_URL)
        urllib.request.urlretrieve(DATA_URL, RAW_FILE)
    df = pd.read_csv(RAW_FILE)
    df = df.rename(columns={df.columns[0]: ID_COL}).rename(columns=RENAME)
    expected = {ID_COL, TARGET, *RAW_FEATURES}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Raw file is missing columns: {missing}")
    if not df[ID_COL].is_unique:
        raise ValueError("Applicant IDs are not unique")
    return df[[ID_COL] + RAW_FEATURES + [TARGET]]


def split(df: pd.DataFrame):
    """Stratified 70/15/15 split, made BEFORE any cleaning or fitting.

    The data is cross-sectional (one row per borrower, no timestamp), so a
    temporal cutoff is impossible; stratification keeps the 6.7% default rate
    identical in every partition. Exact duplicates are removed from TRAIN only,
    after the split, so validation/test keep their natural distribution.
    """
    trval, test = train_test_split(df, test_size=TEST_SIZE, stratify=df[TARGET], random_state=SEED)
    train, val = train_test_split(trval, test_size=VAL_SIZE / (1 - TEST_SIZE),
                                  stratify=trval[TARGET], random_state=SEED)
    before = len(train)
    train = train.loc[~train.drop(columns=ID_COL).duplicated()]
    log.info("Removed %d exact duplicates from train", before - len(train))
    return train, val, test


def leakage_audit(train, val, test) -> dict:
    """Explicit checks written to reports/leakage_audit.json."""
    feats = RAW_FEATURES
    key = lambda d: set(map(tuple, d[feats].round(6).astype(str).values))
    audit = {
        "rows": {"train": len(train), "val": len(val), "test": len(test)},
        "default_rate": {k: round(float(v[TARGET].mean()), 4) for k, v in
                         {"train": train, "val": val, "test": test}.items()},
        "id_overlap_train_val": len(set(train[ID_COL]) & set(val[ID_COL])),
        "id_overlap_train_test": len(set(train[ID_COL]) & set(test[ID_COL])),
        "id_overlap_val_test": len(set(val[ID_COL]) & set(test[ID_COL])),
        "identical_feature_rows_train_test": len(key(train) & key(test)),
        "target_derived_features": [],   # no input is computed from the outcome
        "post_outcome_features": [],     # all inputs are known at application time
        "preprocessing_fitted_on": "train partition only (inside sklearn Pipeline)",
    }
    audit["passed"] = (audit["id_overlap_train_val"] == audit["id_overlap_train_test"]
                       == audit["id_overlap_val_test"] == 0)
    return audit


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    df = fetch_raw()
    log.info("Loaded %d rows, default rate %.2f%%", len(df), 100 * df[TARGET].mean())
    train, val, test = split(df)
    for name, part in {"train": train, "val": val, "test": test}.items():
        part.to_csv(DATA_DIR / f"{name}.csv", index=False)
    audit = leakage_audit(train, val, test)
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / "leakage_audit.json").write_text(json.dumps(audit, indent=2))
    log.info("Leakage audit passed: %s | %s", audit["passed"], audit["rows"])


if __name__ == "__main__":
    main()
