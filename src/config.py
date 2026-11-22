"""Central configuration: paths, schema, split ratios and business costs.

Every other module imports from here so that training, serving and monitoring
share one definition of the feature contract.
"""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ARTIFACT_DIR = ROOT / "artifacts"
REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"

RAW_FILE = DATA_DIR / "cs-training.csv"
DATA_URL = ("https://raw.githubusercontent.com/DrIanGregory/Kaggle-GiveMeSomeCredit/"
            "master/data/GiveMeSomeCredit-training.csv")   # public mirror of Kaggle "Give Me Some Credit" (2011)

SEED = 42
TEST_SIZE = 0.15          # 15% held-out test (touched once, by evaluate.py)
VAL_SIZE = 0.15           # 15% validation (model / threshold selection)

TARGET = "default"
ID_COL = "applicant_id"

# Original Kaggle column -> short snake_case name used across the system
RENAME = {
    "SeriousDlqin2yrs": TARGET,
    "RevolvingUtilizationOfUnsecuredLines": "revol_util",
    "age": "age",
    "NumberOfTime30-59DaysPastDueNotWorse": "late_30_59",
    "DebtRatio": "debt_ratio",
    "MonthlyIncome": "monthly_income",
    "NumberOfOpenCreditLinesAndLoans": "open_lines",
    "NumberOfTimes90DaysLate": "late_90",
    "NumberRealEstateLoansOrLines": "real_estate_lines",
    "NumberOfTime60-89DaysPastDueNotWorse": "late_60_89",
    "NumberOfDependents": "dependents",
}
RAW_FEATURES = ["revol_util", "age", "late_30_59", "late_60_89", "late_90", "debt_ratio",
                "monthly_income", "open_lines", "real_estate_lines", "dependents"]
LATE_COLS = ["late_30_59", "late_60_89", "late_90"]
CAP_COLS = ["revol_util", "debt_ratio", "monthly_income", "monthly_debt", "income_per_member"]

# Business cost of errors (used to choose the decision threshold):
# approving a borrower who later defaults (false negative) costs ~5x
# declining a good borrower (false positive) - lost loss vs lost margin.
COST_FN = 5.0
COST_FP = 1.0

# Drift monitoring thresholds (industry rule of thumb for PSI)
PSI_WARN = 0.10
PSI_ALERT = 0.25
KS_PVALUE_ALERT = 0.01
KS_STAT_MIN = 0.05        # effect-size floor: with 20k+ rows a p-value alone flags trivial shifts
