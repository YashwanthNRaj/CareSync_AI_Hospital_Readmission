from pathlib import Path

# Project root is the parent folder of src/
PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "diabetic_data.csv"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "caresync_readmission_model.joblib"
THRESHOLD_PATH = MODEL_DIR / "threshold.json"

REPORTS_DIR = PROJECT_ROOT / "reports"
METRICS_PATH = REPORTS_DIR / "metrics.json"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TEST_SIZE = 0.20
TARGET_COLUMN = "readmitted"
POSITIVE_CLASS_NAME = "Readmitted within 30 days"


def ensure_directories() -> None:
    """Create output directories if they do not exist."""
    for path in [PROCESSED_DATA_DIR, MODEL_DIR, REPORTS_DIR, FIGURES_DIR]:
        path.mkdir(parents=True, exist_ok=True)
