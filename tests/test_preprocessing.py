import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.data_preprocessing import clean_missing_values, create_binary_target
from src.feature_engineering import add_clinical_features


def test_target_creation():
    df = pd.DataFrame({"readmitted": ["<30", "NO", ">30"]})
    result = create_binary_target(df)
    assert result["readmitted"].tolist() == [1, 0, 0]


def test_missing_value_cleaning():
    df = pd.DataFrame(
        {
            "race": ["Caucasian", "?"],
            "readmitted": ["<30", "NO"],
        }
    )
    cleaned = clean_missing_values(df)
    assert cleaned["race"].isna().sum() == 1


def test_feature_engineering_does_not_crash_with_missing_columns():
    df = pd.DataFrame({"time_in_hospital": [5, 9], "num_medications": [10, 18]})
    engineered = add_clinical_features(df)
    expected_columns = {
        "total_visits",
        "medication_change_flag",
        "diabetes_med_flag",
        "high_utilization_flag",
        "long_stay_flag",
        "num_medications_per_day",
    }
    assert expected_columns.issubset(set(engineered.columns))
