import numpy as np
import pandas as pd


def _numeric_series(df: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    """Return a numeric Series even when the column is missing or dirty."""
    if column not in df.columns:
        return pd.Series(default, index=df.index, dtype="float64")
    return pd.to_numeric(df[column], errors="coerce").fillna(default)


def add_clinical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add robust utilization and discharge-risk features.

    The function is defensive: if optional columns are missing, it creates the
    engineered feature using safe defaults instead of crashing.
    """
    df = df.copy()

    number_inpatient = _numeric_series(df, "number_inpatient")
    number_outpatient = _numeric_series(df, "number_outpatient")
    number_emergency = _numeric_series(df, "number_emergency")
    time_in_hospital = _numeric_series(df, "time_in_hospital")
    num_medications = _numeric_series(df, "num_medications")

    df["total_visits"] = number_inpatient + number_outpatient + number_emergency

    if "change" in df.columns:
        df["medication_change_flag"] = df["change"].astype(str).str.strip().str.lower().isin(["ch", "yes", "1", "true"]).astype(int)
    else:
        df["medication_change_flag"] = 0

    if "diabetesMed" in df.columns:
        df["diabetes_med_flag"] = df["diabetesMed"].astype(str).str.strip().str.lower().isin(["yes", "1", "true"]).astype(int)
    else:
        df["diabetes_med_flag"] = 0

    df["high_utilization_flag"] = (df["total_visits"] >= 2).astype(int)
    df["long_stay_flag"] = (time_in_hospital >= 7).astype(int)

    safe_days = time_in_hospital.replace(0, np.nan)
    df["num_medications_per_day"] = (num_medications / safe_days).replace([np.inf, -np.inf], np.nan).fillna(0)

    return df
