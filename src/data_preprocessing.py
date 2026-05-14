from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import RAW_DATA_PATH, TARGET_COLUMN


ID_COLUMNS = ["encounter_id", "patient_nbr"]


def load_data(path=RAW_DATA_PATH) -> pd.DataFrame:
    """Load the raw diabetes readmission dataset."""
    path = path if hasattr(path, "exists") else RAW_DATA_PATH.__class__(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at {path}. Download diabetic_data.csv from UCI and place it in data/raw/."
        )
    return pd.read_csv(path)


def clean_missing_values(df: pd.DataFrame, extreme_missing_threshold: float = 0.80) -> pd.DataFrame:
    """
    Replace UCI missing marker '?' with NaN and drop extremely sparse columns.

    The target column is never dropped by the missingness rule.
    """
    df = df.copy()
    df = df.replace("?", np.nan)

    # Strip whitespace in text columns to reduce accidental category duplicates.
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].map(lambda value: value.strip() if isinstance(value, str) else value)

    missing_ratio = df.isna().mean()
    columns_to_drop = [
        col for col, ratio in missing_ratio.items()
        if ratio > extreme_missing_threshold and col != TARGET_COLUMN
    ]
    if columns_to_drop:
        df = df.drop(columns=columns_to_drop, errors="ignore")

    return df


def create_binary_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> pd.DataFrame:
    """
    Convert readmitted into a binary target.

    readmitted == '<30' becomes 1. Values 'NO' and '>30' become 0.
    Rows with unexpected target values are removed.
    """
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' was not found in the dataframe.")

    df = df.copy()
    mapping = {"<30": 1, "NO": 0, ">30": 0, 1: 1, 0: 0, "1": 1, "0": 0}
    df[target_column] = df[target_column].map(mapping)
    df = df.dropna(subset=[target_column])
    df[target_column] = df[target_column].astype(int)
    return df


def drop_leakage_or_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Drop identifier columns and any direct leakage columns if present."""
    df = df.copy()
    leakage_or_id_columns = [col for col in ID_COLUMNS if col in df.columns]
    return df.drop(columns=leakage_or_id_columns, errors="ignore")


def split_features_target(df: pd.DataFrame, target_column: str = TARGET_COLUMN) -> Tuple[pd.DataFrame, pd.Series]:
    """Split dataframe into features and target."""
    if target_column not in df.columns:
        raise KeyError(f"Target column '{target_column}' was not found.")
    X = df.drop(columns=[target_column])
    y = df[target_column].astype(int)
    return X, y


def _make_one_hot_encoder() -> OneHotEncoder:
    """
    Create a version-friendly OneHotEncoder.

    The dense output keeps the pipeline compatible with tree, linear, and
    gradient boosting models. dtype float32 reduces memory usage.
    """
    try:
        return OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=False,
            dtype=np.float32,
            min_frequency=10,
        )
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False, dtype=np.float32)


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """Build preprocessing for numeric and categorical columns."""
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", _make_one_hot_encoder()),
        ]
    )

    transformers = []
    if numeric_features:
        transformers.append(("num", numeric_pipeline, numeric_features))
    if categorical_features:
        transformers.append(("cat", categorical_pipeline, categorical_features))

    if not transformers:
        raise ValueError("No usable feature columns were found for preprocessing.")

    return ColumnTransformer(transformers=transformers, remainder="drop")
