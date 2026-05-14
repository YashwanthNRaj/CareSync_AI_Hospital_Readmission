from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, Tuple

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import numpy as np
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from src.config import (
    FIGURES_DIR,
    METRICS_PATH,
    MODEL_PATH,
    RANDOM_STATE,
    RAW_DATA_PATH,
    TARGET_COLUMN,
    TEST_SIZE,
    THRESHOLD_PATH,
    ensure_directories,
)
from src.data_preprocessing import (
    build_preprocessor,
    clean_missing_values,
    create_binary_target,
    drop_leakage_or_id_columns,
    load_data,
    split_features_target,
)
from src.evaluate import (
    calculate_metrics,
    plot_confusion_matrix,
    plot_pr_curve,
    plot_roc_curve,
    print_classification_report,
    save_metrics,
)
from src.feature_engineering import add_clinical_features
from src.threshold_tuning import evaluate_thresholds, save_threshold, select_recall_focused_threshold

try:
    import mlflow
    import mlflow.sklearn
except Exception:  # pragma: no cover - MLflow is optional at import time
    mlflow = None


def get_model_candidates() -> Dict[str, object]:
    """Return model candidates. XGBoost is included only if installed."""
    models: Dict[str, object] = {
        "logistic_regression_balanced": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "random_forest_balanced": RandomForestClassifier(
            n_estimators=250,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    try:
        from xgboost import XGBClassifier

        models["xgboost_optional"] = XGBClassifier(
            n_estimators=250,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
        )
    except Exception:
        print("XGBoost is not installed. Skipping optional XGBoost model.")

    return models


def get_positive_probabilities(model, X) -> np.ndarray:
    """Return positive-class probabilities for classifiers."""
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(X)
        return 1 / (1 + np.exp(-scores))
    return model.predict(X).astype(float)


def maybe_start_mlflow() -> None:
    """Configure MLflow when available."""
    if mlflow is None:
        print("MLflow is not available. Training will continue without experiment tracking.")
        return
    tracking_dir = PROJECT_ROOT / "mlruns"
    tracking_dir.mkdir(exist_ok=True)
    mlflow.set_tracking_uri(tracking_dir.as_uri())
    mlflow.set_experiment("CareGuard AI Hospital Readmission")


def log_run_to_mlflow(model_name: str, pipeline, metrics: Dict, model_params: Dict) -> None:
    """Log one model run to MLflow."""
    if mlflow is None:
        return
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model_name", model_name)
        for key, value in model_params.items():
            if isinstance(value, (str, int, float, bool)) or value is None:
                mlflow.log_param(key, value)
        for key, value in metrics.items():
            if isinstance(value, (int, float)) and value is not None:
                mlflow.log_metric(key, float(value))
        mlflow.sklearn.log_model(pipeline, artifact_path="model")


def load_and_prepare_data() -> Tuple[pd.DataFrame, pd.Series]:
    """Run the complete data preparation pipeline."""
    print(f"Loading dataset from: {RAW_DATA_PATH}")
    df = load_data(RAW_DATA_PATH)
    df = clean_missing_values(df)
    df = create_binary_target(df, TARGET_COLUMN)
    df = drop_leakage_or_id_columns(df)
    df = add_clinical_features(df)
    X, y = split_features_target(df, TARGET_COLUMN)
    print(f"Prepared data shape: X={X.shape}, y={y.shape}")
    print(f"Positive class rate: {y.mean():.4f}")
    return X, y


def train() -> Dict:
    """Train, evaluate, tune threshold, and save the best model."""
    ensure_directories()
    maybe_start_mlflow()

    X, y = load_and_prepare_data()
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    results = []
    trained_models = {}

    for model_name, estimator in get_model_candidates().items():
        print(f"\nTraining model: {model_name}")
        pipeline = ImbPipeline(
            steps=[
                ("preprocessor", build_preprocessor(X_train)),
                ("sampler", RandomOverSampler(random_state=RANDOM_STATE)),
                ("classifier", estimator),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_proba = get_positive_probabilities(pipeline, X_test)
        metrics = calculate_metrics(y_test, y_proba, threshold=0.50)
        metrics["model_name"] = model_name

        print(
            f"{model_name}: recall={metrics['recall']:.3f}, "
            f"precision={metrics['precision']:.3f}, f1={metrics['f1']:.3f}, "
            f"false_negatives={metrics['false_negatives']}"
        )

        log_run_to_mlflow(
            model_name=model_name,
            pipeline=pipeline,
            metrics=metrics,
            model_params=getattr(estimator, "get_params", lambda: {})(),
        )

        results.append(metrics)
        trained_models[model_name] = (pipeline, y_proba)

    # Select best model primarily by recall, then F1, then PR-AUC.
    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values(
        by=["recall", "f1", "average_precision_pr_auc"],
        ascending=[False, False, False],
    )
    best_model_name = result_df.iloc[0]["model_name"]
    best_pipeline, best_y_proba = trained_models[best_model_name]

    print(f"\nBest model selected: {best_model_name}")

    threshold_table = evaluate_thresholds(y_test, best_y_proba)
    selected_threshold, threshold_details = select_recall_focused_threshold(threshold_table)
    save_threshold(selected_threshold, THRESHOLD_PATH, threshold_details)

    final_metrics = calculate_metrics(y_test, best_y_proba, threshold=selected_threshold)
    final_metrics["selected_model"] = best_model_name
    final_metrics["selection_strategy"] = "Maximize recall first, then F1/PR-AUC balance."
    final_metrics["threshold_tuning"] = {
        "selected_threshold": selected_threshold,
        "explanation": "In this clinical context, we lower the threshold to catch more at-risk patients, reducing false negatives.",
        "threshold_table": threshold_table.to_dict(orient="records"),
    }
    final_metrics["model_comparison"] = result_df.to_dict(orient="records")

    save_metrics(final_metrics, METRICS_PATH)

    plot_confusion_matrix(
        y_test,
        best_y_proba,
        selected_threshold,
        FIGURES_DIR / "confusion_matrix.png",
        title=f"{best_model_name} Confusion Matrix",
    )
    plot_roc_curve(y_test, best_y_proba, FIGURES_DIR / "roc_curve.png")
    plot_pr_curve(y_test, best_y_proba, FIGURES_DIR / "pr_curve.png")

    print("\nFinal classification report:")
    print_classification_report(y_test, best_y_proba, selected_threshold)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(best_pipeline, MODEL_PATH)

    print(f"\nSaved model: {MODEL_PATH}")
    print(f"Saved threshold: {THRESHOLD_PATH}")
    print(f"Saved metrics: {METRICS_PATH}")
    print(f"Saved figures in: {FIGURES_DIR}")

    if mlflow is not None:
        with mlflow.start_run(run_name="best_model_recall_tuned"):
            mlflow.log_param("selected_model", best_model_name)
            mlflow.log_param("selected_threshold", selected_threshold)
            for key, value in final_metrics.items():
                if isinstance(value, (int, float)) and value is not None:
                    mlflow.log_metric(key, float(value))
            mlflow.log_artifact(str(METRICS_PATH))
            for fig_name in ["confusion_matrix.png", "roc_curve.png", "pr_curve.png"]:
                fig_path = FIGURES_DIR / fig_name
                if fig_path.exists():
                    mlflow.log_artifact(str(fig_path), artifact_path="figures")
            mlflow.sklearn.log_model(best_pipeline, artifact_path="best_model")

    return final_metrics


if __name__ == "__main__":
    try:
        train()
    except FileNotFoundError as exc:
        print("\nERROR: Dataset file is missing.")
        print(exc)
        print("\nFix: place diabetic_data.csv inside data/raw/ and run python src/train.py again.")
        raise SystemExit(1)
