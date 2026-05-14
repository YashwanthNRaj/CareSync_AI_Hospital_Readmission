# Architecture

## System Architecture Diagram

```text
Raw Dataset
   ↓
Data Cleaning + Missing Value Handling
   ↓
Feature Engineering
   ↓
Train/Test Split
   ↓
Model Training + MLflow Tracking
   ↓
Threshold Tuning
   ↓
Saved Model Artifact
   ↓
FastAPI Risk Scoring API
   ↓
Streamlit Demo Dashboard
   ↓
Risk Explanation + Care Recommendation
```

## Component Breakdown

### 1. Raw Dataset

The user places `diabetic_data.csv` inside `data/raw/`.

### 2. Data Cleaning

The preprocessing module handles missing values represented by `?`, drops identifier columns, and prepares the binary target.

### 3. Feature Engineering

The feature engineering module creates utilization, medication, and length-of-stay features.

### 4. Model Training

The training script compares multiple models using a preprocessing pipeline and balanced learning strategy.

### 5. MLflow Tracking

Each model run logs parameters, metrics, and model artifacts to MLflow.

### 6. Threshold Tuning

A recall-focused threshold is selected to reduce false negatives.

### 7. Saved Artifacts

The final trained model and selected threshold are saved in `models/`.

### 8. FastAPI API

The API loads the trained model and serves prediction endpoints.

### 9. Streamlit Dashboard

The dashboard provides a clean demo interface for patient input, prediction, recommendation, and metrics.

### 10. Explanation and Recommendation

The prediction output includes risk level, explanation, care recommendation, and a clinical disclaimer.
