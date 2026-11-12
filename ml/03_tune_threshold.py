
from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


# --------------------------------------------------
# 1. Paths and configuration
# --------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent

DATA_PATH = ROOT / "data" / "ml_dataset.csv"
MODEL_DIR = ROOT / "ml" / "saved_models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42
TARGET = "Revenue"

# PageValues is excluded to reduce potential target leakage.
FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
    "Month",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "VisitorType",
    "Weekend",
]

CATEGORICAL_FEATURES = ["Month", "VisitorType"]
NUMERIC_FEATURES = [
    col for col in FEATURES if col not in CATEGORICAL_FEATURES
]


# --------------------------------------------------
# 2. Load dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

X = df[FEATURES].copy()
y = df[TARGET].astype(int)

print(f"Dataset loaded: {len(df):,} rows")
print(f"Features: {len(FEATURES)}")


# --------------------------------------------------
# 3. Split: 60% train, 20% validation, 20% test
# --------------------------------------------------

X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=0.40,
    random_state=RANDOM_STATE,
    stratify=y,
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=0.50,
    random_state=RANDOM_STATE,
    stratify=y_temp,
)

print(f"Training rows:   {len(X_train):,}")
print(f"Validation rows: {len(X_val):,}")
print(f"Test rows:       {len(X_test):,}")


# --------------------------------------------------
# 4. Preprocessing and model
# --------------------------------------------------

numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
])

categorical_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric", numeric_pipeline, NUMERIC_FEATURES),
    ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
])

model = Pipeline([
    ("preprocessor", preprocessor),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    ),
])

model.fit(X_train, y_train)


# --------------------------------------------------
# 5. Select threshold using validation data only
# --------------------------------------------------

val_probabilities = model.predict_proba(X_val)[:, 1]

thresholds = np.arange(0.05, 0.501, 0.001)

best_threshold = 0.5
best_f1 = -1.0

for threshold in thresholds:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)

    score = f1_score(
        y_val,
        val_predictions,
        zero_division=0,
    )

    if score > best_f1:
        best_f1 = score
        best_threshold = float(threshold)

print("\nThreshold selection")
print(f"Selected threshold: {best_threshold:.4f}")
print(f"Validation F1:      {best_f1:.4f}")


# --------------------------------------------------
# 6. Evaluate on untouched test data
# --------------------------------------------------

test_probabilities = model.predict_proba(X_test)[:, 1]

default_predictions = (
    test_probabilities >= 0.5
).astype(int)

tuned_predictions = (
    test_probabilities >= best_threshold
).astype(int)


def calculate_metrics(y_true, predictions, probabilities):
    return {
        "accuracy": round(
            accuracy_score(y_true, predictions), 4
        ),
        "precision": round(
            precision_score(
                y_true, predictions, zero_division=0
            ), 4
        ),
        "recall": round(
            recall_score(
                y_true, predictions, zero_division=0
            ), 4
        ),
        "f1": round(
            f1_score(
                y_true, predictions, zero_division=0
            ), 4
        ),
        "roc_auc": round(
            roc_auc_score(y_true, probabilities), 4
        ),
    }


default_metrics = calculate_metrics(
    y_test, default_predictions, test_probabilities
)

tuned_metrics = calculate_metrics(
    y_test, tuned_predictions, test_probabilities
)

comparison = pd.DataFrame([
    {"configuration": "Default threshold (0.5)",
     **default_metrics},
    {"configuration": f"Tuned threshold ({best_threshold:.4f})",
     **tuned_metrics},
])

print("\nTest-set comparison")
print(comparison.to_string(index=False))


# --------------------------------------------------
# 7. Save model, threshold, and evaluation report
# --------------------------------------------------

model_path = MODEL_DIR / "random_forest_tuned.joblib"
threshold_path = MODEL_DIR / "threshold.json"
report_path = REPORT_DIR / "threshold_comparison.csv"

joblib.dump(model, model_path)

metadata = {
    "model_type": "RandomForestClassifier",
    "threshold": best_threshold,
    "validation_f1": round(best_f1, 4),
    "target": TARGET,
    "features": FEATURES,
    "excluded_features": ["PageValues"],
    "random_state": RANDOM_STATE,
    "train_rows": len(X_train),
    "validation_rows": len(X_val),
    "test_rows": len(X_test),
}

with open(threshold_path, "w", encoding="utf-8") as file:
    json.dump(metadata, file, indent=4)

comparison.to_csv(report_path, index=False)

print("\nSaved successfully:")
print(f"Model:     {model_path}")
print(f"Threshold: {threshold_path}")
print(f"Report:    {report_path}")
