from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report,
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_DIR / "data" / "ml_dataset.csv"
MODEL_DIR = PROJECT_DIR / "ml" / "saved_models"
REPORT_DIR = PROJECT_DIR / "reports"

TARGET = "Revenue"

# PageValues is excluded to reduce potential target leakage.
EXCLUDED_FEATURES = ["Revenue", "PageValues"]

CATEGORICAL_FEATURES = ["Month", "VisitorType"]
NUMERIC_FEATURES = [
    "Administrative",
    "Administrative_Duration",
    "Informational",
    "Informational_Duration",
    "ProductRelated",
    "ProductRelated_Duration",
    "BounceRates",
    "ExitRates",
    "SpecialDay",
    "OperatingSystems",
    "Browser",
    "Region",
    "TrafficType",
    "Weekend",
]


def build_preprocessor():
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    return ColumnTransformer([
        ("numeric", numeric_pipeline, NUMERIC_FEATURES),
        ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
    ])


def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Run ml/01_export_ml_dataset.py first."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(DATA_PATH)

    X = df.drop(columns=EXCLUDED_FEATURES)
    y = df[TARGET].astype(bool).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000,
            class_weight="balanced",
            random_state=42,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        ),
    }

    results = []

    for name, classifier in models.items():
        pipeline = Pipeline([
            ("preprocessor", build_preprocessor()),
            ("classifier", classifier),
        ])

        pipeline.fit(X_train, y_train)

        predictions = pipeline.predict(X_test)
        probabilities = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "model": name,
            "accuracy": round(accuracy_score(y_test, predictions), 4),
            "precision": round(
                precision_score(y_test, predictions, zero_division=0), 4
            ),
            "recall": round(
                recall_score(y_test, predictions, zero_division=0), 4
            ),
            "f1_score": round(
                f1_score(y_test, predictions, zero_division=0), 4
            ),
            "roc_auc": round(roc_auc_score(y_test, probabilities), 4),
        }
        results.append(metrics)

        model_filename = (
            "logistic_regression.joblib"
            if name == "Logistic Regression"
            else "random_forest.joblib"
        )
        joblib.dump(pipeline, MODEL_DIR / model_filename)

        print(f"\n=== {name} ===")
        print(metrics)
        print(classification_report(
            y_test,
            predictions,
            target_names=["No Purchase", "Purchase"],
            zero_division=0,
        ))

    results_df = pd.DataFrame(results)
    results_df.to_csv(REPORT_DIR / "model_comparison.csv", index=False)

    with open(REPORT_DIR / "model_comparison.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n=== MODEL COMPARISON ===")
    print(results_df.to_string(index=False))
    print(f"\nSaved comparison to: {REPORT_DIR / 'model_comparison.csv'}")
    print(f"Saved models to: {MODEL_DIR}")


if __name__ == "__main__":
    main()