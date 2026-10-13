from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

DATA_PATH = Path(
    "data/machine_learning/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet"
)


def main():
    print("Loading dataset...")
    df = pd.read_parquet(DATA_PATH)
    df.columns = df.columns.str.strip()

    df = df.dropna(subset=["Label"]).copy()
    df["Label"] = df["Label"].astype(str).str.strip().str.upper()

    df = df[df["Label"].isin(["BENIGN", "DDOS"])]

    # Remove exact duplicate rows before splitting
    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    print("Rows removed as duplicates:", before - len(df))
    print("Rows remaining:", len(df))

    y = (df["Label"] == "DDOS").astype(int)
    X = df.drop(columns=["Label"])

    # Controlled experiment: exclude destination port
    X = X.drop(
        columns=[
            c for c in X.columns
            if c.strip().lower() == "destination port"
        ],
        errors="ignore",
    )

    # Ensure features are numeric
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("rf", RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    print("\nTraining...")
    model.fit(X_train, y_train)

    pred = model.predict(X_test)
    prob = model.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(classification_report(
        y_test,
        pred,
        target_names=["Benign", "DDoS"],
        zero_division=0,
    ))

    cm = confusion_matrix(y_test, pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    print("Confusion matrix:\n", cm)
    print(f"False-positive rate: {fp / (fp + tn):.6f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, prob):.6f}")

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(
        cm,
        index=["Actual Benign", "Actual DDoS"],
        columns=["Predicted Benign", "Predicted DDoS"],
    ).to_csv("results/confusion_matrix_v2.csv")

    importance = pd.Series(
        model.named_steps["rf"].feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    importance.to_csv("results/feature_importance_v2.csv")
    print("\nTop 10 features:")
    print(importance.head(10))


if __name__ == "__main__":
    main()