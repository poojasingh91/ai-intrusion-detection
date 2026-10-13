from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

TRAIN_PATH = Path(
    "data/machine_learning/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet"
)

TEST_PATH = Path(
    "data/machine_learning/"
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet"
)

THRESHOLD = 0.01


def load_data(path):
    df = pd.read_parquet(path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Label"]).copy()
    df["Label"] = df["Label"].astype(str).str.strip().str.upper()
    return df


def prepare_features(df):
    X = df.drop(
        columns=["Label", "Destination Port"],
        errors="ignore",
    )

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]
    return X


def main():
    train = load_data(TRAIN_PATH)
    train = train[train["Label"].isin(["BENIGN", "DDOS"])]
    train = train.drop_duplicates().reset_index(drop=True)

    y_train = (train["Label"] == "DDOS").astype(int)
    X_train = prepare_features(train)

    test = load_data(TEST_PATH)
    X_test = test.drop(
        columns=["Label", "Destination Port"],
        errors="ignore",
    )
    X_test = X_test.reindex(columns=X_train.columns)

    for col in X_test.columns:
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    y_test = (test["Label"] != "BENIGN").astype(int)

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

    print("Training model...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    # Summarize detection by individual attack category.
    results = test[["Label"]].copy()
    results["malicious_probability"] = probabilities
    results["predicted_malicious"] = predictions

    print(f"\nThreshold: {THRESHOLD}")
    print("\nDetection by attack category:")

    summary = results.groupby("Label").agg(
        samples=("predicted_malicious", "size"),
        detected=("predicted_malicious", "sum"),
        mean_score=("malicious_probability", "mean"),
        median_score=("malicious_probability", "median"),
    )

    summary["detection_rate"] = (
        summary["detected"] / summary["samples"]
    )

    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))

    # Compare feature distributions between training and external data.
    benign_train = train[train["Label"] == "BENIGN"]
    malicious_test = test[test["Label"] != "BENIGN"]

    train_features = prepare_features(benign_train)
    test_features = prepare_features(malicious_test)

    common = [
        c for c in X_train.columns
        if c in train_features.columns and c in test_features.columns
    ]

    rows = []

    for col in common:
        a = pd.to_numeric(train_features[col], errors="coerce")
        b = pd.to_numeric(test_features[col], errors="coerce")

        a = a.replace([np.inf, -np.inf], np.nan).dropna()
        b = b.replace([np.inf, -np.inf], np.nan).dropna()

        if len(a) == 0 or len(b) == 0:
            continue

        rows.append({
            "feature": col,
            "training_benign_median": a.median(),
            "external_attack_median": b.median(),
            "training_benign_mean": a.mean(),
            "external_attack_mean": b.mean(),
        })

    comparison = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)
    comparison.to_csv(
        "results/feature_distribution_comparison.csv",
        index=False,
    )

    summary.to_csv("results/detection_by_attack.csv")

    print("\nSaved:")
    print("results/detection_by_attack.csv")
    print("results/feature_distribution_comparison.csv")


if __name__ == "__main__":
    main()