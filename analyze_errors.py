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

RESULTS_DIR = Path("results")
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
    ).copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]

    return X


def main():
    RESULTS_DIR.mkdir(exist_ok=True)

    print("Loading training dataset...")
    train = load_data(TRAIN_PATH)
    train = train[train["Label"].isin(["BENIGN", "DDOS"])]
    train = train.drop_duplicates().reset_index(drop=True)

    y_train = (train["Label"] == "DDOS").astype(int)
    X_train = prepare_features(train)

    print(f"Training rows: {len(train)}")
    print(f"Training features: {len(X_train.columns)}")

    print("\nLoading external test dataset...")
    test = load_data(TEST_PATH)

    X_test = prepare_features(test)
    X_test = X_test.reindex(columns=X_train.columns)
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

    print("\nTraining model...")
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= THRESHOLD).astype(int)

    results = test.copy()
    results["malicious_probability"] = probabilities
    results["predicted_malicious"] = predictions
    results["actual_malicious"] = y_test.to_numpy()

    results["error_type"] = "Correct"
    results.loc[
        (y_test.to_numpy() == 1) & (predictions == 0),
        "error_type",
    ] = "False Negative"
    results.loc[
        (y_test.to_numpy() == 0) & (predictions == 1),
        "error_type",
    ] = "False Positive"

    false_negatives = results[
        results["error_type"] == "False Negative"
    ].copy()

    false_positives = results[
        results["error_type"] == "False Positive"
    ].copy()

    false_negatives.to_csv(
        RESULTS_DIR / "false_negatives.csv",
        index=False,
    )

    false_positives.to_csv(
        RESULTS_DIR / "false_positives.csv",
        index=False,
    )

    print(f"\nThreshold: {THRESHOLD}")
    print("\nDetection by attack category:")

    summary = results.groupby("Label").agg(
        samples=("predicted_malicious", "size"),
        detected=("predicted_malicious", "sum"),
        mean_score=("malicious_probability", "mean"),
        median_score=("malicious_probability", "median"),
    )

    summary["missed"] = summary["samples"] - summary["detected"]
    summary["detection_rate"] = (
        summary["detected"] / summary["samples"]
    )

    print(summary.to_string(float_format=lambda x: f"{x:.4f}"))
    summary.to_csv(RESULTS_DIR / "detection_by_attack.csv")

    score_summary = results.groupby("Label")[
        "malicious_probability"
    ].agg(
        samples="count",
        minimum="min",
        percentile_25=lambda s: s.quantile(0.25),
        median="median",
        percentile_75=lambda s: s.quantile(0.75),
        maximum="max",
        mean="mean",
    )

    score_summary.to_csv(
        RESULTS_DIR / "attack_score_summary.csv"
    )

    print("\nPrediction score distributions:")
    print(score_summary.to_string(float_format=lambda x: f"{x:.6f}"))

    missed_mask = (
        (y_test.to_numpy() == 1) & (predictions == 0)
    )
    detected_mask = (
        (y_test.to_numpy() == 1) & (predictions == 1)
    )

    feature_rows = []

    for col in X_train.columns:
        missed_values = X_test.loc[missed_mask, col].dropna()
        detected_values = X_test.loc[detected_mask, col].dropna()

        if missed_values.empty or detected_values.empty:
            continue

        missed_median = missed_values.median()
        detected_median = detected_values.median()

        scale = abs(missed_median) + abs(detected_median)

        relative_difference = (
            abs(missed_median - detected_median) / scale
            if scale != 0
            else 0.0
        )

        feature_rows.append({
            "feature": col,
            "missed_attack_median": missed_median,
            "detected_attack_median": detected_median,
            "missed_attack_mean": missed_values.mean(),
            "detected_attack_mean": detected_values.mean(),
            "relative_median_difference": relative_difference,
        })

    feature_comparison = pd.DataFrame(feature_rows)

    if not feature_comparison.empty:
        feature_comparison = feature_comparison.sort_values(
            "relative_median_difference",
            ascending=False,
        )

    feature_comparison.to_csv(
        RESULTS_DIR / "false_negative_feature_comparison.csv",
        index=False,
    )

    print("\nMissed attacks by category:")

    if false_negatives.empty:
        print("No false negatives at this threshold.")
    else:
        missed_summary = false_negatives.groupby("Label").agg(
            missed_samples=("Label", "size"),
            mean_score=("malicious_probability", "mean"),
            median_score=("malicious_probability", "median"),
            maximum_score=("malicious_probability", "max"),
        )

        print(
            missed_summary.to_string(
                float_format=lambda x: f"{x:.6f}"
            )
        )

    print(f"\nFalse negatives: {len(false_negatives)}")
    print(f"False positives: {len(false_positives)}")

    print("\nTop 15 features differing between missed and detected attacks:")
    if feature_comparison.empty:
        print("Insufficient samples for feature comparison.")
    else:
        print(feature_comparison.head(15).to_string(index=False))

    print("\nSaved files:")
    for filename in [
        "false_negatives.csv",
        "false_positives.csv",
        "attack_score_summary.csv",
        "false_negative_feature_comparison.csv",
        "detection_by_attack.csv",
    ]:
        print(RESULTS_DIR / filename)


if __name__ == "__main__":
    main()