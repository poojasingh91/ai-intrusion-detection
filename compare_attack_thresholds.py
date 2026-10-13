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

THRESHOLDS = [0.05, 0.02, 0.01, 0.005, 0.002, 0.001]
RANDOM_STATE = 42


def load_dataset(path):
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
    return X.loc[:, X.nunique(dropna=True) > 1]


def main():
    print("Loading datasets...")

    train = load_dataset(TRAIN_PATH)
    train = train[train["Label"].isin(["BENIGN", "DDOS"])]
    train = train.drop_duplicates().reset_index(drop=True)

    X_train = prepare_features(train)
    y_train = (train["Label"] == "DDOS").astype(int)

    test = load_dataset(TEST_PATH)
    X_test = test.drop(
        columns=["Label", "Destination Port"],
        errors="ignore",
    ).reindex(columns=X_train.columns)

    for col in X_test.columns:
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )),
    ])

    print("Training model...")
    model.fit(X_train, y_train)

    scores = model.predict_proba(X_test)[:, 1]
    labels = test["Label"].to_numpy()
    rows = []

    for threshold in THRESHOLDS:
        predictions = scores >= threshold

        for label in sorted(np.unique(labels)):
            mask = labels == label
            samples = int(mask.sum())
            detected = int(predictions[mask].sum())

            rows.append({
                "threshold": threshold,
                "Label": label,
                "samples": samples,
                "detected": detected,
                "missed": samples - detected,
                "detection_rate": detected / samples if samples else 0,
            })

    results = pd.DataFrame(rows)
    Path("results").mkdir(exist_ok=True)

    output = Path("results/threshold_by_attack.csv")
    results.to_csv(output, index=False)

    attacks = results[results["Label"] != "BENIGN"].copy()
    attacks["detection_rate_pct"] = (
        attacks["detection_rate"] * 100
    )

    print("\nDetection rate by attack category and threshold:")
    print(
        attacks.pivot(
            index="threshold",
            columns="Label",
            values="detection_rate_pct",
        ).round(2).to_string()
    )

    print("\nFalse positives on benign traffic:")
    print(
        results[results["Label"] == "BENIGN"][
            ["threshold", "samples", "detected"]
        ].to_string(index=False)
    )

    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
