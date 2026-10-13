from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

TRAIN_PATH = Path(
    "data/machine_learning/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet"
)

TEST_PATH = Path(
    "data/machine_learning/"
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet"
)

RANDOM_STATE = 42


def load_dataset(path):
    df = pd.read_parquet(path)
    df.columns = df.columns.str.strip()

    if "Label" not in df.columns:
        raise ValueError(f"Label column missing in {path}")

    df = df.dropna(subset=["Label"]).copy()
    df["Label"] = df["Label"].astype(str).str.strip().str.upper()

    return df


def prepare_features(df):
    # Remove labels and the destination-port feature.
    X = df.drop(columns=["Label", "Destination Port"], errors="ignore")

    # Match the numeric preprocessing used in the baseline.
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]

    return X


def main():
    print("Loading training dataset...")
    train_df = load_dataset(TRAIN_PATH)

    # Keep only benign and DDoS training examples.
    train_df = train_df[
        train_df["Label"].isin(["BENIGN", "DDOS"])
    ].copy()

    before = len(train_df)
    train_df = train_df.drop_duplicates().reset_index(drop=True)

    print("Training duplicates removed:", before - len(train_df))
    print("Training rows:", len(train_df))

    y_train = (train_df["Label"] == "DDOS").astype(int)
    X_train = prepare_features(train_df)

    # Learn the feature schema from training data only.
    feature_columns = X_train.columns.tolist()

    print("Training features:", len(feature_columns))
    print("\nLoading independent test dataset...")

    test_df = load_dataset(TEST_PATH)

    # BENIGN = 0; every labeled web-attack category = 1.
    # This combines Brute Force, XSS, and SQL Injection.
    valid_test = test_df["Label"].ne("")
    test_df = test_df.loc[valid_test].copy()

    y_test = (test_df["Label"] != "BENIGN").astype(int)

    # Use only training features, in training column order.
    X_test = test_df.drop(
        columns=["Label", "Destination Port"],
        errors="ignore",
    )

    X_test = X_test.reindex(columns=feature_columns)

    for col in X_test.columns:
        X_test[col] = pd.to_numeric(
            X_test[col], errors="coerce"
        )

    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    print("External test rows:", len(X_test))
    print("\nExternal test labels:")
    print(y_test.map({0: "Benign", 1: "Malicious"}).value_counts())

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

    print("\nTraining on Friday DDoS data...")
    model.fit(X_train, y_train)

    print("Evaluating on Thursday Web Attacks data...")
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nExternal evaluation report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Benign", "Malicious"],
        zero_division=0,
        digits=4,
    ))

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print("Confusion matrix:")
    print(cm)

    print(f"\nFalse-positive rate: {fpr:.6f}")
    print(f"True negatives: {tn}")
    print(f"False positives: {fp}")
    print(f"False negatives: {fn}")
    print(f"True positives: {tp}")

    if len(np.unique(y_test)) == 2:
        print(
            "ROC-AUC:",
            f"{roc_auc_score(y_test, probabilities):.6f}",
        )
    else:
        print("ROC-AUC unavailable: test data has one class.")

    Path("results").mkdir(exist_ok=True)

    pd.DataFrame(
        cm,
        index=["Actual Benign", "Actual Malicious"],
        columns=["Predicted Benign", "Predicted Malicious"],
    ).to_csv("results/external_confusion_matrix.csv")

    pd.DataFrame({
        "actual": y_test.to_numpy(),
        "predicted": predictions,
        "malicious_probability": probabilities,
    }).to_csv("results/external_predictions.csv", index=False)

    print("\nEvaluation completed.")
    print("Results saved in the results/ directory.")


if __name__ == "__main__":
    main()