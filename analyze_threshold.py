from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
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


def load_data(path):
    df = pd.read_parquet(path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Label"]).copy()
    df["Label"] = df["Label"].astype(str).str.strip().str.upper()
    return df


def features(df):
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
    X_train = features(train)

    test = load_data(TEST_PATH)
    y_test = (test["Label"] != "BENIGN").astype(int)

    X_test = test.drop(
        columns=["Label", "Destination Port"],
        errors="ignore",
    )
    X_test = X_test.reindex(columns=X_train.columns)

    for col in X_test.columns:
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")

    X_test = X_test.replace([np.inf, -np.inf], np.nan)

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

    # Higher scores indicate greater similarity to the
    # DDoS class learned during training.
    scores = model.predict_proba(X_test)[:, 1]

    print("\nROC-AUC:", round(roc_auc_score(y_test, scores), 6))
    print("\nThreshold comparison")
    print("Threshold | Precision | Recall | F1 | False-positive rate")

    rows = []

    for threshold in [
        0.50, 0.40, 0.30, 0.20, 0.10, 0.05, 0.01
    ]:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel()

        precision = precision_score(
            y_test, pred, zero_division=0
        )
        recall = recall_score(
            y_test, pred, zero_division=0
        )
        f1 = f1_score(y_test, pred, zero_division=0)
        fpr = fp / (fp + tn) if (fp + tn) else 0.0

        rows.append({
            "threshold": threshold,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "TN": tn,
        })

        print(
            f"{threshold:9.2f} | {precision:9.4f} | "
            f"{recall:6.4f} | {f1:.4f} | {fpr:.6f}"
        )

    Path("results").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(
        "results/threshold_analysis.csv",
        index=False,
    )
    print("\nDetailed results saved to results/threshold_analysis.csv")


if __name__ == "__main__":
    main()