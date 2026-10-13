from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, roc_auc_score

TRAIN_PATH = Path(
    "data/machine_learning/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet"
)
TEST_PATH = Path(
    "data/machine_learning/"
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv.parquet"
)
OUTPUT_PATH = Path("results/threshold_comparison.csv")
RANDOM_STATE = 42

def load_dataset(path):
    df = pd.read_parquet(path)
    df.columns = df.columns.str.strip()
    df = df.dropna(subset=["Label"]).copy()
    df["Label"] = df["Label"].astype(str).str.strip().str.upper()
    return df

def prepare_features(df):
    X = df.drop(columns=["Label", "Destination Port"], errors="ignore")
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
        columns=["Label", "Destination Port"], errors="ignore"
    ).reindex(columns=X_train.columns)

    for col in X_test.columns:
        X_test[col] = pd.to_numeric(X_test[col], errors="coerce")
    X_test = X_test.replace([np.inf, -np.inf], np.nan)
    y_test = (test["Label"] != "BENIGN").astype(int).to_numpy()

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

    print("Training baseline model...")
    model.fit(X_train, y_train)
    scores = model.predict_proba(X_test)[:, 1]

    print(f"ROC-AUC: {roc_auc_score(y_test, scores):.6f}")
    rows = []

    for threshold in [0.01, 0.005, 0.002, 0.001, 0.0005, 0.0001]:
        pred = (scores >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(
            y_test, pred, labels=[0, 1]
        ).ravel()

        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )
        fpr = fp / (fp + tn) if fp + tn else 0.0

        rows.append({
            "threshold": threshold,
            "true_positives": tp,
            "false_negatives": fn,
            "false_positives": fp,
            "true_negatives": tn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_positive_rate": fpr,
        })

    results = pd.DataFrame(rows)
    OUTPUT_PATH.parent.mkdir(exist_ok=True)
    results.to_csv(OUTPUT_PATH, index=False)

    print("\nThreshold comparison:")
    print(results.to_string(
        index=False,
        formatters={
            "precision": "{:.4f}".format,
            "recall": "{:.4f}".format,
            "f1": "{:.4f}".format,
            "false_positive_rate": "{:.4f}".format,
        },
    ))
    print(f"\nSaved: {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
