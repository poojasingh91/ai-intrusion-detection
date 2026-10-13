from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # Use a non-interactive plotting backend

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    roc_auc_score,
)

DATA_PATH = Path(
    "data/machine_learning/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet"
)

RANDOM_STATE = 42


def main():
    # 1. Load the dataset
    print("Loading dataset...")
    df = pd.read_parquet(DATA_PATH)
    df.columns = df.columns.str.strip()

    print("Dataset shape:", df.shape)
    print("\nOriginal labels:")
    print(df["Label"].value_counts())

    # 2. Clean labels and create binary target
    df = df.dropna(subset=["Label"]).copy()
    labels = df["Label"].astype(str).str.strip().str.upper()

    valid_labels = labels.isin(["BENIGN", "DDOS"])
    df = df.loc[valid_labels].copy()
    y = (labels.loc[df.index] == "DDOS").astype(int)

    # 3. Prepare numeric features
    X = df.drop(columns=["Label"])

    # Exclude duplicated/identifier fields that may leak information
    excluded = [
        "flow id",
        "source ip",
        "destination ip",
        "timestamp",
    ]

    X = X.drop(
        columns=[
            c for c in X.columns
            if any(term in c.lower() for term in excluded)
        ],
        errors="ignore",
    )

    # Convert invalid numeric values to missing values
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")

    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.dropna(axis=1, how="all")
    X = X.loc[:, X.nunique(dropna=True) > 1]

    print("\nFeature count:", X.shape[1])
    print("Samples:", len(X))

    # 4. Split into training and testing data
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    # 5. Build and train the model
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

    print("\nTraining Random Forest...")
    model.fit(X_train, y_train)

    # 6. Evaluate the model
    predictions = model.predict(X_test)
    probabilities = model.predict_proba(X_test)[:, 1]

    print("\nClassification report:")
    print(classification_report(
        y_test,
        predictions,
        labels=[0, 1],
        target_names=["Benign", "DDoS"],
        zero_division=0,
    ))

    cm = confusion_matrix(y_test, predictions, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    print("Confusion matrix:")
    print(cm)
    print(f"False-positive rate: {fpr:.4f}")
    print(f"ROC-AUC: {roc_auc_score(y_test, probabilities):.4f}")

    # 7. Save confusion matrix plot
    Path("results").mkdir(exist_ok=True)

    ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "DDoS"],
    ).plot(cmap="Blues")

    plt.title("DDoS Detection - Random Forest")
    plt.tight_layout()
    plt.savefig("results/confusion_matrix.png", dpi=150)
    plt.close()

    # 8. Save feature importance
    forest = model.named_steps["classifier"]

    importance = pd.Series(
        forest.feature_importances_,
        index=X.columns,
    ).sort_values(ascending=False)

    importance.head(15).sort_values().plot(
        kind="barh",
        figsize=(9, 6),
    )
    plt.title("Top 15 Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("results/feature_importance.png", dpi=150)
    plt.close()

    importance.to_csv("results/feature_importance.csv")

    print("\nFinished!")
    print("Results saved in the results/ directory.")


if __name__ == "__main__":
    main()