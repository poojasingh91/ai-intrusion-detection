from pathlib import Path

import numpy as np
import pandas as pd

INPUT_PATH = Path(
    "results/feature_distribution_comparison.csv"
)


def main():
    df = pd.read_csv(INPUT_PATH)

    required = [
        "feature",
        "training_benign_median",
        "external_attack_median",
        "training_benign_mean",
        "external_attack_mean",
    ]

    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(
        subset=[
            "training_benign_median",
            "external_attack_median",
            "training_benign_mean",
            "external_attack_mean",
        ]
    )

    # A descriptive, scale-normalized median difference.
    # This is a screening measure, not a statistical test.
    scale = (
        df["training_benign_median"].abs()
        + df["external_attack_median"].abs()
        + 1e-9
    )

    df["relative_median_difference"] = (
        (
            df["external_attack_median"]
            - df["training_benign_median"]
        ).abs()
        / scale
    )

    df = df.sort_values(
        "relative_median_difference",
        ascending=False,
    )

    print("\nTop 15 features by relative median difference:")
    print(
        df[
            [
                "feature",
                "training_benign_median",
                "external_attack_median",
                "relative_median_difference",
            ]
        ].head(15).to_string(index=False)
    )

    Path("results").mkdir(exist_ok=True)
    df.to_csv("results/feature_shift_ranking.csv", index=False)

    print("\nSaved: results/feature_shift_ranking.csv")


if __name__ == "__main__":
    main()