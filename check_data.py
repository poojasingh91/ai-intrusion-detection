import pandas as pd
from pathlib import Path

path = Path(
    "data/machine_learning/"
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv.parquet"
)

df = pd.read_parquet(path)
df.columns = df.columns.str.strip()

print("Rows:", len(df))
print("Columns:", len(df.columns))

# Check for duplicate records
print("\nExact duplicate rows:", df.duplicated().sum())

# Check whether identifiers or timestamps are present
print("\nPotential identifier/time columns:")
for col in df.columns:
    if any(
        term in col.lower()
        for term in [
            "flow id", "source ip", "destination ip",
            "timestamp", "source port", "destination port"
        ]
    ):
        print(" -", col)

# Check missing and infinite numeric values
numeric = df.drop(columns=["Label"], errors="ignore")
numeric = numeric.apply(pd.to_numeric, errors="coerce")

print("\nColumns entirely missing:", numeric.isna().all().sum())
print("Rows with duplicate feature vectors:",
      numeric.duplicated().sum())

print("\nLabel counts:")
print(df["Label"].value_counts())