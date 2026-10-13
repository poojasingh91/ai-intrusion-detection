import pandas as pd
from pathlib import Path

files = list(Path("data").rglob("*.parquet"))

if not files:
    raise FileNotFoundError("No Parquet files found under data/")

df = pd.read_parquet(files[0])

print("File:", files[0])
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print(df.head())

label_columns = [
    c for c in df.columns if c.strip().lower() == "label"
]

if label_columns:
    print(df[label_columns[0]].value_counts(dropna=False))
else:
    print("No Label column found; inspect the dataset schema.")