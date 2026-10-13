import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

results_dir = Path("results")
results_dir.mkdir(exist_ok=True)

df = pd.read_csv(results_dir / "detection_by_attack.csv")
df = df[df["Label"].str.upper() != "BENIGN"].copy()
df["detection_rate_pct"] = df["detection_rate"] * 100

df = df.sort_values("detection_rate_pct", ascending=True)

plt.figure(figsize=(10, 6))
bars = plt.barh(
    df["Label"].str.replace("?", "-", regex=False),
    df["detection_rate_pct"],
    color=["#e76f51", "#f4a261", "#2a9d8f"],
)

plt.xlabel("Detection Rate (%)")
plt.ylabel("Attack Category")
plt.title("External Web Attack Detection Performance")
plt.xlim(0, 100)
plt.grid(axis="x", linestyle="--", alpha=0.4)

for bar, rate in zip(bars, df["detection_rate_pct"]):
    plt.text(
        min(rate + 1, 92),
        bar.get_y() + bar.get_height() / 2,
        f"{rate:.2f}%",
        va="center",
    )

plt.tight_layout()
output = results_dir / "attack_detection_rates.png"
plt.savefig(output, dpi=300, bbox_inches="tight")
plt.close()

print(f"Chart saved to: {output}")
