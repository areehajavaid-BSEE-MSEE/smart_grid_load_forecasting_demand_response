import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load feature importance results

file_path = Path("results/feature_importance.csv")
df = pd.read_csv(file_path)

# Sort features by importance

df = df.sort_values("Importance", ascending=True)

print("\nFEATURE IMPORTANCE RANKING")
print(df.sort_values("Importance", ascending=False).to_string(index=False))

# Create horizontal bar chart

plt.figure(figsize=(10, 6))

plt.barh(
df["Feature"],
df["Importance"]
)

plt.title("Feature Importance in Electricity Demand Forecasting")
plt.xlabel("Importance Score")
plt.ylabel("Forecasting Feature")
plt.tight_layout()

# Save the figure

output_folder = Path("results/figures")
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "feature_importance.png"
plt.savefig(output_file, dpi=300)

print("\nFeature importance graph saved to:", output_file)

plt.show()
