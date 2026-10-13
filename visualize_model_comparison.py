import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load model comparison results

df = pd.read_csv("results/model_comparison.csv")

# Create output folder

output_folder = Path("results/figures")
output_folder.mkdir(parents=True, exist_ok=True)

# Plot MAE comparison

plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["MAE"])
plt.title("Model Comparison: Mean Absolute Error")
plt.xlabel("Machine Learning Model")
plt.ylabel("MAE (kW)")
plt.tight_layout()

output_file = output_folder / "model_comparison_mae.png"
plt.savefig(output_file, dpi=300)
print("Model comparison graph saved to:", output_file)

plt.show()
