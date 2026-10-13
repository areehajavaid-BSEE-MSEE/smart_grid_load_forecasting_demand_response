import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load the processed hourly dataset

file_path = Path("data/processed/hourly_electricity_demand.csv")

if not file_path.exists():
     raise FileNotFoundError("Hourly dataset not found. Run hourly_analysis.py first.")

df = pd.read_csv(file_path, parse_dates=["Datetime"])

# Plot electricity demand over time

plt.figure(figsize=(14, 6))

plt.plot(
df["Datetime"],
df["Average_Demand_kW"],
linewidth=0.6
)

plt.title("Hourly Electricity Demand Over Time")
plt.xlabel("Date")
plt.ylabel("Average Electricity Demand (kW)")
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save the graph

output_folder = Path("results/figures")
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "hourly_electricity_demand.png"
plt.savefig(output_file, dpi=300)

print("Graph saved successfully to:", output_file)

# Display the graph

plt.show()
