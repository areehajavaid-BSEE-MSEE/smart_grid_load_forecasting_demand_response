import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load hourly electricity demand data

file_path = Path("data/processed/hourly_electricity_demand.csv")
df = pd.read_csv(file_path, parse_dates=["Datetime"])

# Extract hour of day

df["Hour"] = df["Datetime"].dt.hour

# Calculate average demand for each hour across all days

hourly_pattern = df.groupby("Hour")["Average_Demand_kW"].mean()

# Identify peak and lowest average demand hours

peak_hour = hourly_pattern.idxmax()
lowest_hour = hourly_pattern.idxmin()

print("\nAVERAGE ELECTRICITY DEMAND BY HOUR")
print(hourly_pattern.round(3))

print(f"\nPeak average-demand hour: {peak_hour:02d}:00")
print(f"Average demand at peak hour: {hourly_pattern.max():.3f} kW")

print(f"\nLowest average-demand hour: {lowest_hour:02d}:00")
print(f"Average demand at lowest hour: {hourly_pattern.min():.3f} kW")

# Plot daily demand pattern

plt.figure(figsize=(12, 6))
plt.plot(hourly_pattern.index, hourly_pattern.values, marker="o")

plt.title("Average Electricity Demand by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Average Electricity Demand (kW)")
plt.xticks(range(0, 24))
plt.grid(True, alpha=0.3)
plt.tight_layout()

# Save results

output_folder = Path("results/figures")
output_folder.mkdir(parents=True, exist_ok=True)

plt.savefig(
output_folder / "average_daily_demand_pattern.png",
dpi=300
)

plt.show()

print("\nDaily demand analysis completed successfully!")
print("Graph saved in results/figures/")
