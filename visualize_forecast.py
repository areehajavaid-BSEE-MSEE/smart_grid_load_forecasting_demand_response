import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Load forecasting results

file_path = Path("results/forecast_results.csv")
df = pd.read_csv(file_path, parse_dates=["Datetime"])

# Select an illustrative 7-day period from the test data

plot_data = df.head(24 * 7).copy()

# Create the comparison graph

plt.figure(figsize=(14, 6))

plt.plot(
plot_data["Datetime"],
plot_data["Actual_Demand_kW"],
label="Actual Demand",
linewidth=1.5
)

plt.plot(
plot_data["Datetime"],
plot_data["Predicted_Demand_kW"],
label="Random Forest Prediction",
linewidth=1.5
)

plt.plot(
plot_data["Datetime"],
plot_data["Baseline_Demand_kW"],
label="Persistence Baseline",
linewidth=1,
alpha=0.8
)

plt.title("Actual vs Predicted Electricity Demand")
plt.xlabel("Date and Time")
plt.ylabel("Electricity Demand (kW)")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(rotation=30)
plt.tight_layout()

# Save the graph

output_folder = Path("results/figures")
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "actual_vs_predicted_demand.png"
plt.savefig(output_file, dpi=300)

print("Forecast comparison graph saved to:", output_file)

plt.show()
