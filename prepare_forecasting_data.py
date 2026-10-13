import pandas as pd
from pathlib import Path

# Load hourly electricity demand

file_path = Path("data/processed/hourly_electricity_demand.csv")
df = pd.read_csv(file_path, parse_dates=["Datetime"])

# Sort chronologically

df = df.sort_values("Datetime").reset_index(drop=True)

# Create time-based features

df["Hour"] = df["Datetime"].dt.hour
df["DayOfWeek"] = df["Datetime"].dt.dayofweek
df["Month"] = df["Datetime"].dt.month
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

# Create historical demand features

df["Demand_1_Hour_Ago"] = df["Average_Demand_kW"].shift(1)
df["Demand_24_Hours_Ago"] = df["Average_Demand_kW"].shift(24)
df["Demand_168_Hours_Ago"] = df["Average_Demand_kW"].shift(168)

# Remove rows without the required historical data

df = df.dropna().reset_index(drop=True)

# Save forecasting dataset

output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "forecasting_features.csv"
df.to_csv(output_file, index=False)

print("Forecasting features created successfully!")

print("\nFirst five rows:")
print(df.head())

print("\nDataset shape:", df.shape)

print("\nFeatures created:")
print([
"Hour",
"DayOfWeek",
"Month",
"IsWeekend",
"Demand_1_Hour_Ago",
"Demand_24_Hours_Ago",
"Demand_168_Hours_Ago"
])

print("\nSaved to:", output_file)
