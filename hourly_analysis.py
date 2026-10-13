import pandas as pd
from pathlib import Path

# Dataset location

file_path = Path("data/household_power_consumption.txt")

print("Loading dataset...")

# Define column names

column_names = [
"Date",
"Time",
"Global_active_power",
"Global_reactive_power",
"Voltage",
"Global_intensity",
"Sub_metering_1",
"Sub_metering_2",
"Sub_metering_3"
]

# Read the dataset

df = pd.read_csv(
file_path,
sep=";",
header=0,
names=column_names,
na_values=["?"],
low_memory=False
)

# Convert measurements to numeric values

numeric_columns = column_names[2:]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Create timestamps

df["Datetime"] = pd.to_datetime(
df["Date"] + " " + df["Time"],
format="%d/%m/%Y %H:%M:%S",
errors="coerce"
)

# Remove rows without valid timestamps or active power

df = df.dropna(subset=["Datetime", "Global_active_power"])

# Sort data chronologically

df = df.sort_values("Datetime")

# Set timestamps as the index

df = df.set_index("Datetime")

# Calculate average electricity demand for each hour

hourly_data = df["Global_active_power"].resample("h").mean()

# Remove hours without measurements

hourly_data = hourly_data.dropna()

# Convert to a DataFrame

hourly_df = hourly_data.to_frame(
name="Average_Demand_kW"
)

# Save hourly data

output_folder = Path("data/processed")
output_folder.mkdir(parents=True, exist_ok=True)

output_file = output_folder / "hourly_electricity_demand.csv"
hourly_df.to_csv(output_file)

print("\nHourly analysis completed successfully!")
print("\nFirst five hourly records:")
print(hourly_df.head())

print("\nTotal hourly records:", len(hourly_df))
print("\nAverage demand (kW):", round(hourly_df["Average_Demand_kW"].mean(), 3))
print("Maximum hourly average demand (kW):", round(hourly_df["Average_Demand_kW"].max(), 3))
print("Minimum hourly average demand (kW):", round(hourly_df["Average_Demand_kW"].min(), 3))

print("\nHourly dataset saved to:", output_file)
