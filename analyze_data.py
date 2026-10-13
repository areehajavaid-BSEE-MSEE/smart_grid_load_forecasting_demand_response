import pandas as pd
from pathlib import Path

# Locate the dataset

file_path = Path("data/household_power_consumption.txt")

# Check whether the dataset exists

if not file_path.exists():
      raise FileNotFoundError(
         "Dataset not found. Check the data folder."
)

print("Loading electricity consumption dataset...")

# Load the dataset using the correct column headers

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

df = pd.read_csv(
file_path,
sep=";",
header=0,
names=column_names,
na_values=["?"],
low_memory=False
)

# Convert electricity measurements to numeric values

numeric_columns = column_names[2:]

for column in numeric_columns:
       df[column] = pd.to_numeric(df[column], errors="coerce")

# Combine Date and Time into a single datetime column

df["Datetime"] = pd.to_datetime(
df["Date"] + " " + df["Time"],
format="%d/%m/%Y %H:%M:%S",
errors="coerce"
)

# Remove rows with invalid timestamps or missing active power

df = df.dropna(subset=["Datetime", "Global_active_power"])

# Sort observations chronologically

df = df.sort_values("Datetime")

print("\nDataset cleaned successfully!")
print("\nFirst five rows:")
print(df.head())

print("\nDataset shape after cleaning:")
print(df.shape)

print("\nMissing values after cleaning:")
print(df.isnull().sum())

print("\nDate range:")
print(df["Datetime"].min(), "to", df["Datetime"].max())
