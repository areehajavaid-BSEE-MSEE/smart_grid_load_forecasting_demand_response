import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load prepared forecasting data

file_path = Path("data/processed/forecasting_features.csv")
df = pd.read_csv(file_path, parse_dates=["Datetime"])

# Define input features and prediction target

features = [
"Hour",
"DayOfWeek",
"Month",
"IsWeekend",
"Demand_1_Hour_Ago",
"Demand_24_Hours_Ago",
"Demand_168_Hours_Ago"
]

target = "Average_Demand_kW"

X = df[features]
y = df[target]

# Split chronologically: first 80% for training, last 20% for testing

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("Training records:", len(X_train))
print("Testing records:", len(X_test))

# Train Random Forest model

print("\nTraining Random Forest model...")

model = RandomForestRegressor(
n_estimators=100,
max_depth=15,
min_samples_leaf=2,
random_state=42,
n_jobs=-1
)

model.fit(X_train, y_train)

# Generate predictions

predictions = model.predict(X_test)

# Evaluate model performance

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))
r2 = r2_score(y_test, predictions)

print("\nRANDOM FOREST MODEL RESULTS")
print(f"MAE:  {mae:.4f} kW")
print(f"RMSE: {rmse:.4f} kW")
print(f"R²:   {r2:.4f}")

# Compare with a persistence baseline:

# predict the next hour using the previous hour's demand

baseline_predictions = X_test["Demand_1_Hour_Ago"]

baseline_mae = mean_absolute_error(y_test, baseline_predictions)
baseline_rmse = np.sqrt(
mean_squared_error(y_test, baseline_predictions)
)

print("\nPERSISTENCE BASELINE RESULTS")
print(f"MAE:  {baseline_mae:.4f} kW")
print(f"RMSE: {baseline_rmse:.4f} kW")

# Save actual and predicted values

results_folder = Path("results")
results_folder.mkdir(parents=True, exist_ok=True)

results = pd.DataFrame({
"Datetime": df["Datetime"].iloc[split_index:].values,
"Actual_Demand_kW": y_test.values,
"Predicted_Demand_kW": predictions,
"Baseline_Demand_kW": baseline_predictions.values
})

results.to_csv(
results_folder / "forecast_results.csv",
index=False
)

# Save feature importance

importance = pd.DataFrame({
"Feature": features,
"Importance": model.feature_importances_
}).sort_values("Importance", ascending=False)

importance.to_csv(
results_folder / "feature_importance.csv",
index=False
)

print("\nForecast results saved to results/forecast_results.csv")
print("Feature importance saved to results/feature_importance.csv")
