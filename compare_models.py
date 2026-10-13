import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Load the dataset

df = pd.read_csv(
"data/processed/forecasting_features.csv",
parse_dates=["Datetime"]
)

# Select input features and target

features = [
"Hour",
"DayOfWeek",
"Month",
"IsWeekend",
"Demand_1_Hour_Ago",
"Demand_24_Hours_Ago",
"Demand_168_Hours_Ago"
]

X = df[features]
y = df["Average_Demand_kW"]

# Split data chronologically: first 80% for training, final 20% for testing

split_index = int(len(df) * 0.8)

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]
y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

# Create both machine-learning models

random_forest = RandomForestRegressor(
n_estimators=100,
max_depth=15,
min_samples_leaf=2,
random_state=42,
n_jobs=-1
)

hist_gradient = HistGradientBoostingRegressor(
max_iter=100,
max_leaf_nodes=31,
learning_rate=0.1,
random_state=42
)

# Train and evaluate Random Forest

print("\nTraining Random Forest...")
random_forest.fit(X_train, y_train)
rf_predictions = random_forest.predict(X_test)

rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_rmse = mean_squared_error(y_test, rf_predictions) ** 0.5
rf_r2 = r2_score(y_test, rf_predictions)

# Train and evaluate HistGradientBoosting

print("\nTraining HistGradientBoosting...")
hist_gradient.fit(X_train, y_train)
hg_predictions = hist_gradient.predict(X_test)

hg_mae = mean_absolute_error(y_test, hg_predictions)
hg_rmse = mean_squared_error(y_test, hg_predictions) ** 0.5
hg_r2 = r2_score(y_test, hg_predictions)

# Store both models' results

results = [
{
"Model": "Random Forest",
"MAE": rf_mae,
"RMSE": rf_rmse,
"R2": rf_r2
},
{
"Model": "HistGradientBoosting",
"MAE": hg_mae,
"RMSE": hg_rmse,
"R2": hg_r2
}
]

# Display comparison

results_df = pd.DataFrame(results)
results_df = results_df.sort_values("MAE")

print("\nMODEL PERFORMANCE COMPARISON")
print(results_df.to_string(index=False))

# Save results

output_path = Path("results/model_comparison.csv")
output_path.parent.mkdir(parents=True, exist_ok=True)
results_df.to_csv(output_path, index=False)

print(f"\nComparison saved to: {output_path}")
print("\nBoth models have been evaluated successfully.")
