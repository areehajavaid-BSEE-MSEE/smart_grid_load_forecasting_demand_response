import pandas as pd
from pathlib import Path

# Load model comparison results

comparison = pd.read_csv("results/model_comparison.csv")

# Load feature importance results

importance = pd.read_csv("results/feature_importance.csv")
importance = importance.sort_values("Importance", ascending=False)

# Display model results

print("\nSMART GRID LOAD FORECASTING PROJECT")
print("=" * 45)

print("\n1. MODEL PERFORMANCE")
print(comparison.to_string(index=False))

# Identify the model with the lowest MAE

best_model = comparison.loc[comparison["MAE"].idxmin()]

print("\n2. BEST MODEL")
print("Model:", best_model["Model"])
print(f"MAE: {best_model['MAE']:.4f} kW")
print(f"RMSE: {best_model['RMSE']:.4f} kW")
print(f"R2: {best_model['R2']:.4f}")

# Display the three most important features

print("\n3. TOP THREE FEATURES")
print(importance.head(3).to_string(index=False))

# Save a copy of the model comparison

output_path = Path("results/project_summary.csv")
comparison.to_csv(output_path, index=False)

print("\nSummary results saved to:", output_path)
