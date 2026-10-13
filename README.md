# Smart Grid Load Forecasting and Demand Response

## Project Overview

This project uses machine learning to analyze historical electricity consumption and forecast short-term electricity demand. The goal is to explore how demand forecasting can support smarter energy management and demand response planning.

## Objectives

* Analyze historical electricity consumption data.
* Identify daily electricity demand patterns.
* Develop forecasting features using historical demand and time information.
* Train and compare machine-learning models.
* Identify the most influential forecasting features.
* Explore how demand forecasting can support smart-grid energy management.

## Dataset

**Source:** UCI Machine Learning Repository — Individual Household Electric Power Consumption Dataset.

The dataset contains household electricity measurements collected between December 2006 and November 2010.

Dataset link: https://archive.ics.uci.edu/dataset/235/individual+household+electric+power+consumption

## Technologies Used

* Python
* Pandas and NumPy
* Matplotlib
* Scikit-learn
* Machine Learning
* Time-series feature engineering

## Methodology

1. Clean and preprocess electricity consumption data.
2. Aggregate measurements into hourly average electricity demand.
3. Analyze daily electricity consumption patterns.
4. Generate forecasting features using hourly, daily, and weekly historical demand.
5. Train and evaluate Random Forest and HistGradientBoosting models using a chronological training and testing split.
6. Compare model performance using MAE, RMSE, and R².
7. Analyze feature importance.

## Model Performance

| Model                | MAE (kW) | RMSE (kW) |     R² |
| -------------------- | -------: | --------: | -----: |
| Random Forest        |   0.3402 |    0.4913 | 0.5630 |
| HistGradientBoosting |   0.3353 |    0.4819 | 0.5796 |

HistGradientBoosting achieved the lowest MAE and RMSE and the highest R² among the two evaluated models.

## Key Findings

* Demand in the previous hour was the most influential feature in the Random Forest model.
* Hour of the day contributed to forecasting performance.
* Demand from the same hour one week earlier also provided useful information.
* HistGradientBoosting performed slightly better than Random Forest on the test dataset.

## Project Structure

```text
smart_grid_load_forecasting_demand_response/
├── data/
│   └── processed/
├── results/
│   ├── figures/
│   ├── forecast_results.csv
│   ├── feature_importance.csv
│   ├── model_comparison.csv
│   └── project_summary.csv
├── main.py
├── analyze_data.py
├── hourly_analysis.py
├── visualize_demand.py
├── daily_demand_analysis.py
├── prepare_forecasting_data.py
├── train_model.py
├── visualize_forecast.py
├── analyze_feature_importance.py
├── compare_models.py
├── visualize_model_comparison.py
└── project_summary.py
```

## Limitations and Future Work

* The dataset represents household electricity consumption rather than an entire electrical grid.
* Forecasting performance can be improved through further feature engineering and model tuning.
* Future work may explore demand response strategies, renewable energy integration, and energy storage coordination.
* Additional evaluation is needed before applying this approach to real-world smart-grid operations.

## Research Relevance

This project supports my research interests in smart grids, machine learning for energy systems, electricity demand forecasting, and intelligent energy management.
