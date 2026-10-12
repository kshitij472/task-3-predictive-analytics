# Predictive Analytics Using Historical Data

## Project Overview

This project uses historical monthly sales data to analyze past trends, train a regression model, evaluate its predictions, and forecast sales for the next 12 months.

The project demonstrates a basic end-to-end predictive analytics workflow using Python and machine learning.

> **Dataset note:** `historical_sales_data.csv` is a sample/simulated dataset created for learning and demonstration. It is not real company data. Forecasts are illustrative and should not be treated as business guarantees.

## Objectives

- Load and inspect historical sales data
- Check missing values and duplicate records
- Clean and preprocess the dataset
- Explore historical, yearly, and monthly sales patterns
- Create time-based and lag-based features
- Train a Random Forest Regression model
- Evaluate the model using MAE, RMSE, and R²
- Compare actual sales with predicted sales
- Forecast sales for the next 12 months
- Visualize results and export output CSV files

## Dataset

The file `historical_sales_data.csv` contains 60 monthly records from January 2021 through December 2025.

| Column | Description |
|---|---|
| `Date` | Month of the recorded observation |
| `Sales` | Monthly sales value |
| `Orders` | Number of orders |
| `Marketing_Spend` | Marketing expenditure |
| `Discount_Rate` | Discount rate for the period |
| `Returns` | Number of returned orders |
| `Sales_Growth_%` | Month-over-month sales growth percentage |

The current model forecasts `Sales` using past sales values and calendar seasonality features.

## Tools and Libraries

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Random Forest Regression

## Project Workflow

### 1. Data Inspection and Cleaning
Load the CSV, inspect its shape and data types, check missing values and duplicates, convert the date and numeric columns, remove invalid records, and sort observations chronologically.

### 2. Exploratory Data Analysis (EDA)
Create visuals for:
- Historical monthly sales trend
- Yearly sales totals
- Average monthly sales pattern

### 3. Feature Engineering
Create time-series features from past sales:
- Previous month's sales
- Sales from two months earlier
- Three-month rolling average
- Month sine and cosine values to represent seasonality

### 4. Model Training
Train a `RandomForestRegressor` using an ordered train/test split. The final 12 usable historical records are held out for testing rather than randomly shuffled.

### 5. Model Evaluation
Evaluate test predictions using:
- **MAE** — Mean Absolute Error
- **RMSE** — Root Mean Squared Error
- **R²** — Coefficient of determination

The scores are printed when the script runs; they are not hard-coded into this README because they depend on the code and dataset version.

### 6. Forecasting
Use the trained model to generate a rolling forecast for the next 12 months. Predicted values are fed back as lag inputs for subsequent forecast months.

### 7. Export Results
Save prediction results, future forecasts, feature importance, and charts for review.

## Folder Structure

```text
Task 3 - Predictive Analytics/
├── predictive_analytics.py
├── historical_sales_data.csv
├── model_ready_data.csv
├── prediction_results.csv
├── future_sales_forecast.csv
├── feature_importance.csv
├── figures/
│   ├── historical_sales_trend.png
│   ├── yearly_sales.png
│   ├── monthly_sales_pattern.png
│   ├── actual_vs_predicted.png
│   ├── future_sales_forecast.png
│   └── feature_importance.png
└── README.md
```

The output CSV files and `figures/` images are generated when the script runs. Their presence depends on whether the script has been executed.

## How to Run

### 1. Install the required libraries

```bash
pip install pandas numpy matplotlib scikit-learn
```

### 2. Keep the Python script and dataset in the same folder

Make sure these files are together:
- `predictive_analytics.py`
- `historical_sales_data.csv`

### 3. Run the project

```bash
python predictive_analytics.py
```

The script prints the dataset checks, model evaluation scores, and 12-month forecast. It also creates output CSV files and visualizations.

## Limitations

- The data is simulated for learning purposes.
- Only 60 monthly observations are available, which is limited for robust long-term forecasting.
- Model performance on this sample dataset does not indicate how it would perform on real business data.
- The forecast assumes that historical patterns continue; real sales can be affected by external factors not represented in this dataset.

## Conclusion

This project demonstrates the main stages of predictive analytics: preparing historical data, exploring trends, engineering time-series features, training and evaluating a regression model, and producing a future sales forecast. It provides a foundation for experimenting with real sales datasets and more advanced forecasting techniques.
