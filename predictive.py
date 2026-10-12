# TASK 3 - PREDICTIVE ANALYTICS USING HISTORICAL DATA

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# STEP 1: LOAD DATASET

file_name = "historical_sales_data.csv"

df = pd.read_csv(file_name)

print("\n========== DATASET PREVIEW ==========")
print(df.head())

print("\n========== DATASET SHAPE ==========")
print(df.shape)

print("\n========== DATASET INFO ==========")
df.info()

print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

print("\n========== DUPLICATE ROWS ==========")
print(df.duplicated().sum())

# STEP 2: DATA CLEANING

df["Date"] = pd.to_datetime(df["Date"], errors="coerce")

numeric_columns = [
    "Sales",
    "Orders",
    "Marketing_Spend",
    "Discount_Rate",
    "Returns",
    "Sales_Growth_%"
]

for column in numeric_columns:
    df[column] = pd.to_numeric(df[column], errors="coerce")

# Remove invalid rows
df = df.dropna(subset=["Date", "Sales"])

# Remove duplicate dates
df = df.drop_duplicates(subset=["Date"])

# Sort by date
df = df.sort_values("Date").reset_index(drop=True)

print("\n========== DATA AFTER CLEANING ==========")
print(df.head())

print("\nRows after cleaning:", len(df))

# STEP 3: CREATE FIGURES FOLDER

os.makedirs("figures", exist_ok=True)

# STEP 4: HISTORICAL SALES TREND

plt.figure(figsize=(12, 6))

plt.plot(
    df["Date"],
    df["Sales"],
    marker="o"
)

plt.title("Historical Monthly Sales Trend")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    "figures/historical_sales_trend.png"
)

plt.show()

# STEP 5: YEARLY SALES ANALYSIS

df["Year"] = df["Date"].dt.year
df["Month"] = df["Date"].dt.month

yearly_sales = (
    df.groupby("Year")["Sales"]
    .sum()
    .reset_index()
)

print("\n========== YEARLY SALES ==========")
print(yearly_sales)


plt.figure(figsize=(9, 5))

plt.bar(
    yearly_sales["Year"].astype(str),
    yearly_sales["Sales"]
)

plt.title("Yearly Sales")
plt.xlabel("Year")
plt.ylabel("Total Sales")

plt.tight_layout()

plt.savefig(
    "figures/yearly_sales.png"
)

plt.show()

# STEP 6: MONTHLY SALES PATTERN

monthly_sales = (
    df.groupby("Month")["Sales"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 5))

plt.plot(
    monthly_sales["Month"],
    monthly_sales["Sales"],
    marker="o"
)

plt.title("Average Monthly Sales Pattern")
plt.xlabel("Month")
plt.ylabel("Average Sales")

plt.xticks(range(1, 13))

plt.tight_layout()

plt.savefig(
    "figures/monthly_sales_pattern.png"
)

plt.show()



# STEP 7: FEATURE ENGINEERING

# Past sales values
df["Sales_Lag_1"] = df["Sales"].shift(1)
df["Sales_Lag_2"] = df["Sales"].shift(2)
df["Sales_Lag_3"] = df["Sales"].shift(3)

# Rolling average using only previous values
df["Rolling_Mean_3"] = (
    df["Sales"]
    .shift(1)
    .rolling(3)
    .mean()
)

# Time features
df["Month_Sin"] = np.sin(
    2 * np.pi * df["Month"] / 12
)

df["Month_Cos"] = np.cos(
    2 * np.pi * df["Month"] / 12
)

# STEP 8: PREPARE MODEL DATA

features = [
    "Sales_Lag_1",
    "Sales_Lag_2",
    "Sales_Lag_3",
    "Rolling_Mean_3",
    "Month_Sin",
    "Month_Cos"
]

target = "Sales"

model_data = df.dropna(
    subset=features + [target]
).copy()

X = model_data[features]
y = model_data[target]

# STEP 9: TRAIN-TEST SPLIT

# Last 12 months reserved for testing
test_size = 12

split_index = len(model_data) - test_size

X_train = X.iloc[:split_index]
X_test = X.iloc[split_index:]

y_train = y.iloc[:split_index]
y_test = y.iloc[split_index:]

print("\n========== TRAIN TEST SPLIT ==========")

print("Training records:", len(X_train))
print("Testing records :", len(X_test))

# STEP 10: TRAIN MACHINE LEARNING MODEL

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42,
    max_depth=8
)

model.fit(
    X_train,
    y_train
)

print("\n========== MODEL TRAINED ==========")
print("Model: Random Forest Regression")

# STEP 11: PREDICT TEST DATA

y_pred = model.predict(X_test)

# STEP 12: MODEL EVALUATION

mae = mean_absolute_error(
    y_test,
    y_pred
)

rmse = np.sqrt(
    mean_squared_error(
        y_test,
        y_pred
    )
)

r2 = r2_score(
    y_test,
    y_pred
)

print("\n========== MODEL EVALUATION ==========")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

# STEP 13: ACTUAL VS PREDICTED

prediction_results = pd.DataFrame({
    "Date": model_data.iloc[split_index:]["Date"].values,
    "Actual_Sales": y_test.values,
    "Predicted_Sales": np.round(y_pred, 2)
})

print("\n========== ACTUAL VS PREDICTED ==========")
print(prediction_results)


plt.figure(figsize=(12, 6))

plt.plot(
    prediction_results["Date"],
    prediction_results["Actual_Sales"],
    marker="o",
    label="Actual Sales"
)

plt.plot(
    prediction_results["Date"],
    prediction_results["Predicted_Sales"],
    marker="o",
    label="Predicted Sales"
)

plt.title("Actual vs Predicted Sales")
plt.xlabel("Date")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/actual_vs_predicted.png"
)

plt.show()

# STEP 14: FUTURE 12-MONTH FORECAST

forecast_months = 12

history = df["Sales"].tolist()

last_date = df["Date"].max()

future_dates = pd.date_range(
    start=last_date + pd.DateOffset(months=1),
    periods=forecast_months,
    freq="MS"
)

future_predictions = []


for future_date in future_dates:

    lag_1 = history[-1]
    lag_2 = history[-2]
    lag_3 = history[-3]

    rolling_mean = np.mean(
        history[-3:]
    )

    month = future_date.month

    month_sin = np.sin(
        2 * np.pi * month / 12
    )

    month_cos = np.cos(
        2 * np.pi * month / 12
    )

    future_input = pd.DataFrame({
        "Sales_Lag_1": [lag_1],
        "Sales_Lag_2": [lag_2],
        "Sales_Lag_3": [lag_3],
        "Rolling_Mean_3": [rolling_mean],
        "Month_Sin": [month_sin],
        "Month_Cos": [month_cos]
    })

    prediction = model.predict(
        future_input
    )[0]

    prediction = max(
        0,
        prediction
    )

    future_predictions.append(
        round(prediction, 2)
    )

    # Add prediction to history
    history.append(prediction)

# STEP 15: FORECAST TABLE

future_df = pd.DataFrame({
    "Date": future_dates,
    "Predicted_Sales": future_predictions
})

print("\n========== NEXT 12 MONTH FORECAST ==========")
print(future_df.to_string(index=False))

# STEP 16: HISTORICAL + FORECAST GRAPH

plt.figure(figsize=(14, 6))

plt.plot(
    df["Date"],
    df["Sales"],
    marker="o",
    label="Historical Sales"
)

plt.plot(
    future_df["Date"],
    future_df["Predicted_Sales"],
    marker="o",
    label="Forecasted Sales"
)

plt.title(
    "Historical Sales and Future Sales Forecast"
)

plt.xlabel("Date")
plt.ylabel("Sales")

plt.xticks(rotation=45)

plt.legend()

plt.tight_layout()

plt.savefig(
    "figures/future_sales_forecast.png"
)

plt.show()

# STEP 17: FORECAST SUMMARY

historical_average = df["Sales"].mean()

forecast_average = (
    future_df["Predicted_Sales"].mean()
)

forecast_change = (
    (forecast_average - historical_average)
    / historical_average
) * 100

print("\n========== FORECAST SUMMARY ==========")

print(
    f"Average Historical Sales : "
    f"{historical_average:.2f}"
)

print(
    f"Average Forecast Sales   : "
    f"{forecast_average:.2f}"
)

print(
    f"Forecast Change          : "
    f"{forecast_change:.2f}%"
)

# STEP 18: FEATURE IMPORTANCE

feature_importance = pd.DataFrame({
    "Feature": features,
    "Importance": model.feature_importances_
})

feature_importance = (
    feature_importance
    .sort_values(
        "Importance",
        ascending=False
    )
)

print("\n========== FEATURE IMPORTANCE ==========")
print(feature_importance)


plt.figure(figsize=(9, 5))

plt.barh(
    feature_importance["Feature"],
    feature_importance["Importance"]
)

plt.title("Feature Importance")
plt.xlabel("Importance")

plt.tight_layout()

plt.savefig(
    "figures/feature_importance.png"
)

plt.show()

# STEP 19: SAVE OUTPUT FILES

prediction_results.to_csv(
    "prediction_results.csv",
    index=False
)

future_df.to_csv(
    "future_sales_forecast.csv",
    index=False
)

model_data.to_csv(
    "model_ready_data.csv",
    index=False
)

# STEP 20: FINAL SUMMARY

print("\n============================================")
print("PREDICTIVE ANALYTICS PROJECT COMPLETED")
print("============================================")

print("\nModel Used: Random Forest Regression")

print(f"MAE  : {mae:.2f}")
print(f"RMSE : {rmse:.2f}")
print(f"R²   : {r2:.3f}")

print("\nGenerated Files:")
print("1. model_ready_data.csv")
print("2. prediction_results.csv")
print("3. future_sales_forecast.csv")
print("4. figures/historical_sales_trend.png")
print("5. figures/yearly_sales.png")
print("6. figures/monthly_sales_pattern.png")
print("7. figures/actual_vs_predicted.png")
print("8. figures/future_sales_forecast.png")
print("9. figures/feature_importance.png")

print("\n============================================")