import yfinance as yf
from fredapi import Fred
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
import matplotlib.pyplot as plt

# =========================
# STEP 1: S&P 500 DATA
# =========================
sp500 = yf.download("^GSPC", start="2010-01-01", interval="1mo")
sp500.columns = sp500.columns.get_level_values(0)
sp500["Return"] = sp500["Close"].pct_change()

# keep only what we need
sp500 = sp500[["Return"]]

# =========================
# STEP 2: MACRO DATA
# =========================
fred = Fred(api_key="95457e4328a17f40f4d84fc26fd668b7")

inflation = fred.get_series("CPIAUCSL")
unemployment = fred.get_series("UNRATE")
interest = fred.get_series("FEDFUNDS")

# convert to DataFrames
inflation = inflation.to_frame(name="Inflation")
unemployment = unemployment.to_frame(name="Unemployment")
interest = interest.to_frame(name="Interest")

# =========================
# STEP 3: MERGE DATA
# =========================
# combine macro data first
macro = inflation.join(unemployment, how="inner")
macro = macro.join(interest, how="inner")

# merge with stock data
df = sp500.join(macro, how="inner")

# =========================
# STEP 4: CLEAN DATA
# =========================
df = df.dropna()

print("\nFINAL DATASET:")
print(df.head())



# =========================
# STEP 4.5: ADD LAG FEATURES
# =========================

df["Inflation_lag1"] = df["Inflation"].shift(1)
df["Unemployment_lag1"] = df["Unemployment"].shift(1)
df["Interest_lag1"] = df["Interest"].shift(1)

df = df.dropna()

# =========================
# STEP 5: MODEL (LINEAR REGRESSION)
# =========================


# define features (X) and target (y)
X = df[[
    "Inflation", "Unemployment", "Interest",
    "Inflation_lag1", "Unemployment_lag1", "Interest_lag1"
]]
y = df["Return"]

# split data into train and test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# create model
model = LinearRegression()

# train model
model.fit(X_train, y_train)

# predictions
predictions = model.predict(X_test)

print("\nMODEL COEFFICIENTS:")
print(model.coef_)

print("\nMODEL INTERCEPT:")
print(model.intercept_)

# =========================
# STEP 6: MODEL EVALUATION
# =========================

mse = mean_squared_error(y_test, predictions)

print("\nMEAN SQUARED ERROR:")
print(mse)


# =========================
# STEP 7: XGBOOST MODEL
# =========================

xgb = XGBRegressor()

# train
xgb.fit(X_train, y_train)

# predict
xgb_preds = xgb.predict(X_test)

# evaluate
xgb_mse = mean_squared_error(y_test, xgb_preds)

print("\nXGBOOST MSE:")
print(xgb_mse)

# =========================
# STEP 8: VISUALIZATION (ACTUAL VS PREDICTED)
# =========================

plt.figure()

plt.plot(y_test.values, label="Actual")
plt.plot(predictions, label="Predicted")

plt.legend()
plt.title("Actual vs Predicted S&P 500 Returns")

plt.show()


# =========================
# STEP 9: FEATURE IMPORTANCE
# =========================

importances = xgb.feature_importances_

feature_names = X.columns

plt.figure()
plt.bar(feature_names, importances)

plt.title("Feature Importance (XGBoost)")
plt.xticks(rotation=45)

plt.show()

# =========================
# STEP 10: COMPARE MODELS VISUALLY
# =========================

plt.figure()

# Actual
plt.plot(y_test.values, label="Actual")

# Linear Regression
plt.plot(predictions, label="Linear Regression")

# XGBoost
plt.plot(xgb_preds, label="XGBoost")

plt.legend()
plt.title("Model Comparison: Actual vs Predictions")

plt.show()
