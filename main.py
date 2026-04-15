import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

from fredapi import Fred
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor

# =========================
# STEP 1: GET S&P 500 DATA
# =========================
print("Downloading S&P 500 data...")

sp500 = yf.download("^GSPC", start="2010-01-01", interval="1mo", auto_adjust=True)
sp500.columns = sp500.columns.get_level_values(0)

if sp500.empty:
    print("ERROR: Failed to download S&P 500 data")
    exit()

sp500["Return"] = sp500["Close"].pct_change()

# =========================
# STEP 2: GET MACRO DATA
# =========================
fred = Fred(api_key="95457e4328a17f40f4d84fc26fd668b7")  # keep your key

inflation = fred.get_series("CPIAUCSL")
unemployment = fred.get_series("UNRATE")
interest = fred.get_series("FEDFUNDS")

# convert to dataframe
macro = pd.DataFrame({
    "Inflation": inflation,
    "Unemployment": unemployment,
    "Interest": interest
})

# =========================
# STEP 3: MERGE DATA
# =========================
sp500 = sp500[["Return"]]

df = sp500.join(macro, how="inner")

df = df.dropna()

# =========================
# STEP 4: ADD LAG FEATURES
# =========================
df["Inflation_lag1"] = df["Inflation"].shift(1)
df["Unemployment_lag1"] = df["Unemployment"].shift(1)
df["Interest_lag1"] = df["Interest"].shift(1)

df = df.dropna()

print("\nFINAL DATASET:")
print(df.head())

# =========================
# STEP 5: MODEL
# =========================
X = df[[
    "Inflation", "Unemployment", "Interest",
    "Inflation_lag1", "Unemployment_lag1", "Interest_lag1"
]]

y = df["Return"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=False
)

# Linear Regression
model = LinearRegression()
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# XGBoost
xgb = XGBRegressor()
xgb.fit(X_train, y_train)
xgb_preds = xgb.predict(X_test)

# =========================
# STEP 6: EVALUATION
# =========================
mse = mean_squared_error(y_test, predictions)
xgb_mse = mean_squared_error(y_test, xgb_preds)

print("\nLINEAR MSE:", mse)
print("XGBOOST MSE:", xgb_mse)

# =========================
# STEP 7: GRAPH 1 (COMPARISON)
# =========================
plt.figure(figsize=(8,5))

plt.plot(y_test.values, label="Actual")
plt.plot(predictions, label="Linear Regression")
plt.plot(xgb_preds, label="XGBoost")

plt.legend()
plt.title("Model Comparison: Actual vs Predictions")

plt.tight_layout()
plt.savefig("images/model_comparison.png")
plt.close()

# =========================
# STEP 8: GRAPH 2 (FEATURE IMPORTANCE)
# =========================
importances = xgb.feature_importances_
feature_names = list(X.columns)

plt.figure(figsize=(8,5))

plt.bar(feature_names, importances)
plt.title("Feature Importance (XGBoost)")
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("images/feature_importance.png")
plt.close()

print("\nGraphs saved successfully!")




