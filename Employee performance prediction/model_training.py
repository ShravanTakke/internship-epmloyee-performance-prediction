import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import xgboost as xgb
import joblib

# Load the dataset
df = pd.read_csv("garments_worker_productivity.csv")

# ---------------------- ANALYSIS ----------------------

# Correlation Heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")

# Descriptive Statistics
print("🔹 First 5 Rows of Dataset:")
print(df.head())

print("\n🔹 Statistical Summary:")
print(df.describe())

print("\n🔹 Data Info (nulls and datatypes):")
print(df.info())

print("\n🔹 Department Distribution:")
print(df['department'].value_counts())

# PREPROCESSING

# Check for null values
print("\n🔍 Checking for Null Values:")
print(df.isnull().sum())

# Clean department names
df['department'] = df['department'].str.strip()

# Drop 'date' column
df = df.drop(['date'], axis=1)

# Drop rows with null values (simplest solution)
df = df.dropna()

# One-hot encoding for categorical variables
df = pd.get_dummies(df, drop_first=True)

# TRAIN/TEST SPLIT

X = df.drop('actual_productivity', axis=1)
y = df['actual_productivity']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# MODEL TRAINING

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_preds = lr.predict(X_test)
lr_r2 = r2_score(y_test, lr_preds)
print(f"Linear Regression R² Score: {lr_r2:.4f}")

# Random Forest
rf = RandomForestRegressor(random_state=42)
rf.fit(X_train, y_train)
rf_preds = rf.predict(X_test)
rf_r2 = r2_score(y_test, rf_preds)
print(f"Random Forest R² Score: {rf_r2:.4f}")

# XGBoost
xgb_model = xgb.XGBRegressor()
xgb_model.fit(X_train, y_train)
xgb_preds = xgb_model.predict(X_test)
xgb_r2 = r2_score(y_test, xgb_preds)
print(f"XGBoost R² Score: {xgb_r2:.4f}")

# CHOOSE BEST MODEL

best_model = None
best_r2 = max(lr_r2, rf_r2, xgb_r2)

if best_r2 == lr_r2:
    best_model = lr
    print("\n Best Model: Linear Regression")
elif best_r2 == rf_r2:
    best_model = rf
    print("\n Best Model: Random Forest")
else:
    best_model = xgb_model
    print("\n Best Model: XGBoost")

# SAVE MODEL

joblib.dump(best_model, 'model.pkl')

# Also save the feature order for prediction input later
joblib.dump(X.columns.tolist(), 'feature_order.pkl')

print("\n Model and feature order saved successfully.")

