
# House Price Prediction Model

# 1. Data Loading and Preprocessing
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None
    print("XGBoost is not installed. Please install it using `pip install xgboost` to run the XGBoost model.")
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import os

# Load the dataset
file_path = 'model_training/data/train.csv'
if 'd:' not in file_path and os.path.exists('d:/House-predict/' + file_path):
    file_path = 'd:/House-predict/' + file_path
    
df = pd.read_csv(file_path)

print("Initial Data Shape:", df.shape)

# Data Preprocessing
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
categorical_cols = df.select_dtypes(include=['object']).columns

# Fill missing numerical values with median
for col in numeric_cols:
    df[col] = df[col].fillna(df[col].median())

# Fill missing categorical values with mode
for col in categorical_cols:
    if not df[col].mode().empty:
        df[col] = df[col].fillna(df[col].mode()[0])
    else:
        df[col] = df[col].fillna('Missing')

# Drop 'Id' column
if 'Id' in df.columns:
    df = df.drop('Id', axis=1)

# One-hot encoding
df = pd.get_dummies(df, drop_first=True)
print("Shape after preprocessing:", df.shape)

# 2. Visualization
# Heatmap
plt.figure(figsize=(12, 10))
corrmat = df.corr()
top_corr_features = corrmat.index[abs(corrmat['SalePrice']) > 0.5]
sns.heatmap(df[top_corr_features].corr(), annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap (Features > 0.5 correlation with SalePrice)')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')
print("Saved correlation_heatmap.png")
# plt.show()

# Scatter Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x='GrLivArea', y='SalePrice', data=df)
plt.title('Living Area vs Sale Price')
plt.xlabel('Ground Living Area (sq ft)')
plt.ylabel('Sale Price ($)')
plt.tight_layout()
plt.savefig('scatter_plot.png')
print("Saved scatter_plot.png")
# plt.show()

# 3. Model Training
X = df.drop('SalePrice', axis=1)
y = df['SalePrice']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Simple Linear Regression
print("\n--- Simple Linear Regression ---")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
lr_pred = lr_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, lr_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, lr_pred)))
print("R2 Score:", r2_score(y_test, lr_pred))

# Random Forest Regression
print("\n--- Random Forest Regression ---")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
rf_pred = rf_model.predict(X_test)
print("MAE:", mean_absolute_error(y_test, rf_pred))
print("RMSE:", np.sqrt(mean_squared_error(y_test, rf_pred)))
print("R2 Score:", r2_score(y_test, rf_pred))

# XGBoost Regression
print("\n--- XGBoost Regression ---")
if XGBRegressor:
    xgb_model = XGBRegressor(n_estimators=1000, learning_rate=0.05, n_jobs=4, random_state=42)
    xgb_model.fit(X_train, y_train)
    xgb_pred = xgb_model.predict(X_test)
    print("MAE:", mean_absolute_error(y_test, xgb_pred))
    print("RMSE:", np.sqrt(mean_squared_error(y_test, xgb_pred)))
    print("R2 Score:", r2_score(y_test, xgb_pred))
else:
    print("Skipping XGBoost (module not found).")
    xgb_pred = None

# 4. Comparison Graph
plt.figure(figsize=(10, 6))
plt.scatter(y_test, rf_pred, alpha=0.5, color='blue', label='Random Forest Predictions')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Ideal')
plt.title('Actual vs Predicted Prices (Random Forest)')
plt.xlabel('Actual Sale Price')
plt.ylabel('Predicted Sale Price')
plt.legend()
plt.tight_layout()
plt.savefig('actual_vs_predicted.png')
print("Saved actual_vs_predicted.png")
# plt.show()
