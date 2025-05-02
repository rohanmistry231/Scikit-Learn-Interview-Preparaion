import numpy as np
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures, PowerTransformer, FunctionTransformer

# %% [1. Introduction to Feature Transformation]
# Feature transformation creates new features or modifies existing ones to improve model performance.
# Scikit-learn provides PolynomialFeatures, PowerTransformer, and FunctionTransformer.

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Generate synthetic dataset: 100 samples with numerical features (price, quantity).
np.random.seed(42)
data = {
    'price': np.random.uniform(10, 100, 100),
    'quantity': np.random.randint(1, 20, 100)
}
df = pd.DataFrame(data)
X = df.values

print("\nDummy Dataset (first 5 rows):")
print(df.head())

# %% [3. PolynomialFeatures]
# PolynomialFeatures generates polynomial and interaction terms to capture non-linear relationships.

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
df_poly = pd.DataFrame(X_poly, columns=poly.get_feature_names_out(['price', 'quantity']))
print("\nPolynomialFeatures Results (first 5 rows):")
print(df_poly.head())

# %% [4. PowerTransformer]
# PowerTransformer applies a power transformation (e.g., Yeo-Johnson) to make data more Gaussian-like.

pt = PowerTransformer(method='yeo-johnson')
X_pt = pt.fit_transform(X)
df_pt = pd.DataFrame(X_pt, columns=['price', 'quantity'])
print("\nPowerTransformer Results (first 5 rows):")
print(df_pt.head())
print("Mean after Transformation:", X_pt.mean(axis=0).round(4))
print("Std after Transformation:", X_pt.std(axis=0).round(4))

# %% [5. FunctionTransformer]
# FunctionTransformer applies a custom function to transform features.

log_transformer = FunctionTransformer(np.log1p, validate=True)
df['price_log'] = log_transformer.fit_transform(df[['price']])
print("\nFunctionTransformer (Log) Results (first 5 rows):")
print(df[['price', 'price_log']].head())

# %% [6. Practical Application: Transformation in a Pipeline]
# Apply transformations in a train-test split for a regression task.

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

X_train, X_test, y_train, y_test = train_test_split(X, np.random.normal(100, 20, 100), test_size=0.2, random_state=42)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train)
X_test_poly = poly.transform(X_test)

reg = LinearRegression()
reg.fit(X_train_poly, y_train)
y_pred = reg.predict(X_test_poly)
print("\nLinearRegression R² with Polynomial Features:", r2_score(y_test, y_pred).round(4))

# %% [7. Combining Transformations]
# Combine PolynomialFeatures and PowerTransformer for complex feature engineering.

pt = PowerTransformer(method='yeo-johnson')
X_pt = pt.fit_transform(X)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_combined = poly.fit_transform(X_pt)
df_combined = pd.DataFrame(X_combined, columns=poly.get_feature_names_out(['price', 'quantity']))
print("\nCombined Transformation Results (first 5 rows):")
print(df_combined.head())

# %% [8. Interview Scenario: Transformation for Non-linear Data]
# Use PolynomialFeatures to improve a regression model on non-linear data.

reg_no_poly = LinearRegression()
reg_no_poly.fit(X_train, y_train)
y_pred_no_poly = reg_no_poly.predict(X_test)
print("\nLinearRegression R² without Polynomial Features:", r2_score(y_test, y_pred_no_poly).round(4))
print("Improvement with Polynomial Features:", (r2_score(y_test, y_pred) - r2_score(y_test, y_pred_no_poly)).round(4))