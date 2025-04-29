import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# %% [1. Introduction to Regression Metrics]
# Regression metrics evaluate the performance of regression models.
# Scikit-learn provides mean_squared_error, mean_absolute_error, and r2_score.

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Generate synthetic dataset: 100 samples, 2 features (size, rooms), 1 target (price).
np.random.seed(42)
data = {
    'size': np.random.uniform(500, 2500, 100),
    'rooms': np.random.randint(1, 5, 100)
}
df = pd.DataFrame(data)
# Target: price = 100*size + 20000*rooms + noise
y = 100 * df['size'] + 20000 * df['rooms'] + np.random.normal(0, 10000, 100)
X = df.values

print("\nSynthetic Dataset (first 5 rows):")
print(df.head())
print("\nTarget (first 5):", y[:5])

# %% [3. Train-Test Split and Model Training]
# Split data and train a LinearRegression model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
reg = LinearRegression()
reg.fit(X_train, y_train)
y_pred = reg.predict(X_test)
print("\nPredicted Values (first 5):", y_pred[:5])

# %% [4. Mean Squared Error]
# MSE measures the average squared difference between predicted and actual values.

mse = mean_squared_error(y_test, y_pred)
print("\nMean Squared Error:", mse.round(4))

# %% [5. Mean Absolute Error]
# MAE measures the average absolute difference between predicted and actual values.

mae = mean_absolute_error(y_test, y_pred)
print("\nMean Absolute Error:", mae.round(4))

# %% [6. R² Score]
# R² measures the proportion of variance explained by the model.

r2 = r2_score(y_test, y_pred)
print("\nR² Score:", r2.round(4))

# %% [7. Practical Application: Comparing Models]
# Compare LinearRegression with a baseline (mean predictor).

baseline_pred = np.full_like(y_test, y_train.mean())
print("\nBaseline (Mean Predictor) Metrics:")
print("MSE:", mean_squared_error(y_test, baseline_pred).round(4))
print("MAE:", mean_absolute_error(y_test, baseline_pred).round(4))
print("R²:", r2_score(y_test, baseline_pred).round(4))
print("\nLinearRegression Metrics:")
print("MSE:", mse.round(4))
print("MAE:", mae.round(4))
print("R²:", r2.round(4))

# %% [8. Outlier Impact]
# Introduce outliers and evaluate metrics.

y_test_outlier = y_test.copy()
y_test_outlier[0] = y_test_outlier[0] * 10  # Extreme outlier
print("\nMetrics with Outlier:")
print("MSE:", mean_squared_error(y_test_outlier, y_pred).round(4))
print("MAE:", mean_absolute_error(y_test_outlier, y_pred).round(4))
print("R²:", r2_score(y_test_outlier, y_pred).round(4))

# %% [9. Interview Scenario: Choosing Metrics]
# Evaluate model with different metrics to justify choice.

print("\nMetric Choice for Regression:")
print("MSE: Sensitive to outliers, good for optimization.")
print("MAE: Robust to outliers, interpretable as average error.")
print("R²: Measures explained variance, useful for model comparison.")

# %% [10. Visualizing Predictions]
# Plot actual vs. predicted values for intuition.

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(y_test, y_pred, color='blue', label='Predicted vs. Actual')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', label='Perfect Prediction')
plt.xlabel('Actual Values')
plt.ylabel('Predicted Values')
plt.title('Regression Predictions')
plt.legend()
plt.savefig('regression_predictions.png')