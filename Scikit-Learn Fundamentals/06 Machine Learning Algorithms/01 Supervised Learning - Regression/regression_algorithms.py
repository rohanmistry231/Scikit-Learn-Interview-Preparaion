import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# %% [1. Introduction to Regression Algorithms]
# Regression predicts continuous target variables using supervised learning.
# Scikit-learn provides LinearRegression, Ridge, Lasso, SVR, and ElasticNet.

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Generate synthetic dataset: 200 samples, 3 features, 1 target.
X, y = make_regression(n_samples=200, n_features=3, noise=10, random_state=42)
feature_names = ['feature1', 'feature2', 'feature3']
df = pd.DataFrame(X, columns=feature_names)
df['target'] = y
print("\nSynthetic Dataset (first 5 rows):")
print(df.head())
print("\nDataset Shape:", X.shape)

# %% [3. Train-Test Split]
# Split data into training (80%) and testing (20%) sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nTrain Shape:", X_train.shape, "Test Shape:", X_test.shape)

# %% [4. LinearRegression]
# LinearRegression fits a linear model to minimize squared errors.
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\nLinearRegression:")
print("MSE:", mean_squared_error(y_test, y_pred_lr).round(4))
print("R²:", r2_score(y_test, y_pred_lr).round(4))
print("Coefficients:", lr.coef_.round(4))

# %% [5. Ridge]
# Ridge adds L2 regularization to prevent overfitting.
ridge = Ridge(alpha=1.0, random_state=42)
ridge.fit(X_train, y_train)
y_pred_ridge = ridge.predict(X_test)
print("\nRidge:")
print("MSE:", mean_squared_error(y_test, y_pred_ridge).round(4))
print("R²:", r2_score(y_test, y_pred_ridge).round(4))
print("Coefficients:", ridge.coef_.round(4))

# %% [6. Lasso]
# Lasso adds L1 regularization, promoting sparsity.
lasso = Lasso(alpha=1.0, random_state=42)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_test)
print("\nLasso:")
print("MSE:", mean_squared_error(y_test, y_pred_lasso).round(4))
print("R²:", r2_score(y_test, y_pred_lasso).round(4))
print("Coefficients:", lasso.coef_.round(4))

# %% [7. SVR]
# SVR uses support vector machines for regression.
svr = SVR(kernel='rbf', C=1.0)
svr.fit(X_train, y_train)
y_pred_svr = svr.predict(X_test)
print("\nSVR:")
print("MSE:", mean_squared_error(y_test, y_pred_svr).round(4))
print("R²:", r2_score(y_test, y_pred_svr).round(4))

# %% [8. ElasticNet]
# ElasticNet combines L1 and L2 regularization.
enet = ElasticNet(alpha=1.0, l1_ratio=0.5, random_state=42)
enet.fit(X_train, y_train)
y_pred_enet = enet.predict(X_test)
print("\nElasticNet:")
print("MSE:", mean_squared_error(y_test, y_pred_enet).round(4))
print("R²:", r2_score(y_test, y_pred_enet).round(4))
print("Coefficients:", enet.coef_.round(4))

# %% [9. Practical Application: Comparing Models]
# Compare all models on test set performance.
models = {
    'LinearRegression': y_pred_lr,
    'Ridge': y_pred_ridge,
    'Lasso': y_pred_lasso,
    'SVR': y_pred_svr,
    'ElasticNet': y_pred_enet
}
print("\nModel Comparison:")
for name, y_pred in models.items():
    print(f"{name}: MSE={mean_squared_error(y_test, y_pred).round(4)}, R²={r2_score(y_test, y_pred).round(4)}")

# %% [10. Interview Scenario: Regularization Impact]
# Demonstrate regularization’s effect on coefficients.
print("\nRegularization Impact:")
print("LinearRegression Coefs:", lr.coef_.round(4))
print("Ridge Coefs:", ridge.coef_.round(4))
print("Lasso Coefs:", lasso.coef_.round(4))
print("Lasso sets some coefficients to zero (sparsity), Ridge shrinks them.")

# Plot predictions for one feature
plt.figure()
plt.scatter(X_test[:, 0], y_test, color='black', label='Actual')
plt.scatter(X_test[:, 0], y_pred_lr, color='blue', label='LinearRegression')
plt.scatter(X_test[:, 0], y_pred_ridge, color='green', label='Ridge')
plt.xlabel('Feature 1')
plt.ylabel('Target')
plt.title('Regression Predictions')
plt.legend()
plt.savefig('regression_plot.png')