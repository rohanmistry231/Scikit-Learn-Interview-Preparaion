import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# %% [1. Introduction to Scaling and Normalization]
# Scaling and normalization transform numerical features to a common range or distribution,
# improving model performance for algorithms sensitive to feature scales (e.g., SVM, KNN).

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Generate synthetic dataset: 100 samples, 3 numerical features (e.g., age, income, score).
np.random.seed(42)
data = {
    'age': np.random.randint(18, 80, 100),           # Discrete integers
    'income': np.random.normal(50000, 15000, 100),   # Normally distributed
    'score': np.random.uniform(0, 100, 100)          # Uniformly distributed
}
df = pd.DataFrame(data)
X = df.values  # Feature matrix

print("\nDummy Dataset (first 5 rows):")
print(df.head())

# %% [3. StandardScaler]
# StandardScaler standardizes features to have mean=0 and variance=1.
# Suitable for normally distributed data or algorithms assuming zero-mean features.

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
df_scaled = pd.DataFrame(X_scaled, columns=['age', 'income', 'score'])

print("\nStandardScaler Results (first 5 rows):")
print(df_scaled.head())
print("Mean:", X_scaled.mean(axis=0).round(4))
print("Std:", X_scaled.std(axis=0).round(4))

# %% [4. MinMaxScaler]
# MinMaxScaler scales features to a fixed range, typically [0, 1].
# Useful for algorithms sensitive to bounded ranges (e.g., neural networks).

minmax_scaler = MinMaxScaler(feature_range=(0, 1))
X_minmax = minmax_scaler.fit_transform(X)
df_minmax = pd.DataFrame(X_minmax, columns=['age', 'income', 'score'])

print("\nMinMaxScaler Results (first 5 rows):")
print(df_minmax.head())
print("Min:", X_minmax.min(axis=0).round(4))
print("Max:", X_minmax.max(axis=0).round(4))

# %% [5. RobustScaler]
# RobustScaler uses the median and IQR, making it robust to outliers.
# Ideal for datasets with significant outliers or non-normal distributions.

robust_scaler = RobustScaler()
X_robust = robust_scaler.fit_transform(X)
df_robust = pd.DataFrame(X_robust, columns=['age', 'income', 'score'])

print("\nRobustScaler Results (first 5 rows):")
print(df_robust.head())
print("Median (approx):", np.median(X_robust, axis=0).round(4))

# %% [6. Practical Application: Avoiding Data Leakage]
# Fit scaler only on training data to prevent data leakage in ML pipelines.

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(X, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)  # Use transform, not fit_transform

print("\nTrain Scaled (first 5 rows):")
print(pd.DataFrame(X_train_scaled, columns=['age', 'income', 'score']).head())
print("\nTest Scaled (first 5 rows):")
print(pd.DataFrame(X_test_scaled, columns=['age', 'income', 'score']).head())

# %% [7. Comparing Scalers]
# Compare the effect of different scalers on feature distributions.

print("\nFeature Statistics After Scaling:")
print("StandardScaler - Mean:", X_scaled.mean(axis=0).round(4), "Std:", X_scaled.std(axis=0).round(4))
print("MinMaxScaler - Min:", X_minmax.min(axis=0).round(4), "Max:", X_minmax.max(axis=0).round(4))
print("RobustScaler - Median:", np.median(X_robust, axis=0).round(4))

# %% [8. Interview Scenario: Scaling for a Classifier]
# Apply scaling before training a classifier (e.g., SVM) to demonstrate necessity.

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X_train, X_test, y_train, y_test = train_test_split(X, np.random.randint(0, 2, 100), test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train)
y_pred = svm.predict(X_test_scaled)
print("\nSVM Accuracy with Scaling:", accuracy_score(y_test, y_pred).round(4))

# Try without scaling (for comparison)
svm.fit(X_train, y_train)
y_pred_no_scaling = svm.predict(X_test)
print("SVM Accuracy without Scaling:", accuracy_score(y_test, y_pred_no_scaling).round(4))