import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# %% [1. Introduction to Data Splitting]
# Data splitting divides a dataset into training and testing sets to evaluate model performance.
# Scikit-learn's train_test_split is the primary tool for this purpose.

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Use the Iris dataset for demonstration: 150 samples, 4 features, 3 classes.
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

print("\nIris Dataset (first 5 rows):")
print(df.head())
print("\nDataset Shape:", X.shape)

# %% [3. Basic Train-Test Split]
# Split dataset into training (80%) and testing (20%) sets.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("\nBasic Train-Test Split:")
print("Train Shape:", X_train.shape, "Test Shape:", X_test.shape)
print("Train Target Distribution:", np.bincount(y_train))
print("Test Target Distribution:", np.bincount(y_test))

# %% [4. Stratified Train-Test Split]
# Stratified split ensures class distribution is preserved in train and test sets.

X_train_strat, X_test_strat, y_train_strat, y_test_strat = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("\nStratified Train-Test Split:")
print("Train Shape:", X_train_strat.shape, "Test Shape:", X_test_strat.shape)
print("Train Target Distribution:", np.bincount(y_train_strat))
print("Test Target Distribution:", np.bincount(y_test_strat))

# %% [5. Custom Split Ratios]
# Experiment with different test sizes (e.g., 30% test set).

X_train_30, X_test_30, y_train_30, y_test_30 = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
print("\nCustom Split (30% Test):")
print("Train Shape:", X_train_30.shape, "Test Shape:", X_test_30.shape)

# %% [6. Practical Application: Model Training]
# Train a classifier on the split data to evaluate performance.

clf = LogisticRegression(random_state=42, max_iter=200)
clf.fit(X_train_strat, y_train_strat)
y_pred = clf.predict(X_test_strat)
print("\nLogistic Regression Accuracy (Stratified Split):", accuracy_score(y_test_strat, y_pred).round(4))

# Try without stratification for comparison
clf.fit(X_train, y_train)
y_pred_no_strat = clf.predict(X_test)
print("Logistic Regression Accuracy (Non-Stratified Split):", accuracy_score(y_test, y_pred_no_strat).round(4))

# %% [7. Multiple Splits for Robustness]
# Perform multiple splits to assess model stability.

accuracies = []
for i in range(5):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=i, stratify=y)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
print("\nAccuracies from Multiple Splits:", [round(acc, 4) for acc in accuracies])
print("Mean Accuracy:", np.mean(accuracies).round(4))

# %% [8. Interview Scenario: Handling Imbalanced Data]
# Simulate imbalanced data and use stratified split to maintain class proportions.

# Create imbalanced dataset by reducing class 2 samples
mask = y != 2
X_imb = np.vstack([X[mask], X[y == 2][:10]])
y_imb = np.hstack([y[mask], y[y == 2][:10]])
X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=42, stratify=y_imb
)
print("\nImbalanced Dataset Split:")
print("Train Target Distribution:", np.bincount(y_train_imb))
print("Test Target Distribution:", np.bincount(y_test_imb))