import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score, cross_validate
from sklearn.linear_model import LogisticRegression

# %% [1. Introduction to Cross-Validation]
# Cross-validation evaluates model performance by splitting data into multiple train-test folds,
# reducing variance compared to a single train-test split.

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Use the Iris dataset: 150 samples, 4 features, 3 classes.
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print("\nIris Dataset (first 5 rows):")
print(df.head())
print("\nDataset Shape:", X.shape)

# %% [3. KFold Cross-Validation]
# KFold splits data into k folds, using each as a test set once.

kf = KFold(n_splits=5, shuffle=True, random_state=42)
clf = LogisticRegression(random_state=42, max_iter=200)
scores = cross_val_score(clf, X, y, cv=kf, scoring='accuracy')
print("\nKFold Cross-Validation Scores:", scores.round(4))
print("Mean Accuracy:", scores.mean().round(4))
print("Std Accuracy:", scores.std().round(4))

# %% [4. StratifiedKFold Cross-Validation]
# StratifiedKFold ensures class distribution is preserved in each fold.

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores_strat = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
print("\nStratifiedKFold Cross-Validation Scores:", scores_strat.round(4))
print("Mean Accuracy:", scores_strat.mean().round(4))
print("Std Accuracy:", scores_strat.std().round(4))

# %% [5. Cross-Validate with Multiple Metrics]
# cross_validate allows evaluation with multiple metrics.

scoring = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}
cv_results = cross_validate(clf, X, y, cv=skf, scoring=scoring, return_train_score=True)
print("\nCross-Validate Results:")
print("Test Accuracy:", cv_results['test_accuracy'].round(4))
print("Test F1 Macro:", cv_results['test_f1_macro'].round(4))
print("Mean Test Accuracy:", cv_results['test_accuracy'].mean().round(4))

# %% [6. Practical Application: Comparing Models]
# Use cross-validation to compare two models (LogisticRegression vs. RandomForest).

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
lr_scores = cross_val_score(clf, X, y, cv=skf, scoring='accuracy')
rf_scores = cross_val_score(rf, X, y, cv=skf, scoring='accuracy')
print("\nModel Comparison:")
print("LogisticRegression Mean Accuracy:", lr_scores.mean().round(4))
print("RandomForest Mean Accuracy:", rf_scores.mean().round(4))

# %% [7. Cross-Validation with Imbalanced Data]
# Simulate imbalanced data and use StratifiedKFold.

mask = y != 2
X_imb = np.vstack([X[mask], X[y == 2][:10]])
y_imb = np.hstack([y[mask], y[y == 2][:10]])
scores_imb = cross_val_score(clf, X_imb, y_imb, cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42), scoring='f1_macro')
print("\nCross-Validation on Imbalanced Data (F1 Macro):")
print("Scores:", scores_imb.round(4))
print("Mean F1 Macro:", scores_imb.mean().round(4))

# %% [8. Interview Scenario: Custom Cross-Validation]
# Implement manual KFold cross-validation to demonstrate understanding.

accuracies = []
kf = KFold(n_splits=5, shuffle=True, random_state=42)
for train_idx, test_idx in kf.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
print("\nManual KFold Cross-Validation Accuracies:", [round(acc, 4) for acc in accuracies])
print("Mean Accuracy:", np.mean(accuracies).round(4))