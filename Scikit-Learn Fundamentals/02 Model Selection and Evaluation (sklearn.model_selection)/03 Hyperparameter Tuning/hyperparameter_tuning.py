import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.svm import SVC
from scipy.stats import uniform, randint

# %% [1. Introduction to Hyperparameter Tuning]
# Hyperparameter tuning optimizes model performance by selecting the best parameter values.
# Scikit-learn provides GridSearchCV and RandomizedSearchCV for this purpose.

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

# %% [3. GridSearchCV]
# GridSearchCV exhaustively searches over a specified parameter grid.

param_grid = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
grid = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='accuracy')
grid.fit(X, y)
print("\nGridSearchCV Results:")
print("Best Parameters:", grid.best_params_)
print("Best Score:", grid.best_score_.round(4))
print("Best Estimator:", grid.best_estimator_)

# %% [4. RandomizedSearchCV]
# RandomizedSearchCV samples a fixed number of parameter combinations.

param_dist = {
    'C': uniform(0.1, 10),
    'kernel': ['linear', 'rbf'],
    'gamma': ['scale', 'auto']
}
random = RandomizedSearchCV(SVC(random_state=42), param_dist, n_iter=10, cv=5, scoring='accuracy', random_state=42)
random.fit(X, y)
print("\nRandomizedSearchCV Results:")
print("Best Parameters:", random.best_params_)
print("Best Score:", random.best_score_.round(4))

# %% [5. Custom Scoring Metrics]
# Use a custom metric (e.g., F1 score) for hyperparameter tuning.

grid_f1 = GridSearchCV(SVC(random_state=42), param_grid, cv=5, scoring='f1_macro')
grid_f1.fit(X, y)
print("\nGridSearchCV with F1 Macro:")
print("Best Parameters:", grid_f1.best_params_)
print("Best F1 Score:", grid_f1.best_score_.round(4))

# %% [6. Practical Application: Tuning for Imbalanced Data]
# Tune parameters on an imbalanced dataset.

mask = y != 2
X_imb = np.vstack([X[mask], X[y == 2][:10]])
y_imb = np.hstack([y[mask], y[y == 2][:10]])
grid_imb = GridSearchCV(SVC(random_state=42), param_grid, cv=StratifiedKFold(n_splits=5), scoring='f1_macro')
grid_imb.fit(X_imb, y_imb)
print("\nGridSearchCV on Imbalanced Data:")
print("Best Parameters:", grid_imb.best_params_)
print("Best F1 Score:", grid_imb.best_score_.round(4))

# %% [7. Nested Cross-Validation]
# Use nested CV to avoid overfitting during hyperparameter tuning.

from sklearn.model_selection import cross_val_score
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_nested = GridSearchCV(SVC(random_state=42), param_grid, cv=inner_cv, scoring='accuracy')
scores_nested = cross_val_score(grid_nested, X, y, cv=outer_cv)
print("\nNested Cross-Validation Scores:", scores_nested.round(4))
print("Mean Accuracy:", scores_nested.mean().round(4))

# %% [8. Interview Scenario: Comparing Grid and Random Search]
# Compare GridSearchCV and RandomizedSearchCV for computational efficiency.

import time
start = time.time()
grid.fit(X, y)
grid_time = time.time() - start
start = time.time()
random.fit(X, y)
random_time = time.time() - start
print("\nSearch Time Comparison:")
print("GridSearchCV Time:", grid_time.round(4), "seconds")
print("RandomizedSearchCV Time:", random_time.round(4), "seconds")
print("GridSearchCV Best Score:", grid.best_score_.round(4))
print("RandomizedSearchCV Best Score:", random.best_score_.round(4))