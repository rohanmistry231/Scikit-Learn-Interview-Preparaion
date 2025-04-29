import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, f_classif, mutual_info_classif
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# %% [1. Introduction to Filter Methods]
# Filter methods select features based on statistical properties, independent of the model.
# Scikit-learn provides VarianceThreshold, SelectKBest with chi2, f_classif, and mutual_info_classif.

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Use Iris dataset and add noisy/low-variance features: 150 samples, 4 original + 3 synthetic features.
iris = load_iris()
X = iris.data
y = iris.target
# Add synthetic features: low variance, constant, and noisy
np.random.seed(42)
X = np.hstack([X, np.random.normal(0, 0.01, (X.shape[0], 1)),  # Low variance
               np.ones((X.shape[0], 1)),                        # Constant
               np.random.normal(0, 1, (X.shape[0], 1))])        # Noisy
feature_names = iris.feature_names + ['low_var', 'constant', 'noisy']
df = pd.DataFrame(X, columns=feature_names)
print("\nDataset with Synthetic Features (first 5 rows):")
print(df.head())
print("\nDataset Shape:", X.shape)

# %% [3. VarianceThreshold]
# VarianceThreshold removes features with variance below a threshold.

var_thresh = VarianceThreshold(threshold=0.1)
X_var = var_thresh.fit_transform(X)
selected_features = feature_names[np.where(var_thresh.get_support())[0]]
print("\nVarianceThreshold Selected Features:", selected_features)
print("Shape after VarianceThreshold:", X_var.shape)

# %% [4. SelectKBest with chi2]
# SelectKBest with chi2 selects the top k features based on chi-squared test (non-negative data).

# Ensure non-negative data for chi2
X_nonneg = X - X.min(axis=0) + 1e-6
kbest_chi2 = SelectKBest(score_func=chi2, k=3)
X_chi2 = kbest_chi2.fit_transform(X_nonneg, y)
selected_features = feature_names[np.where(kbest_chi2.get_support())[0]]
print("\nSelectKBest (chi2) Selected Features:", selected_features)
print("Chi2 Scores:", kbest_chi2.scores_.round(4))

# %% [5. SelectKBest with f_classif]
# SelectKBest with f_classif uses ANOVA F-test for feature importance.

kbest_f = SelectKBest(score_func=f_classif, k=3)
X_f = kbest_f.fit_transform(X, y)
selected_features = feature_names[np.where(kbest_f.get_support())[0]]
print("\nSelectKBest (f_classif) Selected Features:", selected_features)
print("F-classif Scores:", kbest_f.scores_.round(4))

# %% [6. SelectKBest with mutual_info_classif]
# SelectKBest with mutual_info_classif measures mutual information between features and target.

kbest_mi = SelectKBest(score_func=mutual_info_classif, k=3)
X_mi = kbest_mi.fit_transform(X, y)
selected_features = feature_names[np.where(kbest_mi.get_support())[0]]
print("\nSelectKBest (mutual_info_classif) Selected Features:", selected_features)
print("Mutual Info Scores:", kbest_mi.scores_.round(4))

# %% [7. Practical Application: Model Performance]
# Compare model performance with and without feature selection.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nAccuracy without Feature Selection:", accuracy_score(y_test, y_pred).round(4))

# With SelectKBest (f_classif)
X_train_f = kbest_f.fit_transform(X_train, y_train)
X_test_f = kbest_f.transform(X_test)
clf.fit(X_train_f, y_train)
y_pred_f = clf.predict(X_test_f)
print("Accuracy with SelectKBest (f_classif):", accuracy_score(y_test, y_pred_f).round(4))

# %% [8. Combining Filter Methods]
# Combine VarianceThreshold and SelectKBest for robust feature selection.

var_thresh = VarianceThreshold(threshold=0.1)
X_var = var_thresh.fit_transform(X)
kbest_f = SelectKBest(score_func=f_classif, k=3)
X_combined = kbest_f.fit_transform(X_var, y)
selected_features = feature_names[np.where(var_thresh.get_support())[0]][
    np.where(kbest_f.get_support())[0]]
print("\nCombined VarianceThreshold + SelectKBest Features:", selected_features)

# %% [9. Interview Scenario: Feature Selection Pipeline]
# Build a pipeline to apply feature selection and train a model.

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('var_thresh', VarianceThreshold(threshold=0.1)),
    ('select_kbest', SelectKBest(score_func=f_classif, k=3)),
    ('clf', RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred_pipe = pipeline.predict(X_test)
print("\nPipeline Accuracy:", accuracy_score(y_test, y_pred_pipe).round(4))

# %% [10. Visualizing Feature Scores]
# Plot feature scores from SelectKBest (f_classif).

import matplotlib.pyplot as plt
plt.figure()
plt.bar(feature_names, kbest_f.scores_)
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('F-classif Score')
plt.title('Feature Importance (f_classif)')
plt.tight_layout()
plt.savefig('feature_scores.png')