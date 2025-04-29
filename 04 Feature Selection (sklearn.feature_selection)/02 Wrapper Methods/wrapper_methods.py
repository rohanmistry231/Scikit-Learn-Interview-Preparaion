import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import RFE, RFECV
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

# %% [1. Introduction to Wrapper Methods]
# Wrapper methods select features by evaluating subsets using a specific model.
# Scikit-learn provides RFE (Recursive Feature Elimination) and RFECV.

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Use Iris dataset: 150 samples, 4 features, 3 classes.
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
df = pd.DataFrame(X, columns=feature_names)
print("\nIris Dataset (first 5 rows):")
print(df.head())
print("\nDataset Shape:", X.shape)

# %% [3. RFE]
# RFE recursively eliminates features based on model importance until a specified number remain.

clf = LogisticRegression(random_state=42, max_iter=200)
rfe = RFE(estimator=clf, n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)
selected_features = np.array(feature_names)[rfe.support_]
print("\nRFE Selected Features:", selected_features)
print("Feature Ranking:", rfe.ranking_)
print("Shape after RFE:", X_rfe.shape)

# %% [4. RFECV]
# RFECV uses cross-validation to automatically select the optimal number of features.

rfecv = RFECV(estimator=clf, step=1, cv=5, scoring='accuracy')
X_rfecv = rfecv.fit_transform(X, y)
selected_features = np.array(feature_names)[rfecv.support_]
print("\nRFECV Selected Features:", selected_features)
print("Optimal Number of Features:", rfecv.n_features_)
print("Shape after RFECV:", X_rfecv.shape)

# %% [5. Practical Application: Model Performance]
# Compare model performance with and without RFE.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nAccuracy without Feature Selection:", accuracy_score(y_test, y_pred).round(4))

# With RFE
rfe = RFE(estimator=LogisticRegression(max_iter=200), n_features_to_select=2)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)
clf.fit(X_train_rfe, y_train)
y_pred_rfe = clf.predict(X_test_rfe)
print("Accuracy with RFE:", accuracy_score(y_test, y_pred_rfe).round(4))

# %% [6. RFECV with Different Models]
# Use RFECV with a different estimator (RandomForest).

rfecv_rf = RFECV(estimator=RandomForestClassifier(random_state=42), step=1, cv=5, scoring='accuracy')
X_rfecv_rf = rfecv_rf.fit_transform(X, y)
selected_features = np.array(feature_names)[rfecv_rf.support_]
print("\nRFECV (RandomForest) Selected Features:", selected_features)
print("Optimal Number of Features:", rfecv_rf.n_features_)

# %% [7. Visualizing RFECV Results]
# Plot cross-validation scores vs. number of features.

import matplotlib.pyplot as plt
plt.figure()
plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
plt.xlabel('Number of Features')
plt.ylabel('Cross-Validation Accuracy')
plt.title('RFECV Feature Selection')
plt.savefig('rfecv_scores.png')

# %% [8. Interview Scenario: RFE vs. Filter Methods]
# Compare RFE with filter methods (e.g., SelectKBest) for model performance.

from sklearn.feature_selection import SelectKBest, f_classif
kbest = SelectKBest(score_func=f_classif, k=2)
X_train_kbest = kbest.fit_transform(X_train, y_train)
X_test_kbest = kbest.transform(X_test)
clf.fit(X_train_kbest, y_train)
y_pred_kbest = clf.predict(X_test_kbest)
print("\nAccuracy with SelectKBest:", accuracy_score(y_test, y_pred_kbest).round(4))
print("RFE vs. SelectKBest: RFE considers model performance, SelectKBest uses statistical tests.")

# %% [9. Handling High-Dimensional Data]
# Simulate high-dimensional data and apply RFE.

X_high_dim = np.hstack([X, np.random.normal(0, 1, (X.shape[0], 10))])
feature_names_high = feature_names + [f'noise_{i}' for i in range(10)]
rfe_high = RFE(estimator=LogisticRegression(max_iter=200), n_features_to_select=4)
X_rfe_high = rfe_high.fit_transform(X_high_dim, y)
selected_features = np.array(feature_names_high)[rfe_high.support_]
print("\nRFE on High-Dimensional Data Selected Features:", selected_features)

# %% [10. Pipeline with RFE]
# Build a pipeline with RFE and a classifier.

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('rfe', RFE(estimator=LogisticRegression(max_iter=200), n_features_to_select=2)),
    ('clf', RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred_pipe = pipeline.predict(X_test)
print("\nPipeline Accuracy with RFE:", accuracy_score(y_test, y_pred_pipe).round(4))