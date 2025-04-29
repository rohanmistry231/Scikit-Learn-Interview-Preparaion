import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import Lasso

# %% [1. Introduction to Embedded Methods]
# Embedded methods integrate feature selection into the model training process.
# Scikit-learn provides SelectFromModel and feature importance from models like RandomForest.

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

# %% [3. SelectFromModel with RandomForest]
# SelectFromModel selects features based on RandomForest feature importance.

rf = RandomForestClassifier(random_state=42)
selector = SelectFromModel(estimator=rf, threshold='mean')
X_rf = selector.fit_transform(X, y)
selected_features = np.array(feature_names)[selector.get_support()]
print("\nSelectFromModel (RandomForest) Selected Features:", selected_features)
print("Feature Importances:", selector.estimator_.feature_importances_.round(4))
print("Shape after Selection:", X_rf.shape)

# %% [4. SelectFromModel with Lasso]
# Use Lasso (L1 regularization) to select features by shrinking coefficients to zero.

lasso = Lasso(alpha=0.1, random_state=42)
selector_lasso = SelectFromModel(estimator=lasso)
X_lasso = selector_lasso.fit_transform(X, y)
selected_features = np.array(feature_names)[selector_lasso.get_support()]
print("\nSelectFromModel (Lasso) Selected Features:", selected_features)
print("Lasso Coefficients:", selector_lasso.estimator_.coef_.round(4))

# %% [5. Practical Application: Model Performance]
# Compare model performance with and without feature selection.

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nAccuracy without Feature Selection:", accuracy_score(y_test, y_pred).round(4))

# With SelectFromModel (RandomForest)
selector = SelectFromModel(estimator=RandomForestClassifier(random_state=42))
X_train_rf = selector.fit_transform(X_train, y_train)
X_test_rf = selector.transform(X_test)
clf.fit(X_train_rf, y_train)
y_pred_rf = clf.predict(X_test_rf)
print("Accuracy with SelectFromModel (RandomForest):", accuracy_score(y_test, y_pred_rf).round(4))

# %% [6. Tuning Threshold in SelectFromModel]
# Experiment with different thresholds for feature selection.

selector = SelectFromModel(estimator=RandomForestClassifier(random_state=42), threshold='1.5*mean')
X_rf_tuned = selector.fit_transform(X, y)
selected_features = np.array(feature_names)[selector.get_support()]
print("\nSelectFromModel (RandomForest, 1.5*mean) Selected Features:", selected_features)
print("Shape after Tuned Selection:", X_rf_tuned.shape)

# %% [7. Visualizing Feature Importances]
# Plot feature importances from RandomForest.

import matplotlib.pyplot plt
rf.fit(X, y)
importances = rf.feature_importances_
plt.figure()
plt.bar(feature_names, importances)
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('RandomForest Feature Importances')
plt.tight_layout()
plt.savefig('feature_importances.png')

# %% [8. Interview Scenario: Embedded vs. Wrapper Methods]
# Compare SelectFromModel with RFE for model performance.

from sklearn.feature_selection import RFE
rfe = RFE(estimator=LogisticRegression(max_iter=200), n_features_to_select=2)
X_train_rfe = rfe.fit_transform(X_train, y_train)
X_test_rfe = rfe.transform(X_test)
clf.fit(X_train_rfe, y_train)
y_pred_rfe = clf.predict(X_test_rfe)
print("\nAccuracy with RFE:", accuracy_score(y_test, y_pred_rfe).round(4))
print("Embedded vs. RFE: Embedded integrates selection during training, RFE iteratively evaluates.")

# %% [9. Handling High-Dimensional Data]
# Simulate high-dimensional data and apply SelectFromModel.

X_high_dim = np.hstack([X, np.random.normal(0, 1, (X.shape[0], 10))])
feature_names_high = feature_names + [f'noise_{i}' for i in range(10)]
selector = SelectFromModel(estimator=RandomForestClassifier(random_state=42))
X_high_selected = selector.fit_transform(X_high_dim, y)
selected_features = np.array(feature_names_high)[selector.get_support()]
print("\nSelectFromModel on High-Dimensional Data Selected Features:", selected_features)

# %% [10. Pipeline with Embedded Methods]
# Build a pipeline with SelectFromModel and a classifier.

from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('select', SelectFromModel(estimator=RandomForestClassifier(random_state=42))),
    ('clf', RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred_pipe = pipeline.predict(X_test)
print("\nPipeline Accuracy with SelectFromModel:", accuracy_score(y_test, y_pred_pipe).round(4))