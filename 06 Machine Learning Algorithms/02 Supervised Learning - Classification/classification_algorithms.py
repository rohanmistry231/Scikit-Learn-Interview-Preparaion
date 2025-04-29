import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# %% [1. Introduction to Classification Algorithms]
# Classification predicts discrete class labels using supervised learning.
# Scikit-learn provides LogisticRegression, SVC, RandomForestClassifier, GradientBoostingClassifier, and KNeighborsClassifier.

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

# %% [3. Train-Test Split]
# Split data into training (80%) and testing (20%) sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nTrain Shape:", X_train.shape, "Test Shape:", X_test.shape)

# %% [4. LogisticRegression]
# LogisticRegression predicts class probabilities using a logistic function.
lr = LogisticRegression(max_iter=200, random_state=42)
lr.fit(X_train, y_train)
y_pred_lr = lr.predict(X_test)
print("\nLogisticRegression:")
print("Accuracy:", accuracy_score(y_test, y_pred_lr).round(4))
print("F1 Score (Macro):", f1_score(y_test, y_pred_lr, average='macro').round(4))

# %% [5. SVC]
# SVC uses support vector machines for classification.
svc = SVC(kernel='rbf', random_state=42)
svc.fit(X_train, y_train)
y_pred_svc = svc.predict(X_test)
print("\nSVC:")
print("Accuracy:", accuracy_score(y_test, y_pred_svc).round(4))
print("F1 Score (Macro):", f1_score(y_test, y_pred_svc, average='macro').round(4))

# %% [6. RandomForestClassifier]
# RandomForestClassifier builds multiple decision trees for robust classification.
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandomForestClassifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf).round(4))
print("F1 Score (Macro):", f1_score(y_test, y_pred_rf, average='macro').round(4))

# %% [7. GradientBoostingClassifier]
# GradientBoostingClassifier builds trees sequentially to correct errors.
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print("\nGradientBoostingClassifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_gb).round(4))
print("F1 Score (Macro):", f1_score(y_test, y_pred_gb, average='macro').round(4))

# %% [8. KNeighborsClassifier]
# KNeighborsClassifier predicts based on nearest neighbors.
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("\nKNeighborsClassifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_knn).round(4))
print("F1 Score (Macro):", f1_score(y_test, y_pred_knn, average='macro').round(4))

# %% [9. Practical Application: Comparing Models]
# Compare all models on test set performance.
models = {
    'LogisticRegression': y_pred_lr,
    'SVC': y_pred_svc,
    'RandomForest': y_pred_rf,
    'GradientBoosting': y_pred_gb,
    'KNeighbors': y_pred_knn
}
print("\nModel Comparison:")
for name, y_pred in models.items():
    print(f"{name}: Accuracy={accuracy_score(y_test, y_pred).round(4)}, F1={f1_score(y_test, y_pred, average='macro').round(4)}")

# %% [10. Interview Scenario: Model Selection]
# Discuss model choice for small datasets.
print("\nModel Selection for Small Datasets:")
print("LogisticRegression: Simple, interpretable, good for linear boundaries.")
print("RandomForest: Robust, handles non-linearity, less sensitive to preprocessing.")
print("KNeighbors: Non-parametric, sensitive to distance metrics.")

# Plot feature importances for RandomForest
plt.figure()
plt.bar(feature_names, rf.feature_importances_)
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('RandomForest Feature Importances')
plt.tight_layout()
plt.savefig('rf_importances.png')