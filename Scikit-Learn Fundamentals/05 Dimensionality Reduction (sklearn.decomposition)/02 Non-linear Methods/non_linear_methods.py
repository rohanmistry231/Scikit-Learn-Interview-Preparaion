import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import KernelPCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# %% [1. Introduction to Non-linear Methods]
# Non-linear dimensionality reduction captures complex patterns not handled by linear methods.
# Scikit-learn provides KernelPCA for non-linear feature transformation.

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

# %% [3. Data Preprocessing]
# Standardize features for KernelPCA.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaled Data (first 5 rows):")
print(pd.DataFrame(X_scaled, columns=feature_names).head())

# %% [4. KernelPCA with RBF Kernel]
# KernelPCA applies a kernel (e.g., RBF) to project data into a lower-dimensional space.
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X_scaled)
print("\nKernelPCA (RBF) Transformed Data Shape:", X_kpca.shape)

# %% [5. KernelPCA with Polynomial Kernel]
# Try a polynomial kernel for comparison.
kpca_poly = KernelPCA(n_components=2, kernel='poly', degree=3)
X_kpca_poly = kpca_poly.fit_transform(X_scaled)
print("\nKernelPCA (Polynomial) Transformed Data Shape:", X_kpca_poly.shape)

# %% [6. Visualizing KernelPCA Results]
# Plot KernelPCA (RBF) transformed data with class labels.
plt.figure()
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_kpca[y == i, 0], X_kpca[y == i, 1], label=target_name)
plt.xlabel('Component 1')
plt.ylabel('Component 2')
plt.title('KernelPCA (RBF) on Iris Dataset')
plt.legend()
plt.savefig('kpca_rbf_plot.png')

# %% [7. Practical Application: Model Performance]
# Compare classifier performance with and without KernelPCA.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nAccuracy without KernelPCA:", accuracy_score(y_test, y_pred).round(4))

# With KernelPCA (RBF)
X_train_kpca = kpca.fit_transform(X_train)
X_test_kpca = kpca.transform(X_test)
clf.fit(X_train_kpca, y_train)
y_pred_kpca = clf.predict(X_test_kpca)
print("Accuracy with KernelPCA (RBF):", accuracy_score(y_test, y_pred_kpca).round(4))

# %% [8. Tuning KernelPCA Parameters]
# Experiment with different gamma values for RBF kernel.
gammas = [0.01, 0.1, 1.0]
accuracies = []
for gamma in gammas:
    kpca = KernelPCA(n_components=2, kernel='rbf', gamma=gamma)
    X_train_kpca = kpca.fit_transform(X_train)
    X_test_kpca = kpca.transform(X_test)
    clf.fit(X_train_kpca, y_train)
    y_pred = clf.predict(X_test_kpca)
    accuracies.append(accuracy_score(y_test, y_pred))
print("\nKernelPCA (RBF) Accuracies for Different Gammas:", [round(acc, 4) for acc in accuracies])
print("Gammas:", gammas)

# %% [9. Interview Scenario: KernelPCA vs. PCA]
# Compare KernelPCA with PCA for non-linear data.
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
clf.fit(X_train_pca, y_train)
y_pred_pca = clf.predict(X_test_pca)
print("\nPCA vs. KernelPCA:")
print("PCA Accuracy:", accuracy_score(y_test, y_pred_pca).round(4))
print("KernelPCA (RBF) Accuracy:", accuracy_score(y_test, y_pred_kpca).round(4))
print("KernelPCA captures non-linear patterns, PCA assumes linearity.")

# %% [10. Pipeline with KernelPCA]
# Build a pipeline with KernelPCA and a classifier.
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('kpca', KernelPCA(n_components=2, kernel='rbf', gamma=0.1)),
    ('clf', RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred_pipe = pipeline.predict(X_test)
print("\nPipeline Accuracy with KernelPCA:", accuracy_score(y_test, y_pred_pipe).round(4))