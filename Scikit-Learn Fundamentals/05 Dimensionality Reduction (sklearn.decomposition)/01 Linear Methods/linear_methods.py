import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# %% [1. Introduction to Linear Methods]
# Linear dimensionality reduction methods project data onto a lower-dimensional space.
# Scikit-learn provides PCA (Principal Component Analysis) and TruncatedSVD.

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
# Standardize features for PCA and TruncatedSVD (zero mean, unit variance).
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaled Data (first 5 rows):")
print(pd.DataFrame(X_scaled, columns=feature_names).head())

# %% [4. PCA]
# PCA projects data onto principal components that maximize variance.
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
print("\nPCA Transformed Data Shape:", X_pca.shape)
print("Explained Variance Ratio:", pca.explained_variance_ratio_.round(4))
print("Total Explained Variance:", sum(pca.explained_variance_ratio_).round(4))

# %% [5. TruncatedSVD]
# TruncatedSVD is similar to PCA but works on sparse matrices and doesn't center data.
svd = TruncatedSVD(n_components=2)
X_svd = svd.fit_transform(X_scaled)
print("\nTruncatedSVD Transformed Data Shape:", X_svd.shape)
print("Explained Variance Ratio:", svd.explained_variance_ratio_.round(4))
print("Total Explained Variance:", sum(svd.explained_variance_ratio_).round(4))

# %% [6. Visualizing PCA Results]
# Plot PCA-transformed data with class labels.
plt.figure()
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset')
plt.legend()
plt.savefig('pca_plot.png')

# %% [7. Practical Application: Model Performance]
# Compare classifier performance with and without PCA.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nAccuracy without PCA:", accuracy_score(y_test, y_pred).round(4))

# With PCA
X_train_pca = pca.fit_transform(X_train)
X_test_pca = pca.transform(X_test)
clf.fit(X_train_pca, y_train)
y_pred_pca = clf.predict(X_test_pca)
print("Accuracy with PCA (2 components):", accuracy_score(y_test, y_pred_pca).round(4))

# %% [8. Choosing Number of Components]
# Evaluate explained variance for different numbers of components.
pca_full = PCA()
pca_full.fit(X_scaled)
cumulative_variance = np.cumsum(pca_full.explained_variance_ratio_)
print("\nCumulative Explained Variance:", cumulative_variance.round(4))
plt.figure()
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('PCA Explained Variance')
plt.savefig('pca_variance.png')

# %% [9. Interview Scenario: PCA vs. TruncatedSVD]
# Compare PCA and TruncatedSVD results.
print("\nPCA vs. TruncatedSVD:")
print("PCA centers data and maximizes variance.")
print("TruncatedSVD is efficient for sparse data but doesn't center.")
print("PCA Explained Variance:", pca.explained_variance_ratio_.round(4))
print("SVD Explained Variance:", svd.explained_variance_ratio_.round(4))

# %% [10. Pipeline with PCA]
# Build a pipeline with PCA and a classifier.
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('pca', PCA(n_components=2)),
    ('clf', RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred_pipe = pipeline.predict(X_test)
print("\nPipeline Accuracy with PCA:", accuracy_score(y_test, y_pred_pipe).round(4))