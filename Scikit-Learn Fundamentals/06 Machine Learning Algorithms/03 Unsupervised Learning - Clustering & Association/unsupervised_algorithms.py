import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# %% [1. Introduction to Unsupervised Learning]
# Unsupervised learning finds patterns in unlabeled data.
# Scikit-learn provides clustering (KMeans, DBSCAN, etc.) and anomaly detection (IsolationForest, OneClassSVM).

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Generate synthetic dataset: 300 samples, 2 features, 3 clusters.
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
print("\nSynthetic Dataset (first 5 rows):")
print(df.head())
print("\nDataset Shape:", X.shape)

# %% [3. KMeans]
# KMeans partitions data into k clusters based on centroids.
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)
print("\nKMeans Silhouette Score:", silhouette_score(X, y_kmeans).round(4))

# %% [4. DBSCAN]
# DBSCAN clusters data based on density, identifying outliers.
dbscan = DBSCAN(eps=0.5, min_samples=5)
y_dbscan = dbscan.fit_predict(X)
print("\nDBSCAN Silhouette Score:", silhouette_score(X, y_dbscan, sample_size=1000).round(4) if -1 not in y_dbscan else "N/A (outliers detected)")

# %% [5. AgglomerativeClustering]
# AgglomerativeClustering builds a hierarchy of clusters.
agg = AgglomerativeClustering(n_clusters=3)
y_agg = agg.fit_predict(X)
print("\nAgglomerativeClustering Silhouette Score:", silhouette_score(X, y_agg).round(4))

# %% [6. GaussianMixture]
# GaussianMixture models data as a mixture of Gaussian distributions.
gmm = GaussianMixture(n_components=3, random_state=42)
y_gmm = gmm.fit_predict(X)
print("\nGaussianMixture Silhouette Score:", silhouette_score(X, y_gmm).round(4))

# %% [7. IsolationForest]
# IsolationForest detects anomalies by isolating points.
iso = IsolationForest(contamination=0.1, random_state=42)
y_iso = iso.fit_predict(X)
print("\nIsolationForest Anomalies Detected:", sum(y_iso == -1))

# %% [8. OneClassSVM]
# OneClassSVM detects anomalies using a single-class SVM.
ocsvm = OneClassSVM(kernel='rbf', nu=0.1)
y_ocsvm = ocsvm.fit_predict(X)
print("\nOneClassSVM Anomalies Detected:", sum(y_ocsvm == -1))

# %% [9. Practical Application: Visualizing Clusters]
# Plot KMeans clusters.
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_kmeans, cmap='viridis', label='Clusters')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans Clustering')
plt.legend()
plt.savefig('kmeans_plot.png')

# Plot IsolationForest anomalies
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_iso, cmap='coolwarm', label='Normal vs. Anomalies')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('IsolationForest Anomaly Detection')
plt.legend()
plt.savefig('isolationforest_plot.png')

# %% [10. Interview Scenario: Choosing Algorithms]
# Discuss clustering vs. anomaly detection.
print("\nChoosing Unsupervised Algorithms:")
print("KMeans: Assumes spherical clusters, needs k specified.")
print("DBSCAN: Handles arbitrary shapes, detects outliers.")
print("IsolationForest: Fast for anomaly detection, scalable.")