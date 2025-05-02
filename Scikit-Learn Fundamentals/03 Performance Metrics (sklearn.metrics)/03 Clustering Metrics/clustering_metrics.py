import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, adjusted_rand_score, davies_bouldin_score

# %% [1. Introduction to Clustering Metrics]
# Clustering metrics evaluate the quality of unsupervised clustering.
# Scikit-learn provides silhouette_score, adjusted_rand_score, and davies_bouldin_score.

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Generate synthetic dataset: 200 samples, 3 clusters, 2 features.
X, y_true = make_blobs(n_samples=200, centers=3, cluster_std=1.0, random_state=42)
df = pd.DataFrame(X, columns=['feature1', 'feature2'])
df['true_labels'] = y_true

print("\nSynthetic Dataset (first 5 rows):")
print(df.head())
print("\nDataset Shape:", X.shape)

# %% [3. KMeans Clustering]
# Apply KMeans clustering to generate predicted labels.
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred = kmeans.fit_predict(X)
print("\nPredicted Cluster Labels (first 5):", y_pred[:5])

# %% [4. Silhouette Score]
# Silhouette score measures how similar an object is to its own cluster vs. other clusters (range: -1 to 1).

sil_score = silhouette_score(X, y_pred)
print("\nSilhouette Score:", sil_score.round(4))

# %% [5. Adjusted Rand Index]
# Adjusted Rand Index measures similarity between true and predicted labels, adjusted for chance.

ari = adjusted_rand_score(y_true, y_pred)
print("\nAdjusted Rand Index:", ari.round(4))

# %% [6. Davies-Bouldin Score]
# Davies-Bouldin score measures cluster separation and compactness (lower is better).

db_score = davies_bouldin_score(X, y_pred)
print("\nDavies-Bouldin Score:", db_score.round(4))

# %% [7. Practical Application: Comparing Cluster Numbers]
# Evaluate KMeans with different numbers of clusters.

scores = {'silhouette': [], 'ari': [], 'db': []}
for k in range(2, 5):
    kmeans = KMeans(n_clusters=k, random_state=42)
    y_pred = kmeans.fit_predict(X)
    scores['silhouette'].append(silhouette_score(X, y_pred))
    scores['ari'].append(adjusted_rand_score(y_true, y_pred))
    scores['db'].append(davies_bouldin_score(X, y_pred))
print("\nClustering Metrics for k=2 to 4:")
print("Silhouette Scores:", [round(s, 4) for s in scores['silhouette']])
print("ARI Scores:", [round(s, 4) for s in scores['ari']])
print("DB Scores:", [round(s, 4) for s in scores['db']])

# %% [8. Visualizing Clusters]
# Plot clusters to visualize clustering quality.

import matplotlib.pyplot as plt
plt.figure()
plt.scatter(X[:, 0], X[:, 1], c=y_pred, cmap='viridis', label='Predicted Clusters')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], c='red', marker='x', s=200, label='Centroids')
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('KMeans Clustering (k=3)')
plt.legend()
plt.savefig('clustering_plot.png')

# %% [9. Interview Scenario: Choosing Metrics]
# Evaluate clustering with different metrics to justify choice.

print("\nMetric Choice for Clustering:")
print("Silhouette: Measures cohesion and separation, good for internal evaluation.")
print("ARI: Requires true labels, measures agreement with ground truth.")
print("Davies-Bouldin: Lower values indicate better clustering, no labels needed.")

# %% [10. Handling Noisy Data]
# Add noise to the dataset and evaluate metrics.

X_noisy = np.vstack([X, np.random.uniform(X.min(), X.max(), (20, 2))])
y_true_noisy = np.hstack([y_true, [-1] * 20])  # -1 for noise
kmeans = KMeans(n_clusters=3, random_state=42)
y_pred_noisy = kmeans.fit_predict(X_noisy)
print("\nMetrics with Noisy Data:")
print("Silhouette Score:", silhouette_score(X_noisy, y_pred_noisy).round(4))
print("Adjusted Rand Index:", adjusted_rand_score(y_true_noisy, y_pred_noisy).round(4))
print("Davies-Bouldin Score:", davies_bouldin_score(X_noisy, y_pred_noisy).round(4))