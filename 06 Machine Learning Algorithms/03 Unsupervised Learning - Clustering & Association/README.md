# Unsupervised Learning

## 📖 Introduction
Unsupervised learning finds patterns in unlabeled data through clustering and anomaly detection. This guide covers `KMeans`, `DBSCAN`, `AgglomerativeClustering`, `GaussianMixture`, `IsolationForest`, and `OneClassSVM`, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand unsupervised learning concepts.
- Master scikit-learn’s clustering and anomaly detection algorithms.
- Apply algorithms to synthetic data and evaluate performance.
- Visualize clustering and anomaly detection results.

## 🔑 Key Concepts
- **KMeans**: Partitions data into k clusters using centroids.
- **DBSCAN**: Density-based clustering, identifies outliers.
- **AgglomerativeClustering**: Hierarchical clustering.
- **GaussianMixture**: Probabilistic clustering with Gaussian distributions.
- **IsolationForest**: Anomaly detection via tree isolation.
- **OneClassSVM**: Anomaly detection using SVM.

## 📝 Example Walkthrough
The `unsupervised_algorithms.py` file demonstrates:
1. **Dataset**: Synthetic dataset (300 samples, 2 features).
2. **Clustering**: Applying `KMeans`, `DBSCAN`, `AgglomerativeClustering`, and `GaussianMixture`.
3. **Anomaly Detection**: Using `IsolationForest` and `OneClassSVM`.
4. **Visualization**: Plotting KMeans clusters and IsolationForest anomalies.

Example code:
```python
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=42)
y_kmeans = kmeans.fit_predict(X)
```

## 🛠️ Practical Tasks
1. Apply `KMeans` to a synthetic dataset and compute silhouette score.
2. Use `DBSCAN` to identify clusters and outliers.
3. Compare `GaussianMixture` with `KMeans` on clustering performance.
4. Detect anomalies with `IsolationForest` and visualize results.

## 💡 Interview Tips
- **Common Questions**:
  - How does DBSCAN differ from KMeans?
  - When is IsolationForest preferred for anomaly detection?
  - What are the assumptions of GaussianMixture?
- **Tips**:
  - Explain DBSCAN’s density-based approach.
  - Highlight IsolationForest’s scalability.
  - Be ready to code a KMeans clustering pipeline and discuss silhouette score.

## 📚 Resources
- [Scikit-learn Clustering Documentation](https://scikit-learn.org/stable/modules/clustering.html)
- [Scikit-learn Anomaly Detection Documentation](https://scikit-learn.org/stable/modules/outlier_detection.html)
- [Kaggle: Unsupervised Learning Tutorial](https://www.kaggle.com/learn/intro-to-machine-learning)