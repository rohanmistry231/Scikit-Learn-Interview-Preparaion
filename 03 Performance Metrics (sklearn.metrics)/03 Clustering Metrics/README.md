# Clustering Metrics (`sklearn.metrics`)

## 📖 Introduction
Clustering metrics evaluate the quality of unsupervised clustering algorithms. This guide covers `silhouette_score`, `adjusted_rand_score`, and `davies_bouldin_score`.

## 🎯 Learning Objectives
- Understand the role of clustering metrics.
- Master silhouette, ARI, and Davies-Bouldin scores.
- Evaluate clustering performance with and without ground truth.
- Analyze the impact of noise on clustering metrics.

## 🔑 Key Concepts
- **Silhouette Score**: Measures cohesion and separation (-1 to 1).
- **Adjusted Rand Index (ARI)**: Measures similarity to true labels, adjusted for chance.
- **Davies-Bouldin Score**: Evaluates cluster compactness and separation (lower is better).
- **Internal vs. External Metrics**: Silhouette and Davies-Bouldin are internal; ARI requires labels.

## 📝 Example Walkthrough
The `clustering_metrics.py` file demonstrates:
1. **Dataset**: Synthetic dataset with 3 clusters.
2. **Clustering**: Applying KMeans with k=3.
3. **Metrics**: Computing silhouette, ARI, and Davies-Bouldin scores.
4. **Noise**: Evaluating metrics on noisy data.

Example code:
```python
from sklearn.metrics import silhouette_score
sil_score = silhouette_score(X, y_pred)
```

## 🛠️ Practical Tasks
1. Apply KMeans to a synthetic dataset and compute silhouette score.
2. Compute ARI using true labels and compare with random clustering.
3. Evaluate Davies-Bouldin score for different k values in KMeans.
4. Add noise to a dataset and analyze its impact on clustering metrics.

## 💡 Interview Tips
- **Common Questions**:
  - What does a high silhouette score indicate?
  - When is ARI preferred over silhouette score?
  - How does Davies-Bouldin score evaluate clustering quality?
- **Tips**:
  - Explain silhouette’s range and interpretation.
  - Highlight ARI’s need for ground truth labels.
  - Be ready to code a clustering evaluation with multiple metrics.

## 📚 Resources
- [Scikit-learn Clustering Metrics Documentation](https://scikit-learn.org/stable/modules/clustering.html#clustering-evaluation)
- [Kaggle: Clustering Tutorial](https://www.kaggle.com/learn/machine-learning-with-scikit-learn)