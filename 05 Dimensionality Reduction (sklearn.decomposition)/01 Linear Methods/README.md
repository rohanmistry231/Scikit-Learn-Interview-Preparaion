# Linear Methods (`sklearn.decomposition`)

## 📖 Introduction
Linear dimensionality reduction methods project data onto a lower-dimensional space while preserving variance. This guide covers `PCA` (Principal Component Analysis) and `TruncatedSVD`, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand the principles of linear dimensionality reduction.
- Master `PCA` and `TruncatedSVD` for reducing dimensions.
- Apply linear methods to improve model performance and visualization.
- Build pipelines with linear dimensionality reduction.

## 🔑 Key Concepts
- **PCA**: Projects data onto principal components that maximize variance, requires centered data.
- **TruncatedSVD**: Similar to PCA but designed for sparse matrices, doesn’t center data.
- **Explained Variance**: Measures the proportion of variance retained by components.
- **Preprocessing**: Standardization is critical for PCA to ensure equal feature contributions.

## 📝 Example Walkthrough
The `linear_methods.py` file demonstrates:
1. **Dataset**: Iris dataset.
2. **Dimensionality Reduction**: Applying `PCA` and `TruncatedSVD` to reduce to 2 dimensions.
3. **Visualization**: Plotting PCA-transformed data.
4. **Pipeline**: Combining PCA with a classifier.

Example code:
```python
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
```

## 🛠️ Practical Tasks
1. Apply `PCA` to the Iris dataset and visualize the 2D projection.
2. Use `TruncatedSVD` on a standardized dataset and compare explained variance with PCA.
3. Train a classifier with and without PCA and compare accuracy.
4. Plot cumulative explained variance to choose the number of PCA components.

## 💡 Interview Tips
- **Common Questions**:
  - What is the difference between PCA and TruncatedSVD?
  - Why is standardization necessary for PCA?
  - How do you choose the number of PCA components?
- **Tips**:
  - Explain PCA’s maximization of variance and orthogonality of components.
  - Highlight TruncatedSVD’s efficiency for sparse data.
  - Be ready to code a PCA pipeline and interpret explained variance.

## 📚 Resources
- [Scikit-learn Decomposition Documentation](https://scikit-learn.org/stable/modules/decomposition.html)
- [Kaggle: Dimensionality Reduction Tutorial](https://www.kaggle.com/learn/feature-engineering)