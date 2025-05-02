# Visualization Aids (`sklearn.manifold`)

## 📖 Introduction
Visualization aids reduce data to 2D or 3D for visual exploration of high-dimensional data. This guide covers `TSNE` (via `sklearn.manifold`) and `UMAP` (via `umap-learn`), with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand the role of visualization aids in data exploration.
- Master `TSNE` and `UMAP` for 2D visualization.
- Tune visualization parameters (e.g., perplexity).
- Compare visualization aids with other dimensionality reduction methods.

## 🔑 Key Concepts
- **TSNE**: Preserves local structure for non-linear 2D/3D visualization, computationally intensive.
- **UMAP**: Faster, preserves local and global structure, requires `umap-learn`.
- **Perplexity**: TSNE parameter controlling neighbor balance (typically 5-50).
- **Visualization Focus**: Primarily for exploration, not model input.

## 📝 Example Walkthrough
The `visualization_aids.py` file demonstrates:
1. **Dataset**: Iris dataset and synthetic high-dimensional data.
2. **Visualization**: Applying `TSNE` and `UMAP` (if installed) for 2D plots.
3. **Parameter Tuning**: Experimenting with TSNE perplexity.
4. **Comparison**: Visualizing PCA for contrast.

Example code:
```python
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=42)
X_tsne = tsne.fit_transform(X)
```

## 🛠️ Practical Tasks
1. Apply `TSNE` to the Iris dataset and visualize the 2D projection.
2. Install `umap-learn` and compare UMAP with TSNE on the same dataset.
3. Tune TSNE’s perplexity and observe changes in visualization.
4. Apply TSNE to a high-dimensional synthetic dataset and interpret the plot.

## 💡 Interview Tips
- **Common Questions**:
  - What is the difference between TSNE and UMAP?
  - How does perplexity affect TSNE results?
  - Why is TSNE not used for model training?
- **Tips**:
  - Explain TSNE’s focus on local structure and computational cost.
  - Highlight UMAP’s speed and scalability.
  - Be ready to code a TSNE visualization and discuss parameter tuning.

## 📚 Resources
- [Scikit-learn Manifold Documentation](https://scikit-learn.org/stable/modules/manifold.html)
- [UMAP Documentation](https://umap-learn.readthedocs.io/en/latest/)
- [Kaggle: Dimensionality Reduction Tutorial](https://www.kaggle.com/learn/feature-engineering)