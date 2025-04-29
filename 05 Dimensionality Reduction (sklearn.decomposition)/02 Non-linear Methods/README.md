# Non-linear Methods (`sklearn.decomposition`)

## 📖 Introduction
Non-linear dimensionality reduction captures complex, non-linear patterns in data. This guide covers `KernelPCA`, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand the principles of non-linear dimensionality reduction.
- Master `KernelPCA` with different kernels (RBF, polynomial).
- Apply KernelPCA for visualization and model improvement.
- Build pipelines with non-linear methods.

## 🔑 Key Concepts
- **KernelPCA**: Uses kernel trick to project data into a lower-dimensional space.
- **Kernels**: RBF and polynomial kernels capture non-linear relationships.
- **Gamma**: Controls the flexibility of the RBF kernel.
- **Preprocessing**: Standardization is essential for KernelPCA.

## 📝 Example Walkthrough
The `non_linear_methods.py` file demonstrates:
1. **Dataset**: Iris dataset.
2. **Dimensionality Reduction**: Applying `KernelPCA` with RBF and polynomial kernels.
3. **Visualization**: Plotting KernelPCA-transformed data.
4. **Pipeline**: Combining KernelPCA with a classifier.

Example code:
```python
from sklearn.decomposition import KernelPCA
kpca = KernelPCA(n_components=2, kernel='rbf', gamma=0.1)
X_kpca = kpca.fit_transform(X_scaled)
```

## 🛠️ Practical Tasks
1. Apply `KernelPCA` with RBF kernel to the Iris dataset and visualize the results.
2. Experiment with polynomial kernel and compare with RBF.
3. Tune the `gamma` parameter in `KernelPCA` and evaluate classifier performance.
4. Build a pipeline with `KernelPCA` and a classifier, comparing with PCA.

## 💡 Interview Tips
- **Common Questions**:
  - How does `KernelPCA` differ from PCA?
  - What is the role of the kernel in `KernelPCA`?
  - How do you choose the `gamma` parameter?
- **Tips**:
  - Explain the kernel trick and non-linear mappings.
  - Highlight KernelPCA’s ability to capture complex patterns.
  - Be ready to code a `KernelPCA` pipeline and discuss kernel choice.

## 📚 Resources
- [Scikit-learn Decomposition Documentation](https://scikit-learn.org/stable/modules/decomposition.html)
- [Kaggle: Dimensionality Reduction Tutorial](https://www.kaggle.com/learn/feature-engineering)