import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# %% [1. Introduction to Visualization Aids]
# Visualization aids reduce data to 2D or 3D for visual exploration.
# Scikit-learn provides TSNE, and UMAP is available via umap-learn (requires installation).

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

# %% [3. TSNE]
# TSNE reduces data to 2D by preserving local structure (non-linear).
tsne = TSNE(n_components=2, random_state=42, perplexity=30)
X_tsne = tsne.fit_transform(X)
print("\nTSNE Transformed Data Shape:", X_tsne.shape)

# %% [4. Visualizing TSNE Results]
# Plot TSNE-transformed data with class labels.
plt.figure()
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=target_name)
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.title('TSNE on Iris Dataset')
plt.legend()
plt.savefig('tsne_plot.png')

# %% [5. UMAP (Optional)]
# UMAP is a faster, non-linear method but requires umap-learn.
try:
    from umap import UMAP
    umap = UMAP(n_components=2, random_state=42)
    X_umap = umap.fit_transform(X)
    print("\nUMAP Transformed Data Shape:", X_umap.shape)

    # Plot UMAP-transformed data
    plt.figure()
    for i, target_name in enumerate(iris.target_names):
        plt.scatter(X_umap[y == i, 0], X_umap[y == i, 1], label=target_name)
    plt.xlabel('UMAP Component 1')
    plt.ylabel('UMAP Component 2')
    plt.title('UMAP on Iris Dataset')
    plt.legend()
    plt.savefig('umap_plot.png')
except ImportError:
    print("\nUMAP not installed. Install with: pip install umap-learn")

# %% [6. Practical Application: Comparing TSNE and PCA]
# Compare TSNE with PCA for visualization.
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
plt.figure()
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_pca[y == i, 0], X_pca[y == i, 1], label=target_name)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('PCA on Iris Dataset')
plt.legend()
plt.savefig('pca_comparison_plot.png')
print("\nTSNE vs. PCA: TSNE preserves local structure, PCA maximizes global variance.")

# %% [7. Tuning TSNE Parameters]
# Experiment with different perplexity values for TSNE.
perplexities = [5, 30, 50]
for perp in perplexities:
    tsne = TSNE(n_components=2, random_state=42, perplexity=perp)
    X_tsne = tsne.fit_transform(X)
    plt.figure()
    for i, target_name in enumerate(iris.target_names):
        plt.scatter(X_tsne[y == i, 0], X_tsne[y == i, 1], label=target_name)
    plt.xlabel('TSNE Component 1')
    plt.ylabel('TSNE Component 2')
    plt.title(f'TSNE with Perplexity={perp}')
    plt.legend()
    plt.savefig(f'tsne_perplexity_{perp}.png')

# %% [8. Interview Scenario: TSNE vs. UMAP]
# Discuss differences between TSNE and UMAP.
print("\nTSNE vs. UMAP:")
print("TSNE: Focuses on local structure, computationally intensive.")
print("UMAP: Faster, preserves both local and global structure, scalable.")

# %% [9. High-Dimensional Synthetic Data]
# Apply TSNE to a high-dimensional synthetic dataset.
from sklearn.datasets import make_classification
X_high, y_high = make_classification(n_samples=200, n_features=20, n_classes=3, random_state=42)
tsne = TSNE(n_components=2, random_state=42)
X_tsne_high = tsne.fit_transform(X_high)
plt.figure()
for i in range(3):
    plt.scatter(X_tsne_high[y_high == i, 0], X_tsne_high[y_high == i, 1], label=f'Class {i}')
plt.xlabel('TSNE Component 1')
plt.ylabel('TSNE Component 2')
plt.title('TSNE on High-Dimensional Synthetic Data')
plt.legend()
plt.savefig('tsne_high_dim_plot.png')

# %% [10. Combining Visualization with Models]
# Use TSNE-transformed data for classification (demonstrative, not typical).
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
tsne = TSNE(n_components=2, random_state=42)
X_train_tsne = tsne.fit_transform(X_train)
X_test_tsne = tsne.fit_transform(X_test)  # Note: TSNE is typically for visualization
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train_tsne, y_train)
y_pred = clf.predict(X_test_tsne)
print("\nAccuracy with TSNE-transformed Data (Demonstrative):", accuracy_score(y_test, y_pred).round(4))
print("Note: TSNE is primarily for visualization, not model input.")