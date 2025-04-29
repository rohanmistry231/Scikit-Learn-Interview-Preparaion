import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.decomposition import FastICA, FactorAnalysis
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# %% [1. Introduction to Other Decomposition Techniques]
# Other decomposition techniques extract independent or latent factors from data.
# Scikit-learn provides FastICA (Independent Component Analysis) and FactorAnalysis.

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
# Standardize features for FastICA and FactorAnalysis.
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
print("\nScaled Data (first 5 rows):")
print(pd.DataFrame(X_scaled, columns=feature_names).head())

# %% [4. FastICA]
# FastICA separates data into statistically independent components.
ica = FastICA(n_components=2, random_state=42)
X_ica = ica.fit_transform(X_scaled)
print("\nFastICA Transformed Data Shape:", X_ica.shape)

# %% [5. FactorAnalysis]
# FactorAnalysis models data as a combination of latent factors and noise.
fa = FactorAnalysis(n_components=2, random_state=42)
X_fa = fa.fit_transform(X_scaled)
print("\nFactorAnalysis Transformed Data Shape:", X_fa.shape)

# %% [6. Visualizing FastICA Results]
# Plot FastICA-transformed data with class labels.
plt.figure()
for i, target_name in enumerate(iris.target_names):
    plt.scatter(X_ica[y == i, 0], X_ica[y == i, 1], label=target_name)
plt.xlabel('Independent Component 1')
plt.ylabel('Independent Component 2')
plt.title('FastICA on Iris Dataset')
plt.legend()
plt.savefig('ica_plot.png')

# %% [7. Practical Application: Model Performance]
# Compare classifier performance with and without FastICA.
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42, stratify=y)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nAccuracy without FastICA:", accuracy_score(y_test, y_pred).round(4))

# With FastICA
X_train_ica = ica.fit_transform(X_train)
X_test_ica = ica.transform(X_test)
clf.fit(X_train_ica, y_train)
y_pred_ica = clf.predict(X_test_ica)
print("Accuracy with FastICA:", accuracy_score(y_test, y_pred_ica).round(4))

# %% [8. Comparing FastICA and FactorAnalysis]
# Compare classifier performance with FactorAnalysis.
X_train_fa = fa.fit_transform(X_train)
X_test_fa = fa.transform(X_test)
clf.fit(X_train_fa, y_train)
y_pred_fa = clf.predict(X_test_fa)
print("\nAccuracy with FactorAnalysis:", accuracy_score(y_test, y_pred_fa).round(4))
print("FastICA vs. FactorAnalysis: FastICA seeks independence, FactorAnalysis models latent factors.")

# %% [9. Interview Scenario: FastICA Applications]
# Discuss FastICA’s use in signal separation.
print("\nFastICA Application Example:")
print("FastICA is used in blind source separation (e.g., separating mixed audio signals).")
print("It assumes non-Gaussian, independent sources.")

# %% [10. Pipeline with FastICA]
# Build a pipeline with FastICA and a classifier.
from sklearn.pipeline import Pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('ica', FastICA(n_components=2, random_state=42)),
    ('clf', RandomForestClassifier(random_state=42))
])
pipeline.fit(X_train, y_train)
y_pred_pipe = pipeline.predict(X_test)
print("\nPipeline Accuracy with FastICA:", accuracy_score(y_test, y_pred_pipe).round(4))