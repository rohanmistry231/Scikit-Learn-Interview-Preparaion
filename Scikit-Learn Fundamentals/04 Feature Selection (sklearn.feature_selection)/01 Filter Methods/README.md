# Filter Methods (`sklearn.feature_selection`)

## 📖 Introduction
Filter methods select features based on statistical properties, independent of the model. This guide covers `VarianceThreshold`, `SelectKBest`, `chi2`, `f_classif`, and `mutual_info_classif`, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand the role of filter methods in feature selection.
- Master `VarianceThreshold` and `SelectKBest` with different scoring functions.
- Apply filter methods to improve model performance.
- Build feature selection pipelines.

## 🔑 Key Concepts
- **VarianceThreshold**: Removes features with low variance (e.g., near-constant).
- **SelectKBest**: Selects top k features based on a scoring function.
- **Scoring Functions**:
  - `chi2`: Chi-squared test for non-negative data.
  - `f_classif`: ANOVA F-test for classification.
  - `mutual_info_classif`: Mutual information for non-linear relationships.
- **Advantages**: Fast, model-agnostic, and prevents overfitting.

## 📝 Example Walkthrough
The `filter_methods.py` file demonstrates:
1. **Dataset**: Iris dataset with added noisy/low-variance features.
2. **Feature Selection**: Applying `VarianceThreshold` and `SelectKBest` with `chi2`, `f_classif`, and `mutual_info_classif`.
3. **Model Impact**: Comparing RandomForest performance with/without selection.
4. **Pipeline**: Building a feature selection pipeline.

Example code:
```python
from sklearn.feature_selection import SelectKBest, f_classif
kbest = SelectKBest(score_func=f_classif, k=3)
X_selected = kbest.fit_transform(X, y)
```

## 🛠️ Practical Tasks
1. Apply `VarianceThreshold` to remove low-variance features from a dataset.
2. Use `SelectKBest` with `f_classif` to select top 3 features and verify scores.
3. Compare `chi2` and `mutual_info_classif` on a classification dataset.
4. Build a pipeline with `VarianceThreshold`, `SelectKBest`, and a classifier.

## 💡 Interview Tips
- **Common Questions**:
  - When should you use `chi2` vs. `f_classif`?
  - What are the advantages of filter methods over wrapper methods?
  - How does `VarianceThreshold` improve model performance?
- **Tips**:
  - Explain `chi2`’s requirement for non-negative data.
  - Highlight filter methods’ computational efficiency.
  - Be ready to code a `SelectKBest` pipeline and interpret feature scores.

## 📚 Resources
- [Scikit-learn Feature Selection Documentation](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Kaggle: Feature Selection Tutorial](https://www.kaggle.com/learn/feature-engineering)