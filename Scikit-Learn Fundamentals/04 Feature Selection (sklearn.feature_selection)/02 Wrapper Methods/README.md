# Wrapper Methods (`sklearn.feature_selection`)

## 📖 Introduction
Wrapper methods select features by evaluating subsets using a specific model, optimizing performance. This guide covers `RFE` (Recursive Feature Elimination) and `RFECV`, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand the role of wrapper methods in feature selection.
- Master `RFE` and `RFECV` for model-based feature selection.
- Apply wrapper methods to improve model performance.
- Build feature selection pipelines with wrapper methods.

## 🔑 Key Concepts
- **RFE**: Recursively eliminates least important features based on model coefficients or importance.
- **RFECV**: Uses cross-validation to select the optimal number of features.
- **Model Dependency**: Wrapper methods rely on a specific model (e.g., LogisticRegression).
- **Trade-off**: More accurate than filter methods but computationally expensive.

## 📝 Example Walkthrough
The `wrapper_methods.py` file demonstrates:
1. **Dataset**: Iris dataset.
2. **Feature Selection**: Applying `RFE` and `RFECV` with LogisticRegression and RandomForest.
3. **Model Impact**: Comparing RandomForest performance with/without selection.
4. **Pipeline**: Building an RFE-based pipeline.

Example code:
```python
from sklearn.feature_selection import RFE
rfe = RFE(estimator=LogisticRegression(), n_features_to_select=2)
X_rfe = rfe.fit_transform(X, y)
```

## 🛠️ Practical Tasks
1. Apply `RFE` to select 2 features from the Iris dataset and check feature rankings.
2. Use `RFECV` to find the optimal number of features with cross-validation.
3. Compare `RFE` performance with different estimators (e.g., LogisticRegression vs. RandomForest).
4. Build a pipeline with `RFE` and a classifier, evaluating accuracy.

## 💡 Interview Tips
- **Common Questions**:
  - How does `RFE` differ from filter methods?
  - What is the advantage of `RFECV` over `RFE`?
  - Why are wrapper methods computationally expensive?
- **Tips**:
  - Explain RFE’s iterative elimination process.
  - Highlight RFECV’s use of cross-validation for robustness.
  - Be ready to code an RFE pipeline and interpret feature rankings.

## 📚 Resources
- [Scikit-learn Feature Selection Documentation](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Kaggle: Feature Selection Tutorial](https://www.kaggle.com/learn/feature-engineering)