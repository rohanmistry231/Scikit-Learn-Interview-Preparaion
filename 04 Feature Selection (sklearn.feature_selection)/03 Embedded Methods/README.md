# Embedded Methods (`sklearn.feature_selection`)

## 📖 Introduction
Embedded methods integrate feature selection into the model training process, leveraging model-specific properties. This guide covers `SelectFromModel` and feature importance (e.g., RandomForest), with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand the role of embedded methods in feature selection.
- Master `SelectFromModel` with RandomForest and Lasso.
- Apply embedded methods to improve model performance.
- Build feature selection pipelines with embedded methods.

## 🔑 Key Concepts
- **SelectFromModel**: Selects features based on model importance or coefficients.
- **Feature Importance**: RandomForest provides importance scores for ranking features.
- **L1 Regularization**: Lasso shrinks irrelevant feature coefficients to zero.
- **Advantages**: Balances accuracy and efficiency compared to wrapper methods.

## 📝 Example Walkthrough
The `embedded_methods.py` file demonstrates:
1. **Dataset**: Iris dataset.
2. **Feature Selection**: Applying `SelectFromModel` with RandomForest and Lasso.
3. **Model Impact**: Comparing RandomForest performance with/without selection.
4. **Pipeline**: Building a SelectFromModel-based pipeline.

Example code:
```python
from sklearn.feature_selection import SelectFromModel
selector = SelectFromModel(estimator=RandomForestClassifier())
X_selected = selector.fit_transform(X, y)
```

## 🛠️ Practical Tasks
1. Apply `SelectFromModel` with RandomForest to select features from the Iris dataset.
2. Use `SelectFromModel` with Lasso and verify selected features.
3. Compare performance with different thresholds in `SelectFromModel`.
4. Build a pipeline with `SelectFromModel` and a classifier, evaluating accuracy.

## 💡 Interview Tips
- **Common Questions**:
  - How does `SelectFromModel` differ from RFE?
  - Why use Lasso for feature selection?
  - What are the benefits of embedded methods?
- **Tips**:
  - Explain how RandomForest importance scores drive selection.
  - Highlight Lasso’s sparsity-inducing property.
  - Be ready to code a `SelectFromModel` pipeline and interpret importances.

## 📚 Resources
- [Scikit-learn Feature Selection Documentation](https://scikit-learn.org/stable/modules/feature_selection.html)
- [Kaggle: Feature Selection Tutorial](https://www.kaggle.com/learn/feature-engineering)