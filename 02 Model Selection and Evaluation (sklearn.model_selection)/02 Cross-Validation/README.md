# Cross-Validation (`sklearn.model_selection`)

## 📖 Introduction
Cross-validation evaluates model performance by splitting data into multiple train-test folds, providing a robust estimate of generalization. This guide covers `KFold`, `StratifiedKFold`, `cross_val_score`, and `cross_validate`.

## 🎯 Learning Objectives
- Understand the purpose and benefits of cross-validation.
- Master `KFold`, `StratifiedKFold`, and related functions.
- Evaluate models with multiple metrics using `cross_validate`.
- Apply cross-validation to imbalanced datasets.

## 🔑 Key Concepts
- **KFold**: Splits data into k folds for cross-validation.
- **StratifiedKFold**: Preserves class distribution in folds, ideal for classification.
- **cross_val_score**: Computes performance scores for each fold.
- **cross_validate**: Evaluates multiple metrics and returns train/test scores.

## 📝 Example Walkthrough
The `cross_validation.py` file demonstrates:
1. **Dataset**: Using the Iris dataset.
2. **Cross-Validation**: Applying `KFold` and `StratifiedKFold` with `cross_val_score`.
3. **Multiple Metrics**: Using `cross_validate` for accuracy and F1 score.
4. **Imbalanced Data**: Handling class imbalance with `StratifiedKFold`.

Example code:
```python
from sklearn.model_selection import cross_val_score, StratifiedKFold
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(LogisticRegression(), X, y, cv=skf, scoring='accuracy')
```

## 🛠️ Practical Tasks
1. Apply `KFold` cross-validation to the Iris dataset and compute mean accuracy.
2. Use `StratifiedKFold` on an imbalanced dataset and compare with `KFold`.
3. Evaluate a model with `cross_validate` using accuracy and F1 score.
4. Implement manual KFold cross-validation to understand the process.

## 💡 Interview Tips
- **Common Questions**:
  - What is the difference between `KFold` and `StratifiedKFold`?
  - Why use cross-validation instead of a single train-test split?
  - How does `cross_validate` differ from `cross_val_score`?
- **Tips**:
  - Highlight cross-validation’s role in reducing variance in performance estimates.
  - Explain stratification for imbalanced datasets.
  - Be ready to code a cross-validation loop manually.

## 📚 Resources
- [Scikit-learn Cross-Validation Documentation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Kaggle: Model Validation Tutorial](https://www.kaggle.com/learn/machine-learning-with-scikit-learn)