# Hyperparameter Tuning (`sklearn.model_selection`)

## 📖 Introduction
Hyperparameter tuning optimizes model performance by selecting the best parameter values. This guide covers `GridSearchCV` and `RandomizedSearchCV`, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand the role of hyperparameter tuning in ML.
- Master `GridSearchCV` and `RandomizedSearchCV`.
- Apply tuning with custom metrics and imbalanced data.
- Implement nested cross-validation for robust evaluation.

## 🔑 Key Concepts
- **GridSearchCV**: Exhaustively searches over a parameter grid.
- **RandomizedSearchCV**: Samples random parameter combinations, faster for large grids.
- **Scoring Metrics**: Customize evaluation (e.g., accuracy, F1 score).
- **Nested CV**: Combines inner tuning with outer validation to avoid overfitting.

## 📝 Example Walkthrough
The `hyperparameter_tuning.py` file demonstrates:
1. **Dataset**: Using the Iris dataset.
2. **Tuning**: Applying `GridSearchCV` and `RandomizedSearchCV` on an SVM.
3. **Custom Metrics**: Tuning with F1 score for balanced evaluation.
4. **Nested CV**: Evaluating tuning robustness with outer cross-validation.

Example code:
```python
from sklearn.model_selection import GridSearchCV
param_grid = {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf']}
grid = GridSearchCV(SVC(), param_grid, cv=5)
grid.fit(X, y)
```

## 🛠️ Practical Tasks
1. Use `GridSearchCV` to tune an SVM on the Iris dataset and find the best parameters.
2. Apply `RandomizedSearchCV` with a continuous parameter distribution (e.g., `C`).
3. Tune a model with `f1_macro` scoring on an imbalanced dataset.
4. Implement nested cross-validation to evaluate a tuned model.

## 💡 Interview Tips
- **Common Questions**:
  - What is the difference between `GridSearchCV` and `RandomizedSearchCV`?
  - When would you use nested cross-validation?
  - How do you choose hyperparameters to tune?
- **Tips**:
  - Highlight `RandomizedSearchCV`’s efficiency for large parameter spaces.
  - Explain the risk of overfitting without nested CV.
  - Be ready to code a tuning pipeline with `Pipeline`.

## 📚 Resources
- [Scikit-learn Hyperparameter Tuning Documentation](https://scikit-learn.org/stable/modules/grid_search.html)
- [Kaggle: Hyperparameter Tuning Tutorial](https://www.kaggle.com/learn/machine-learning-with-scikit-learn)