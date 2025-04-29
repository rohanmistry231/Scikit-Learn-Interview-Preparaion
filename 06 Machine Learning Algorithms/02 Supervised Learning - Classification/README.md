# Supervised Learning - Classification

## 📖 Introduction
Classification algorithms predict discrete class labels in supervised learning. This guide covers `LogisticRegression`, `SVC`, `RandomForestClassifier`, `GradientBoostingClassifier`, and `KNeighborsClassifier`, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand classification algorithms and their assumptions.
- Master scikit-learn’s classification models.
- Apply models to the Iris dataset and evaluate performance.
- Compare model strengths and weaknesses.

## 🔑 Key Concepts
- **LogisticRegression**: Uses logistic function for class probability prediction.
- **SVC**: Support vector classifier for linear/non-linear boundaries.
- **RandomForestClassifier**: Ensemble of decision trees for robustness.
- **GradientBoostingClassifier**: Sequential trees for error correction.
- **KNeighborsClassifier**: Non-parametric, distance-based classification.

## 📝 Example Walkthrough
The `classification_algorithms.py` file demonstrates:
1. **Dataset**: Iris dataset (150 samples, 4 features).
2. **Models**: Training `LogisticRegression`, `SVC`, `RandomForestClassifier`, etc.
3. **Evaluation**: Computing accuracy and F1 score for each model.
4. **Visualization**: Plotting RandomForest feature importances.

Example code:
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

## 🛠️ Practical Tasks
1. Train `LogisticRegression` on the Iris dataset and compute accuracy.
2. Compare `SVC` with linear and RBF kernels.
3. Evaluate `RandomForestClassifier` feature importances and interpret results.
4. Tune `KNeighborsClassifier`’s `n_neighbors` and compare performance.

## 💡 Interview Tips
- **Common Questions**:
  - When would you use SVC over LogisticRegression?
  - How does RandomForest handle overfitting?
  - What are the advantages of GradientBoosting?
- **Tips**:
  - Explain RandomForest’s bagging for robustness.
  - Highlight SVC’s kernel trick for non-linear data.
  - Be ready to code a classification pipeline and discuss model selection.

## 📚 Resources
- [Scikit-learn Classification Documentation](https://scikit-learn.org/stable/supervised_learning.html)
- [Kaggle: Classification Tutorial](https://www.kaggle.com/learn/intro-to-machine-learning)