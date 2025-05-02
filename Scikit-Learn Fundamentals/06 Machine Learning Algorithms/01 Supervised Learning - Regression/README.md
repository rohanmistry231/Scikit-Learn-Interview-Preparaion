# Supervised Learning - Regression

## 📖 Introduction
Regression algorithms predict continuous target variables in supervised learning. This guide covers `LinearRegression`, `Ridge`, `Lasso`, `SVR`, and `ElasticNet`, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand regression algorithms and their assumptions.
- Master scikit-learn’s regression models.
- Apply models to synthetic data and evaluate performance.
- Compare regularization techniques (L1, L2).

## 🔑 Key Concepts
- **LinearRegression**: Fits a linear model minimizing squared errors.
- **Ridge**: Adds L2 regularization to reduce overfitting.
- **Lasso**: Adds L1 regularization, promoting sparsity.
- **SVR**: Support vector regression for non-linear problems.
- **ElasticNet**: Combines L1 and L2 regularization.

## 📝 Example Walkthrough
The `regression_algorithms.py` file demonstrates:
1. **Dataset**: Synthetic regression dataset (200 samples, 3 features).
2. **Models**: Training `LinearRegression`, `Ridge`, `Lasso`, `SVR`, and `ElasticNet`.
3. **Evaluation**: Computing MSE and R² for each model.
4. **Visualization**: Plotting predictions for comparison.

Example code:
```python
from sklearn.linear_model import LinearRegression
lr = LinearRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
```

## 🛠️ Practical Tasks
1. Train `LinearRegression` on a synthetic dataset and compute MSE.
2. Compare `Ridge` and `Lasso` coefficients to understand regularization.
3. Apply `SVR` with different kernels (e.g., linear, RBF) and evaluate performance.
4. Tune `ElasticNet`’s `l1_ratio` and compare with `Lasso`.

## 💡 Interview Tips
- **Common Questions**:
  - What is the difference between L1 and L2 regularization?
  - When would you use SVR over LinearRegression?
  - Why does Lasso produce sparse models?
- **Tips**:
  - Explain Lasso’s feature selection via sparsity.
  - Highlight Ridge’s ability to handle multicollinearity.
  - Be ready to code a regression pipeline and interpret coefficients.

## 📚 Resources
- [Scikit-learn Regression Documentation](https://scikit-learn.org/stable/modules/linear_model.html)
- [Kaggle: Regression Tutorial](https://www.kaggle.com/learn/intro-to-machine-learning)