# Regression Metrics (`sklearn.metrics`)

## 📖 Introduction
Regression metrics evaluate the performance of regression models by measuring prediction errors and explained variance. This guide covers `mean_squared_error`, `mean_absolute_error`, and `r2_score`.

## 🎯 Learning Objectives
- Understand the role of regression metrics.
- Master MSE, MAE, and R² for evaluating models.
- Analyze the impact of outliers on metrics.
- Compare models using appropriate metrics.

## 🔑 Key Concepts
- **Mean Squared Error (MSE)**: Average squared error, sensitive to outliers.
- **Mean Absolute Error (MAE)**: Average absolute error, robust to outliers.
- **R² Score**: Proportion of variance explained, ranges from 0 to 1 (or negative).
- **Outlier Sensitivity**: MSE is more affected by outliers than MAE.

## 📝 Example Walkthrough
The `regression_metrics.py` file demonstrates:
1. **Dataset**: Synthetic dataset with size, rooms, and price.
2. **Model**: Training a LinearRegression model.
3. **Metrics**: Computing MSE, MAE, and R².
4. **Outliers**: Evaluating metrics with an introduced outlier.

Example code:
```python
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
```

## 🛠️ Practical Tasks
1. Train a regression model on a synthetic dataset and compute MSE and MAE.
2. Compare a model’s R² score with a baseline (mean predictor).
3. Introduce outliers to the test set and analyze their impact on metrics.
4. Plot actual vs. predicted values to visualize model performance.

## 💡 Interview Tips
- **Common Questions**:
  - What is the difference between MSE and MAE?
  - When is R² a poor metric for regression?
  - How do outliers affect regression metrics?
- **Tips**:
  - Explain MSE’s sensitivity to outliers vs. MAE’s robustness.
  - Highlight R²’s limitations (e.g., negative values for poor models).
  - Be ready to code a metric calculation and interpret results.

## 📚 Resources
- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Kaggle: Regression Metrics Tutorial](https://www.kaggle.com/learn/machine-learning-with-scikit-learn)