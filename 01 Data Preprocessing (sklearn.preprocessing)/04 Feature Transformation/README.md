# Feature Transformation (`sklearn.preprocessing`)

## 📖 Introduction
Feature transformation enhances model performance by creating or modifying features to capture complex patterns. This guide covers `PolynomialFeatures`, `PowerTransformer`, and `FunctionTransformer`, with examples and interview-focused insights.

## 🎯 Learning Objectives
- Understand the role of feature transformation in ML.
- Master `PolynomialFeatures`, `PowerTransformer`, and `FunctionTransformer`.
- Apply transformations to capture non-linear relationships.
- Integrate transformations in ML pipelines.

## 🔑 Key Concepts
- **PolynomialFeatures**: Generates polynomial and interaction terms for non-linear modeling.
- **PowerTransformer**: Applies power transformations (e.g., Yeo-Johnson) to normalize data.
- **FunctionTransformer**: Applies custom functions for flexible transformations.
- **Pipeline Integration**: Combine transformations with models for robust feature engineering.

## 📝 Example Walkthrough
The `feature_transformation.py` file demonstrates:
1. **Dummy Data**: Synthetic dataset with price and quantity features.
2. **Transformation**: Applying `PolynomialFeatures`, `PowerTransformer`, and `FunctionTransformer`.
3. **Pipeline**: Using transformations in a regression task.
4. **Model Impact**: Comparing regression performance with and without transformations.

Example code:
```python
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X)
```

## 🛠️ Practical Tasks
1. Apply `PolynomialFeatures` to a dataset and verify the new feature columns.
2. Use `PowerTransformer` to normalize skewed data and check distribution.
3. Implement a custom log transformation with `FunctionTransformer`.
4. Build a pipeline with `PolynomialFeatures` and a regression model, comparing performance.

## 💡 Interview Tips
- **Common Questions**:
  - When should you use `PolynomialFeatures`?
  - What is the difference between Box-Cox and Yeo-Johnson in `PowerTransformer`?
  - How do transformations improve model performance?
- **Tips**:
  - Explain how `PolynomialFeatures` captures non-linear relationships.
  - Discuss computational cost of high-degree polynomials.
  - Be ready to code a transformation pipeline with `Pipeline`.

## 📚 Resources
- [Scikit-learn Feature Transformation Documentation](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing)
- [Kaggle: Feature Engineering Tutorial](https://www.kaggle.com/learn/feature-engineering)