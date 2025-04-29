# Scaling and Normalization (`sklearn.preprocessing`)

## 📖 Introduction
Scaling and normalization are critical preprocessing steps in machine learning to ensure features have consistent scales, improving model convergence and performance. This guide covers `StandardScaler`, `MinMaxScaler`, and `RobustScaler`, with practical examples and interview-focused insights.

## 🎯 Learning Objectives
- Understand the purpose of scaling and normalization.
- Master the use of `StandardScaler`, `MinMaxScaler`, and `RobustScaler`.
- Learn to avoid data leakage in preprocessing pipelines.
- Apply scaling to improve model performance.

## 🔑 Key Concepts
- **StandardScaler**: Standardizes features to have mean=0 and variance=1, ideal for normally distributed data.
- **MinMaxScaler**: Scales features to a fixed range (e.g., [0, 1]), suitable for bounded inputs.
- **RobustScaler**: Uses median and IQR, robust to outliers, for non-normal data.
- **Data Leakage**: Fit scalers only on training data to prevent test data influencing preprocessing.

## 📝 Example Walkthrough
The `scaling_normalization.py` file demonstrates:
1. **Dummy Data**: Synthetic dataset with age, income, and score features.
2. **Scaling**: Applying `StandardScaler`, `MinMaxScaler`, and `RobustScaler`.
3. **Pipeline**: Avoiding data leakage by fitting on training data only.
4. **Model Impact**: Comparing SVM performance with and without scaling.

Example code:
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

## 🛠️ Practical Tasks
1. Generate a dataset with 3 numerical features and apply `StandardScaler`. Verify mean and variance.
2. Use `MinMaxScaler` to scale the same dataset to [-1, 1]. Check min and max values.
3. Apply `RobustScaler` to a dataset with outliers (e.g., extreme income values). Compare results.
4. Split a dataset, scale only the training set, and apply to a classifier (e.g., KNN).

## 💡 Interview Tips
- **Common Questions**:
  - What is the difference between `StandardScaler` and `MinMaxScaler`?
  - When should you use `RobustScaler`?
  - How do you prevent data leakage during preprocessing?
- **Tips**:
  - Explain why scaling is critical for distance-based algorithms (e.g., KNN, SVM).
  - Highlight `RobustScaler`’s advantage with outliers.
  - Be ready to code a scaling pipeline with `train_test_split`.

## 📚 Resources
- [Scikit-learn Preprocessing Documentation](https://scikit-learn.org/stable/modules/preprocessing.html)
- [Kaggle: Data Preprocessing Tutorial](https://www.kaggle.com/learn/data-cleaning)