# Handling Missing Values (`sklearn.preprocessing`)

## 📖 Introduction
Handling missing values is essential to ensure machine learning models can process incomplete datasets. This guide covers `SimpleImputer` and `KNNImputer`, with examples and interview-focused insights.

## 🎯 Learning Objectives
- Understand strategies for imputing missing values.
- Master `SimpleImputer` for numerical and categorical data.
- Apply `KNNImputer` for neighbor-based imputation.
- Integrate imputation in ML pipelines.

## 🔑 Key Concepts
- **SimpleImputer**: Replaces missing values with mean, median, most_frequent, or constant.
- **KNNImputer**: Uses k-nearest neighbors to impute numerical values, considering data patterns.
- **Pipeline Integration**: Fit imputers on training data to avoid data leakage.
- **Categorical vs. Numerical**: Different strategies apply based on data type.

## 📝 Example Walkthrough
The `handling_missing_values.py` file demonstrates:
1. **Dummy Data**: Synthetic dataset with missing values in age, income, and category.
2. **Imputation**: Applying `SimpleImputer` (mean, median, most_frequent) and `KNNImputer`.
3. **Pipeline**: Imputing in a train-test split to prevent leakage.
4. **Classifier**: Using imputed data for a RandomForest classifier.

Example code:
```python
from sklearn.impute import SimpleImputer
imp = SimpleImputer(strategy='mean')
X_imputed = imp.fit_transform(X)
```

## 🛠️ Practical Tasks
1. Impute missing numerical values using `SimpleImputer` with mean and median strategies.
2. Apply `SimpleImputer` with 'most_frequent' to a categorical column.
3. Use `KNNImputer` on a numerical feature and compare with mean imputation.
4. Build a pipeline with imputation and a classifier, ensuring no data leakage.

## 💡 Interview Tips
- **Common Questions**:
  - What are the pros and cons of mean vs. KNN imputation?
  - How do you handle missing categorical values?
  - Why is fitting imputers on training data important?
- **Tips**:
  - Explain when `KNNImputer` is preferred (e.g., correlated features).
  - Discuss the impact of missing data on model performance.
  - Be ready to code an imputation pipeline with `Pipeline`.

## 📚 Resources
- [Scikit-learn Imputation Documentation](https://scikit-learn.org/stable/modules/impute.html)
- [Kaggle: Handling Missing Values](https://www.kaggle.com/learn/data-cleaning)