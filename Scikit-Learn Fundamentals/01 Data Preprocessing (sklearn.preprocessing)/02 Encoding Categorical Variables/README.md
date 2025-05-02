# Encoding Categorical Variables (`sklearn.preprocessing`)

## 📖 Introduction
Encoding categorical variables transforms non-numerical data into formats suitable for machine learning models. This guide covers `LabelEncoder`, `OneHotEncoder`, and `OrdinalEncoder`, with examples and interview-focused insights.

## 🎯 Learning Objectives
- Understand when to use different encoding methods.
- Master `LabelEncoder`, `OneHotEncoder`, and `OrdinalEncoder`.
- Handle multiple categorical features and unknown categories.
- Apply encoding in ML pipelines.

## 🔑 Key Concepts
- **LabelEncoder**: Assigns integers to categories, best for single-column or ordinal data.
- **OneHotEncoder**: Creates binary columns for each category, ideal for nominal data.
- **OrdinalEncoder**: Assigns integers with a specified order, for ordinal data.
- **Handling Unknowns**: Use `handle_unknown='ignore'` in `OneHotEncoder` for test data robustness.

## 📝 Example Walkthrough
The `encoding_categorical.py` file demonstrates:
1. **Dummy Data**: Synthetic dataset with color, size, and brand features.
2. **Encoding**: Applying `LabelEncoder`, `OneHotEncoder`, and `OrdinalEncoder`.
3. **Multi-Column Encoding**: Using `OneHotEncoder` for multiple features.
4. **Classifier**: Encoding data for a RandomForest classifier.

Example code:
```python
from sklearn.preprocessing import OneHotEncoder
ohe = OneHotEncoder(sparse=False)
X_ohe = ohe.fit_transform(df[['color']])
```

## 🛠️ Practical Tasks
1. Encode a single categorical column using `LabelEncoder` and verify the mapping.
2. Apply `OneHotEncoder` to a nominal feature (e.g., color) and check the binary output.
3. Use `OrdinalEncoder` for an ordinal feature (e.g., size) with a custom order.
4. Encode multiple categorical features and train a classifier, ensuring no data leakage.

## 💡 Interview Tips
- **Common Questions**:
  - When should you use `OneHotEncoder` vs. `LabelEncoder`?
  - How do you handle unknown categories in test data?
  - What are the risks of using `LabelEncoder` for nominal data?
- **Tips**:
  - Explain why `OneHotEncoder` prevents ordinal assumptions in nominal data.
  - Highlight `sparse` parameter for memory efficiency in large datasets.
  - Be ready to code an encoding pipeline with `ColumnTransformer`.

## 📚 Resources
- [Scikit-learn Preprocessing Documentation](https://scikit-learn.org/stable/modules/preprocessing.html#preprocessing-categorical-features)
- [Kaggle: Categorical Data Tutorial](https://www.kaggle.com/learn/feature-engineering)