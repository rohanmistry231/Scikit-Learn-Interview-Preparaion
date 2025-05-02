# Data Splitting (`sklearn.model_selection`)

## 📖 Introduction
Data splitting divides a dataset into training and testing sets to evaluate model performance. This guide covers `train_test_split`, focusing on its use, stratification, and interview-relevant scenarios.

## 🎯 Learning Objectives
- Understand the purpose of data splitting.
- Master `train_test_split` with and without stratification.
- Apply splitting in ML pipelines.
- Handle imbalanced datasets with stratified splits.

## 🔑 Key Concepts
- **Train-Test Split**: Divides data into training (model learning) and testing (model evaluation) sets.
- **Stratification**: Ensures class distribution is preserved in splits, critical for imbalanced data.
- **Random State**: Controls reproducibility of splits.
- **Test Size**: Proportion of data allocated to the test set (e.g., 0.2 for 20%).

## 📝 Example Walkthrough
The `data_splitting.py` file demonstrates:
1. **Dataset**: Using the Iris dataset (150 samples, 4 features).
2. **Splitting**: Applying `train_test_split` with and without stratification.
3. **Custom Splits**: Experimenting with different test sizes.
4. **Model Impact**: Training a LogisticRegression model to compare split strategies.

Example code:
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
```

## 🛠️ Practical Tasks
1. Split the Iris dataset with a 70-30 train-test ratio and verify shapes.
2. Apply stratified splitting on an imbalanced dataset and check class distributions.
3. Train a classifier on split data and compare accuracy with and without stratification.
4. Perform multiple splits with different random seeds and compute mean accuracy.

## 💡 Interview Tips
- **Common Questions**:
  - Why is stratification important for imbalanced datasets?
  - How does `random_state` affect `train_test_split`?
  - What is the typical test size for a train-test split?
- **Tips**:
  - Emphasize stratification for maintaining class proportions in classification tasks.
  - Explain the importance of a separate test set to avoid overfitting.
  - Be ready to code a stratified split for an imbalanced dataset.

## 📚 Resources
- [Scikit-learn Model Selection Documentation](https://scikit-learn.org/stable/modules/cross_validation.html)
- [Kaggle: Data Splitting Tutorial](https://www.kaggle.com/learn/data-cleaning)