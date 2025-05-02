# Classification Metrics (`sklearn.metrics`)

## 📖 Introduction
Classification metrics evaluate the performance of classification models, providing insights into accuracy, precision, and more. This guide covers `accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `confusion_matrix`, `classification_report`, and `roc_auc_score`.

## 🎯 Learning Objectives
- Understand the role of classification metrics.
- Master key metrics for multi-class and imbalanced data.
- Interpret confusion matrices and classification reports.
- Apply metrics in practical ML scenarios.

## 🔑 Key Concepts
- **Accuracy**: Proportion of correct predictions.
- **Precision/Recall/F1**: Evaluate class-specific performance (macro for multi-class).
- **Confusion Matrix**: Shows true vs. predicted labels.
- **Classification Report**: Summarizes precision, recall, F1, and support.
- **ROC AUC**: Measures area under the ROC curve (one-vs-rest for multi-class).

## 📝 Example Walkthrough
The `classification_metrics.py` file demonstrates:
1. **Dataset**: Using the Iris dataset.
2. **Model**: Training a LogisticRegression model.
3. **Metrics**: Computing accuracy, precision, recall, F1, confusion matrix, and ROC AUC.
4. **Imbalanced Data**: Evaluating metrics on a simulated imbalanced dataset.

Example code:
```python
from sklearn.metrics import accuracy_score, f1_score
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred, average='macro')
```

## 🛠️ Practical Tasks
1. Train a classifier on the Iris dataset and compute accuracy and F1 score.
2. Generate a confusion matrix and interpret the results.
3. Compute ROC AUC for a multi-class problem using predict_proba.
4. Evaluate a classifier on an imbalanced dataset and compare accuracy vs. F1 score.

## 💡 Interview Tips
- **Common Questions**:
  - What is the difference between precision and recall?
  - Why is F1 score preferred for imbalanced data?
  - How do you interpret a confusion matrix?
- **Tips**:
  - Explain when accuracy is misleading (e.g., imbalanced data).
  - Highlight F1 score’s balance of precision and recall.
  - Be ready to code a classification report and explain its components.

## 📚 Resources
- [Scikit-learn Metrics Documentation](https://scikit-learn.org/stable/modules/model_evaluation.html)
- [Kaggle: Classification Metrics Tutorial](https://www.kaggle.com/learn/machine-learning-with-scikit-learn)