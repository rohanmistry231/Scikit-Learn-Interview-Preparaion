import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.preprocessing import label_binarize

# %% [1. Introduction to Classification Metrics]
# Classification metrics evaluate the performance of classification models.
# Scikit-learn provides accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, and roc_auc_score.

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Use the Iris dataset: 150 samples, 4 features, 3 classes.
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
df['target'] = y

print("\nIris Dataset (first 5 rows):")
print(df.head())
print("\nDataset Shape:", X.shape)

# %% [3. Train-Test Split and Model Training]
# Split data and train a LogisticRegression model.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
clf = LogisticRegression(random_state=42, max_iter=200)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nPredicted Labels (first 5):", y_pred[:5])

# %% [4. Accuracy Score]
# Accuracy measures the proportion of correct predictions.

accuracy = accuracy_score(y_test, y_pred)
print("\nAccuracy Score:", accuracy.round(4))

# %% [5. Precision, Recall, and F1 Score]
# Precision, recall, and F1 score evaluate performance for multi-class problems (macro averaging).

precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
print("\nPrecision (Macro):", precision.round(4))
print("Recall (Macro):", recall.round(4))
print("F1 Score (Macro):", f1.round(4))

# %% [6. Confusion Matrix]
# Confusion matrix shows true vs. predicted labels.

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(cm)

# %% [7. Classification Report]
# Classification report provides precision, recall, F1, and support for each class.

report = classification_report(y_test, y_pred, target_names=iris.target_names)
print("\nClassification Report:")
print(report)

# %% [8. ROC AUC Score]
# ROC AUC score measures the area under the ROC curve for multi-class (one-vs-rest).

y_test_bin = label_binarize(y_test, classes=[0, 1, 2])
y_score = clf.predict_proba(X_test)
roc_auc = roc_auc_score(y_test_bin, y_score, multi_class='ovr')
print("\nROC AUC Score (One-vs-Rest):", roc_auc.round(4))

# %% [9. Practical Application: Imbalanced Data]
# Simulate imbalanced data and evaluate metrics.

mask = y != 2
X_imb = np.vstack([X[mask], X[y == 2][:10]])
y_imb = np.hstack([y[mask], y[y == 2][:10]])
X_train_imb, X_test_imb, y_train_imb, y_test_imb = train_test_split(
    X_imb, y_imb, test_size=0.2, random_state=42, stratify=y_imb
)
clf.fit(X_train_imb, y_train_imb)
y_pred_imb = clf.predict(X_test_imb)
print("\nImbalanced Data Metrics:")
print("Accuracy:", accuracy_score(y_test_imb, y_pred_imb).round(4))
print("F1 Score (Macro):", f1_score(y_test_imb, y_pred_imb, average='macro').round(4))
print("Classification Report:")
print(classification_report(y_test_imb, y_pred_imb))

# %% [10. Interview Scenario: Choosing Metrics]
# Evaluate model with different metrics to justify choice for imbalanced data.

print("\nMetric Choice for Imbalanced Data:")
print("Accuracy may be misleading for imbalanced data.")
print("F1 Score (macro) balances precision and recall across classes.")
print("ROC AUC is useful for probabilistic predictions.")