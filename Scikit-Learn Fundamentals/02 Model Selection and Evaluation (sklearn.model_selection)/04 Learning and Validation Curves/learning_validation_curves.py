import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import learning_curve, validation_curve
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# %% [1. Introduction to Learning and Validation Curves]
# Learning and validation curves diagnose model performance by analyzing training size and hyperparameter effects.
# Scikit-learn provides learning_curve and validation_curve for this purpose.

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Use the Iris dataset: 150 samples, 4 features, 3 classes.
iris = load_iris()
X = iris.data
y = iris.target
df = pd.DataFrame(X, columns=iris.feature_names)
print("\nIris Dataset (first 5 rows):")
print(df.head())
print("\nDataset Shape:", X.shape)

# %% [3. Learning Curve]
# Learning curve shows training and validation scores as a function of training set size.

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(random_state=42), X, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10)
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)
print("\nLearning Curve Results:")
print("Train Sizes:", train_sizes)
print("Train Scores Mean:", train_mean.round(4))
print("Validation Scores Mean:", val_mean.round(4))

# Plot learning curve
plt.figure()
plt.plot(train_sizes, train_mean, label='Training Score')
plt.plot(train_sizes, val_mean, label='Validation Score')
plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(train_sizes, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Training Examples')
plt.ylabel('Accuracy')
plt.title('Learning Curve')
plt.legend()
plt.savefig('learning_curve.png')

# %% [4. Validation Curve]
# Validation curve shows training and validation scores as a function of a hyperparameter.

param_range = np.arange(1, 21, 2)
train_scores, val_scores = validation_curve(
    RandomForestClassifier(random_state=42), X, y, param_name='n_estimators',
    param_range=param_range, cv=5, scoring='accuracy'
)
train_mean = np.mean(train_scores, axis=1)
train_std = np.std(train_scores, axis=1)
val_mean = np.mean(val_scores, axis=1)
val_std = np.std(val_scores, axis=1)
print("\nValidation Curve Results:")
print("Parameter Range (n_estimators):", param_range)
print("Train Scores Mean:", train_mean.round(4))
print("Validation Scores Mean:", val_mean.round(4))

# Plot validation curve
plt.figure()
plt.plot(param_range, train_mean, label='Training Score')
plt.plot(param_range, val_mean, label='Validation Score')
plt.fill_between(param_range, train_mean - train_std, train_mean + train_std, alpha=0.1)
plt.fill_between(param_range, val_mean - val_std, val_mean + val_std, alpha=0.1)
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
plt.title('Validation Curve')
plt.legend()
plt.savefig('validation_curve.png')

# %% [5. Practical Application: Diagnosing Bias and Variance]
# Use learning curve to diagnose high bias or variance.

train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=5, random_state=42), X, y, cv=5, scoring='accuracy',
    train_sizes=np.linspace(0.1, 1.0, 10)
)
val_mean = np.mean(val_scores, axis=1)
print("\nLearning Curve for Underfit Model (n_estimators=5):")
print("Validation Scores Mean:", val_mean.round(4))
print("Diagnosis: High bias if validation score plateaus early and is low.")

# %% [6. Learning Curve with Imbalanced Data]
# Apply learning curve on an imbalanced dataset.

mask = y != 2
X_imb = np.vstack([X[mask], X[y == 2][:10]])
y_imb = np.hstack([y[mask], y[y == 2][:10]])
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(random_state=42), X_imb, y_imb, cv=StratifiedKFold(n_splits=5),
    scoring='f1_macro', train_sizes=np.linspace(0.1, 1.0, 10)
)
val_mean = np.mean(val_scores, axis=1)
print("\nLearning Curve for Imbalanced Data (F1 Macro):")
print("Validation Scores Mean:", val_mean.round(4))

# %% [7. Combining Curves for Model Selection]
# Use validation curve to select optimal n_estimators, then confirm with learning curve.

optimal_n = param_range[np.argmax(val_mean)]
print("\nOptimal n_estimators from Validation Curve:", optimal_n)
train_sizes, train_scores, val_scores = learning_curve(
    RandomForestClassifier(n_estimators=optimal_n, random_state=42), X, y, cv=5,
    scoring='accuracy', train_sizes=np.linspace(0.1, 1.0, 10)
)
print("Learning Curve with Optimal n_estimators:")
print("Validation Scores Mean:", np.mean(val_scores, axis=1).round(4))

# %% [8. Interview Scenario: Interpreting Curves]
# Analyze a learning curve to explain model performance.

print("\nLearning Curve Interpretation:")
print("If training and validation scores converge at a high value: Well-fit model.")
print("If training score is high but validation score is low: Overfitting (high variance).")
print("If both scores are low and converge: Underfitting (high bias).")