import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, VotingClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
import matplotlib.pyplot as plt

# %% [1. Introduction to Ensemble Methods]
# Ensemble methods combine multiple models to improve performance.
# Scikit-learn provides RandomForest, GradientBoosting, AdaBoost, VotingClassifier, and StackingClassifier.

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Use Iris dataset: 150 samples, 4 features, 3 classes.
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
df = pd.DataFrame(X, columns=feature_names)
print("\nIris Dataset (first 5 rows):")
print(df.head())
print("\nDataset Shape:", X.shape)

# %% [3. Train-Test Split]
# Split data into training (80%) and testing (20%) sets.
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
print("\nTrain Shape:", X_train.shape, "Test Shape:", X_test.shape)

# %% [4. RandomForestClassifier]
# RandomForest combines multiple decision trees using bagging.
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("\nRandomForestClassifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_rf).round(4))
print("F1 Score (Macro):", f1_score(y_test, y_pred_rf, average='macro').round(4))

# %% [5. GradientBoostingClassifier]
# GradientBoosting builds trees sequentially to correct errors.
gb = GradientBoostingClassifier(random_state=42)
gb.fit(X_train, y_train)
y_pred_gb = gb.predict(X_test)
print("\nGradientBoostingClassifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_gb).round(4))
print("F1 Score (Macro):", f1_score(y_test, y_pred_gb, average='macro').round(4))

# %% [6. AdaBoostClassifier]
# AdaBoost boosts weak learners by adjusting weights.
ada = AdaBoostClassifier(random_state=42)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
print("\nAdaBoostClassifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_ada).round(4))
print("F1 Score (Macro):", f1_score(y_test, y_pred_ada, average='macro').round(4))

# %% [7. VotingClassifier]
# VotingClassifier combines predictions via majority voting or averaging.
estimators = [('lr', LogisticRegression(max_iter=200)), ('svc', SVC(probability=True)), ('rf', RandomForestClassifier(random_state=42))]
voting = VotingClassifier(estimators=estimators, voting='soft')
voting.fit(X_train, y_train)
y_pred_voting = voting.predict(X_test)
print("\nVotingClassifier (Soft):")
print("Accuracy:", accuracy_score(y_test, y_pred_voting).round(4))
print("F1 Score (Macro):", f1_score(y_test, y_pred_voting, average='macro').round(4))

# %% [8. StackingClassifier]
# StackingClassifier stacks predictions using a meta-learner.
stacking = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression(max_iter=200))
stacking.fit(X_train, y_train)
y_pred_stacking = stacking.predict(X_test)
print("\nStackingClassifier:")
print("Accuracy:", accuracy_score(y_test, y_pred_stacking).round(4))
print("F1 Score (Macro):", f1_score(y_test, y_pred_stacking, average='macro').round(4))

# %% [9. Practical Application: Comparing Models]
# Compare all ensemble models on test set performance.
models = {
    'RandomForest': y_pred_rf,
    'GradientBoosting': y_pred_gb,
    'AdaBoost': y_pred_ada,
    'Voting': y_pred_voting,
    'Stacking': y_pred_stacking
}
print("\nModel Comparison:")
for name, y_pred in models.items():
    print(f"{name}: Accuracy={accuracy_score(y_test, y_pred).round(4)}, F1={f1_score(y_test, y_pred, average='macro').round(4)}")

# %% [10. Interview Scenario: Ensemble Benefits]
# Discuss advantages of ensemble methods.
print("\nEnsemble Benefits:")
print("RandomForest: Reduces variance via bagging.")
print("GradientBoosting: Reduces bias via boosting.")
print("Stacking: Combines diverse models for improved predictions.")

# Plot feature importances for GradientBoosting
plt.figure()
plt.bar(feature_names, gb.feature_importances_)
plt.xticks(rotation=45)
plt.xlabel('Features')
plt.ylabel('Importance')
plt.title('GradientBoosting Feature Importances')
plt.tight_layout()
plt.savefig('gb_importances.png')