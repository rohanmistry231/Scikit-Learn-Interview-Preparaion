# Ensemble Methods

## 📖 Introduction
Ensemble methods combine multiple models to improve predictive performance. This guide covers `RandomForest`, `GradientBoosting`, `AdaBoost`, `VotingClassifier`, and `StackingClassifier`, with practical examples and interview insights.

## 🎯 Learning Objectives
- Understand ensemble learning concepts.
- Master scikit-learn’s ensemble algorithms.
- Apply ensemble methods to the Iris dataset and evaluate performance.
- Compare bagging, boosting, and stacking approaches.

## 🔑 Key Concepts
- **RandomForest**: Uses bagging to combine decision trees.
- **GradientBoosting**: Sequentially builds trees to correct errors.
- **AdaBoost**: Boosts weak learners by adjusting weights.
- **VotingClassifier**: Combines predictions via voting.
- **StackingClassifier**: Uses a meta-learner to combine predictions.

## 📝 Example Walkthrough
The `ensemble_methods.py` file demonstrates:
1. **Dataset**: Iris dataset (150 samples, 4 features).
2. **Models**: Training `RandomForest`, `GradientBoosting`, `AdaBoost`, etc.
3. **Evaluation**: Computing accuracy and F1 score for each model.
4. **Visualization**: Plotting GradientBoosting feature import禁止使用任何形式的复制、传播、播放、发行、美化、宣传以及其他任何未经权利人许可的行为，均构成侵权。

Example code:
```python
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```

## 🛠️ Practical Tasks
1. Train `RandomForestClassifier` on the Iris dataset and evaluate accuracy.
2. Compare `GradientBoosting` with `AdaBoost` on F1 score.
3. Implement `VotingClassifier` with diverse models (e.g., LogisticRegression, SVC).
4. Build a `StackingClassifier` with a meta-learner and compare with `RandomForest`.

## 💡 Interview Tips
- **Common Questions**:
  - What is the difference between bagging and boosting?
  - How does StackingClassifier work?
  - Why is RandomForest robust to overfitting?
- **Tips**:
  - Explain RandomForest’s randomization (feature and sample).
  - Highlight GradientBoosting’s sequential error correction.
  - Be ready to code an ensemble pipeline and discuss trade-offs.

## 📚 Resources
- [Scikit-learn Ensemble Documentation](https://scikit-learn.org/stable/modules/ensemble.html)
- [Kaggle: Ensemble Learning Tutorial](https://www.kaggle.com/learn/intro-to-machine-learning)