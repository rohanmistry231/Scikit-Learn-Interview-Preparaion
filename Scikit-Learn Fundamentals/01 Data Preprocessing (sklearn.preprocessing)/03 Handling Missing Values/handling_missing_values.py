import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer

# %% [1. Introduction to Handling Missing Values]
# Missing values can disrupt ML models. Scikit-learn provides SimpleImputer and KNNImputer
# to impute missing data based on statistical or neighbor-based methods.

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Generate synthetic dataset: 100 samples with missing values in numerical and categorical features.
np.random.seed(42)
data = {
    'age': np.random.randint(18, 80, 100).astype(float),
    'income': np.random.normal(50000, 15000, 100),
    'category': np.random.choice(['A', 'B', 'C'], 100)
}
df = pd.DataFrame(data)
# Introduce missing values
df.loc[np.random.choice(df.index, 20), 'age'] = np.nan
df.loc[np.random.choice(df.index, 15), 'income'] = np.nan
df.loc[np.random.choice(df.index, 10), 'category'] = np.nan

print("\nDummy Dataset with Missing Values (first 5 rows):")
print(df.head())
print("\nMissing Values Count:")
print(df.isna().sum())

# %% [3. SimpleImputer (Numerical)]
# SimpleImputer replaces missing values with mean, median, or constant for numerical data.

imp_mean = SimpleImputer(strategy='mean')
df['age_mean'] = imp_mean.fit_transform(df[['age']])
print("\nSimpleImputer (Mean) for Age (first 5 rows):")
print(df[['age', 'age_mean']].head())

imp_median = SimpleImputer(strategy='median')
df['income_median'] = imp_median.fit_transform(df[['income']])
print("\nSimpleImputer (Median) for Income (first 5 rows):")
print(df[['income', 'income_median']].head())

# %% [4. SimpleImputer (Categorical)]
# SimpleImputer with 'most_frequent' strategy for categorical data.

imp_cat = SimpleImputer(strategy='most_frequent')
df['category_frequent'] = imp_cat.fit_transform(df[['category']])
print("\nSimpleImputer (Most Frequent) for Category (first 5 rows):")
print(df[['category', 'category_frequent']].head())

# %% [5. KNNImputer]
# KNNImputer uses k-nearest neighbors to impute missing values, suitable for numerical data.

knn_imp = KNNImputer(n_neighbors=5)
df['age_knn'] = knn_imp.fit_transform(df[['age']])
print("\nKNNImputer for Age (first 5 rows):")
print(df[['age', 'age_knn']].head())

# %% [6. Practical Application: Imputation in a Pipeline]
# Impute missing values in a train-test split to avoid data leakage.

from sklearn.model_selection import train_test_split
X = df[['age', 'income']].values
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

imp = SimpleImputer(strategy='mean')
X_train_imp = imp.fit_transform(X_train)
X_test_imp = imp.transform(X_test)
print("\nImputed Train Data (first 5 rows):")
print(pd.DataFrame(X_train_imp, columns=['age', 'income']).head())

# %% [7. Comparing Imputation Methods]
# Compare imputation results for numerical features.

print("\nImputation Comparison for Age (first 5 rows):")
print(df[['age', 'age_mean', 'age_knn']].head())
print("\nMean vs. KNN Age Statistics:")
print("Mean Imputed Mean:", df['age_mean'].mean().round(4))
print("KNN Imputed Mean:", df['age_knn'].mean().round(4))

# %% [8. Interview Scenario: Imputation for a Classifier]
# Apply imputation before training a classifier to handle missing values.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

imp = SimpleImputer(strategy='mean')
X_imp = imp.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_imp, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nRandomForest Accuracy with Imputation:", accuracy_score(y_test, y_pred).round(4))