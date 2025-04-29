import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder

# %% [1. Introduction to Encoding Categorical Variables]
# Categorical variables need encoding to numerical formats for ML models.
# Scikit-learn provides LabelEncoder, OneHotEncoder, and OrdinalEncoder for this purpose.

print("Scikit-learn version:", pd.__version__)

# %% [2. Dummy Dataset]
# Generate synthetic dataset: 100 samples with categorical features (color, size, brand).
np.random.seed(42)
data = {
    'color': np.random.choice(['red', 'blue', 'green'], 100),
    'size': np.random.choice(['small', 'medium', 'large'], 100),
    'brand': np.random.choice(['A', 'B', 'C'], 100)
}
df = pd.DataFrame(data)
print("\nDummy Dataset (first 5 rows):")
print(df.head())

# %% [3. LabelEncoder]
# LabelEncoder assigns a unique integer to each category, suitable for ordinal data or single-column encoding.

le = LabelEncoder()
df['color_encoded'] = le.fit_transform(df['color'])
print("\nLabelEncoder Results (first 5 rows):")
print(df[['color', 'color_encoded']].head())
print("Classes:", le.classes_)

# %% [4. OneHotEncoder]
# OneHotEncoder creates binary columns for each category, ideal for nominal data.

ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_ohe = ohe.fit_transform(df[['color']])
df_ohe = pd.DataFrame(X_ohe, columns=ohe.get_feature_names_out(['color']))
print("\nOneHotEncoder Results (first 5 rows):")
print(df_ohe.head())

# %% [5. OrdinalEncoder]
# OrdinalEncoder assigns integers to categories with an explicit order, for ordinal data.

oe = OrdinalEncoder(categories=[['small', 'medium', 'large']])
df['size_encoded'] = oe.fit_transform(df[['size']])
print("\nOrdinalEncoder Results (first 5 rows):")
print(df[['size', 'size_encoded']].head())
print("Categories:", oe.categories_)

# %% [6. Practical Application: Encoding Multiple Columns]
# Apply OneHotEncoder to multiple columns in a pipeline.

ohe_multi = OneHotEncoder(sparse=False)
X_multi = ohe_multi.fit_transform(df[['color', 'brand']])
df_multi = pd.DataFrame(X_multi, columns=ohe_multi.get_feature_names_out(['color', 'brand']))
print("\nOneHotEncoder Multi-Column Results (first 5 rows):")
print(df_multi.head())

# %% [7. Handling Unknown Categories]
# Use handle_unknown='ignore' in OneHotEncoder for unseen categories in test data.

from sklearn.model_selection import train_test_split
X_train, X_test = train_test_split(df[['color']], test_size=0.2, random_state=42)
ohe = OneHotEncoder(sparse=False, handle_unknown='ignore')
X_train_ohe = ohe.fit_transform(X_train)
X_test_ohe = ohe.transform(X_test)
print("\nOneHotEncoder Test Data (first 5 rows):")
print(pd.DataFrame(X_test_ohe, columns=ohe.get_feature_names_out(['color'])).head())

# %% [8. Interview Scenario: Encoding for a Classifier]
# Apply encoding before training a classifier to handle categorical features.

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

X = ohe_multi.fit_transform(df[['color', 'brand']])
y = np.random.randint(0, 2, 100)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nRandomForest Accuracy with Encoding:", accuracy_score(y_test, y_pred).round(4))