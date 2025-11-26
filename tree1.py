import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn import datasets
import gc


# -----------------------------
#  بارگذاری داده‌ها
# -----------------------------
cancer = datasets.load_breast_cancer()
x = cancer.data   
y = cancer.target

X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

params = {
    "max_depth": [None, 3, 5, 10, 20],
    "max_leaf_nodes": [None, 10, 20, 50, 100],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 5, 10]
}

grid = GridSearchCV(
    DecisionTreeClassifier(),
    params,
    cv=5,
    scoring="accuracy",
    n_jobs=-1  # برای استفاده از تمام هسته‌های CPU
)

grid.fit(X_train, y_train)
print(grid.best_params_)
print(grid.best_score_)
print(grid.best_estimator_)
model = grid.best_estimator_

if model:
    model.fit(X_train, y_train)
    pred = model.predict(X_test)

# ===== محاسبه Accuracy =====
print("\nTest Accuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))


"""
rng = np.random.RandomState(42)
x = np.sort(5 * rng.rand(100,1), axis=0)
y = np.sin(x).ravel()
y[::2] += 0.5 * (0.5 - rng.rand(50))
model1 = DecisionTreeRegressor(max_depth=2, random_state=42)
model2 = DecisionTreeRegressor(max_depth=5, random_state=42)
model1.fit(x,y)
model2.fit(x,y)
X_test = np.arange(0, 0.5, 0.01)[:, np.newaxis]
pred1 = model1.predict(X_test)
pred2 = model2.predict(X_test)
df = pd.DataFrame(pred2)
print(df.head())

"""

# ===== آزاد کردن حافظه =====
# del 
gc.collect()