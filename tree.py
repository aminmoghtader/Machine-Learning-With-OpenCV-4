import random
import pandas as pd
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.tree import DecisionTreeClassifier
import gc

# ===== ثابت کردن seed =====
random.seed(42)
np.random.seed(42)

# ===== تولید داده =====
def data_augment(num_entries):
    data = []
    for _ in range(num_entries):
        data.append({
            'age': random.randint(20,100),
            'sex': random.choice(['M','F']),
            'BP': random.choice(['low','high','normal']),
            'Choletrol': random.choice(['low','high','normal']),
            'Na': random.random(),
            'K': random.random(),
            'drug': random.choice(['A','B','C','D'])
        })
    return data

# ===== آماده سازی داده =====
data = data_augment(500)
df = pd.DataFrame(data)
target = df.pop('drug')

vec = DictVectorizer()
X = vec.fit_transform(df.to_dict(orient='records')).toarray().astype(np.float32)    
le = LabelEncoder()
y = le.fit_transform(target).astype(np.float32)  

# تقسیم داده
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ===== مدل درخت تصمیم =====
tree = DecisionTreeClassifier(random_state=42)
tree.fit(X_train, y_train)
pred = tree.predict(X_test)

# ===== محاسبه Accuracy =====
print("\nTest Accuracy:", accuracy_score(y_test, pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
print("\nClassification Report:\n", classification_report(y_test, pred))

# ===== آزاد کردن حافظه =====
del df, target, X, y, X_train, X_test, y_train, y_test, pred, tree
gc.collect()
