import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import pandas as pd
import numpy as np
from sklearn import metrics, svm, datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, classification_report


iris = datasets.load_iris()
cv_scores =[]
y = iris.target
x = iris.data[:, :4] 
folds = 10
ks = list(range(1,int(len(x) * ((folds - 1) / folds))))
ks = [k for k in ks if k % 3 !=0]
for k in ks:
    model = KNeighborsClassifier(n_neighbors=k)
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
    scores = cross_val_score(model, x, y, cv=cv, scoring='accuracy')
    mean = scores.mean()
    cv_scores.append(mean)
    #print(k,mean)

mse = [1 - x for x in cv_scores]
op_k = ks[mse.index(min(mse))]
#print(f"optimal k = {op_k}")

model = KNeighborsClassifier(n_neighbors=op_k)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))