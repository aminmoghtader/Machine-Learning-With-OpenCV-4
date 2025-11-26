import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import pandas as pd
import numpy as np
from sklearn import metrics, svm, datasets
import os
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
#os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = "/usr/lib/qt/plugins"
#os.environ["QT_QPA_PLATFORM"] = "eglfs"


# -----------------------------
#  بارگذاری داده‌ها
# -----------------------------
iris = datasets.load_iris()
x = iris.data[:, :2]   # فقط دو ویژگی اول: sepal length و sepal width
y = iris.target

#  تقسیم داده به Train و Test (۲۰٪ برای تست)
X_train, X_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
#  تعریف و آموزش مدل SVM
# -----------------------------
folds = 10
cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=42)
#model = svm.SVC(kernel='rbf', C=1.0, gamma='scale')
#scores = cross_val_score(model, x, y, cv=cv, scoring='accuracy')
#model = svm.SVC(kernel='poly', degree=4, C=1, gamma=5)
#model.fit(X_train, y_train)

param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': ['scale', 0.01, 0.1, 1, 10],
    #'degree': [3, 4],
    #'coef0': [0.0, 0.1, 0.5],
    #'kernel': ['poly']
    'kernel': ['rbf']
}

grid = GridSearchCV(svm.SVC(), param_grid, cv=cv, scoring='accuracy')
grid.fit(X_train, y_train)
print("Best parameters from CV:", grid.best_params_)
print("Best CV accuracy:", grid.best_score_)
model = grid.best_estimator_
model.fit(X_train, y_train)
# -----------------------------
#  تعیین محدوده محور برای رسم
# -----------------------------
x_min, x_max = x[:,0].min() - 1, x[:,0].max() + 1
y_min, y_max = x[:,1].min() - 1, x[:,1].max() + 1

# -----------------------------
#  تعیین گرید و ساخت mesh
# -----------------------------
h = (x_max / x_min) / 100  # اصلاح شده
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))

# -----------------------------
#  پیش‌بینی کلاس‌ها روی گرید
# -----------------------------
#z = model.predict(np.c_[xx.ravel(), yy.ravel()])
#z = z.reshape(xx.shape)

# -----------------------------
#  رسم decision boundary
# -----------------------------
#plt.contourf(xx, yy, z, cmap=plt.cm.Accent, alpha=0.8)

# -----------------------------
#  رسم نقاط واقعی داده‌ها
# -----------------------------
#colors = ['red', 'green', 'blue']
#for color, i, target in zip(colors, [0, 1, 2], iris.target_names):
    #plt.scatter(x[y==i, 0], x[y==i, 1], color=color, label=target, edgecolor='k')

# -----------------------------
#  برچسب‌ها و عنوان
# -----------------------------
#plt.xlabel('sepal length')
#plt.ylabel('sepal width')
#plt.title('SVC with poly Kernel')
#plt.legend(loc='best', shadow=False, scatterpoints=1)

# -----------------------------
#  نمایش نمودار
# -----------------------------
#plt.show()
#plt.savefig("plot.png")
# -----------------------------
#  پیش‌بینی روی داده اصلی و بررسی تعداد نمونه‌ها
# -----------------------------
y_pred = model.predict(X_test)
print("Predicted class counts:", np.unique(y_pred, return_counts=True))
#print("Accuracy:", accuracy_score(y_test, y_pred))
#print(classification_report(y_test, y_pred))
print("\nTest Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))