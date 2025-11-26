import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import pandas as pd
import numpy as np
from sklearn import metrics, svm
import os
from sklearn.model_selection import train_test_split, \
    StratifiedKFold, cross_val_score, GridSearchCV
from sklearn.metrics import accuracy_score, \
    classification_report, confusion_matrix
import cv2
import random
import gc

# -----------------------------
#  بارگذاری داده‌ها
# -----------------------------

f_n = "/home/am/Downloads/p_neg/"
f_p = "/home/am/Downloads/ps/"

win_size = (64, 128)
block_size = (16, 16)
block_stride = (8, 8)
cell_size = (8, 8)
num_bins = 9
hog = cv2.HOGDescriptor(win_size, block_size, block_stride, cell_size, num_bins)

random.seed(42)

# مثبت‌ها
x_pos = []
files = os.listdir(f_p)
selected = random.sample(files, min(len(files), 400))
for fname in selected:
    path = os.path.join(f_p, fname)
    img = cv2.imread(path)
    img = cv2.resize(img, win_size)
    if img is None:
        print("Could not read:", path)
        continue
    x_pos.append(hog.compute(img))
x_pos = np.array(x_pos, dtype=np.float32)
y_pos = np.ones(x_pos.shape[0], dtype=np.int32)
x_neg = []
#hroi = 128
#wroi = 64
hroi, wroi = win_size[1], win_size[0]
for neg in os.listdir(f_n):
    path = os.path.join(f_n, neg)
    img = cv2.imread(path)
    img = cv2.resize(img, (512, 512))
    for j in range(5):
        rand_y = random.randint(0, img.shape[0] - hroi)
        rand_x = random.randint(0, img.shape[1] - wroi)
        roi = img[rand_y:rand_y + hroi, rand_x:rand_x + wroi, :]
        x_neg.append(hog.compute(roi))

x_neg = np.array(x_neg, dtype=np.float32)
y_neg = -np.ones(x_neg.shape[0], dtype=np.int32)
X = np.concatenate((x_pos, x_neg))
y = np.concatenate((y_pos, y_neg))
print("X shape:", X.shape)
print("Y shape:", y.shape)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

def t_svm(X_train, y_train, kernel_type=cv2.ml.SVM_LINEAR):
    svm = cv2.ml.SVM_create()
    svm.setKernel(kernel_type)
    svm.setType(cv2.ml.SVM_C_SVC)
    svm.setC(0.01)

    if kernel_type in [cv2.ml.SVM_RBF, cv2.ml.SVM_POLY, cv2.ml.SVM_SIGMOID]:
        svm.setGamma(0.5)

    svm.train(X_train, cv2.ml.ROW_SAMPLE, y_train)
    return svm

def score_svm(svm, X, y):
    _, pred = svm.predict(X)
    pred = pred.ravel()  # تبدیل به 1D
    #print("\nTest Accuracy:", accuracy_score(y, pred))
    #print("\nConfusion Matrix:\n", confusion_matrix(y, pred))
    #print("\nClassification Report:\n", classification_report(y, pred))
    return accuracy_score(y, pred)

#svm = t_svm(X_train, y_train, kernel_type=cv2.ml.SVM_RBF)
#score_svm(svm, X_test, y_test)
#rho, _, _ = svm.getDecisionFunction(0)
#sv = svm.getSupportVectors()
#detector = np.append(sv[0], -rho)
#hog.setSVMDetector(detector)

score_train = []
score_test = []
pred_test_list = []

for j in range(3):
    svm = t_svm(X_train, y_train, kernel_type=cv2.ml.SVM_LINEAR)
    score_train.append(score_svm(svm, X_train, y_train))
    score_test.append(score_svm(svm, X_test, y_test))
    _, pred = svm.predict(X_test)
    pred_test_list.append(pred)
    false_pos = np.logical_and((y_test.ravel() == -1), (pred.ravel() == 1))
    if not np.any(false_pos):
        print('done')
        break
    X_train = np.concatenate((X_train, X_test[false_pos, :]), axis=0)
    y_train = np.concatenate((y_train, y_test[false_pos]), axis=0)


print(score_train)
print(score_test)
for i, pred in enumerate(pred_test_list):
    print(f"\n=== Iteration {i+1} ===")
    print("\nConfusion Matrix:\n", confusion_matrix(y_test, pred))
    print("\nClassification Report:\n", classification_report(y_test, pred))

# -----------------------------
# sliding window برای تست تصویر
# -----------------------------
stride = 16
img_test = cv2.imread('/home/am/Downloads/test.jpg')
found = []

for y in range(0, img_test.shape[0] - hroi, stride):
    for x in range(0, img_test.shape[1] - wroi, stride):
        roi = img_test[y:y+hroi, x:x+wroi]
        feat = np.array([hog.compute(roi, (64,64))], dtype=np.float32)
        _, pred = svm.predict(feat)
        if np.allclose(pred, 1):
            found.append((x, y, wroi, hroi))

#rho, _, _ = svm.getDecisionFunction(0)
#sv = svm.getSupportVectors()
#hog.setSVMDetector(np.append(sv[0, :].ravel(), rho))
#found = hog.detectMultiScale(img_test)
#hogdef = cv2.HOGDescriptor()
#pdetect = cv2.HOGDescriptor_getDefaultPeopleDetector()
#hogdef.setSVMDetector(pdetect)
#found, _ = hog.detectMultiScale(
    #img_test,
    #winStride=(8, 8),       # گام کوچک‌تر برای حساسیت بالاتر
    #padding=(8, 8),
    #scale=1.05,              # pyramid scale
    #hitThreshold=0          # حساسیت بالاتر
#)

fig = plt.figure(figsize=(10, 6))
ax = fig.add_subplot(111)
ax.imshow(cv2.cvtColor(img_test, cv2.COLOR_BGR2RGB))
for f in found:
    # تبدیل f به آرایه numpy و سپس flatten
    f = np.array(f).ravel()  
    if f.size == 4:  
        x, y, w, h = f
        ax.add_patch(mpa.Rectangle((x, y), w, h, color='r', linewidth=2, fill=False))
plt.savefig('detected.png')
plt.close(fig)



#del all
gc.collect()