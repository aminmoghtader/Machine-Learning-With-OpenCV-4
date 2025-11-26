import matplotlib.pyplot as plt
import matplotlib.patches as mpa
import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_california_housing, load_breast_cancer
from mpl_toolkits.mplot3d import Axes3D
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
import os
#os.environ["QT_QPA_PLATFORM"] = "xcb"

#dataset = fetch_california_housing()
#df = pd.DataFrame(dataset.data, columns=dataset.feature_names)
#df['medv'] = dataset.target
#print(df.head())
#df.info()
#print(df.isnull().sum())
#corr = df.corr()
#print(corr)
#print(df.columns)
#print(corr.abs().nlargest(5, 'medv').values[:,8])
#plt.scatter(df['MedInc'], df['medv'], marker='o')
#plt.xlabel('MedInc')
#plt.ylabel('medv')
#plt.show()
#fig = plt.figure(figsize=(5,5))
#ax = fig.add_subplot(111, projection = '3d')
#ax.scatter(df['HouseAge'], df['Latitude'], df['medv'],c='b')
#plt.show()
#x = pd.DataFrame(np.c_[df['MedInc'], df['AveRooms'], df['Latitude'], df['HouseAge']],
                 #columns=['MedInc', 'AveRooms', 'Latitude', 'HouseAge'])
#Y = df['medv']
#x_train, x_test, Y_train, Y_test = train_test_split(x, Y, test_size=0.3, random_state=5)
#model = LinearRegression()
#model.fit(x_train,Y_train)
#pr_pred = model.predict(x_test)
#print('R-Squ: %.4f' %model.score(x_test,Y_test))
#mse = mean_squared_error(Y_test, pr_pred)
#print(mse)
#plt.scatter(Y_test,pr_pred)
#plt.show()
#deg = 3
#p = PolynomialFeatures(degree=deg)
#x_train_p = p.fit_transform(x_train)
#print(p.get_feature_names_out())
#model = LinearRegression()
#model.fit(x_train_p,Y_train)
#x_test_p = p.fit_transform(x_test)
#print('R-Squ: %.4f' %model.score(x_test_p,Y_test))
#print(model.intercept_)
#print(model.coef_)
cancer = load_breast_cancer()
#x = []
#for target in range(2):
    #x.append([[],[],[]])
    #for i in range(len(cancer.data)):
        #if cancer.target[i] == target:
            #x[target][0].append(cancer.data[i][0])
            #x[target][1].append(cancer.data[i][1])
            #x[target][2].append(cancer.data[i][2])
#colours = ("r","b")
#fig = plt.figure(figsize=(5,6))
#ax = fig.add_subplot(111, projection = '3d')
#for target in range(2):
    #ax.scatter(x[target][0], x[target][1], x[target][2], c=colours[target])
#ax.set_xlabel("mean radius")
#ax.set_ylabel("mean texture")
#ax.set_zlabel("mean perimeter")
#plt.show()
X = cancer.data[:,0]
y = cancer.target
colors = {0:'red',1:'blue'}
edgecolors = np.array(['red', 'blue'])[y]
#plt.scatter(X,y, facecolors='none', edgecolors=edgecolors)
#plt.xlabel("mean radius")
#plt.ylabel("result")
red = mpa.Patch(color='red', label = 'malignant')
blue = mpa.Patch(color='blue', label = 'benign')
#plt.legend(handles=[red,blue], loc=1)
#plt.show()
#l_regr = LogisticRegression()
#l_regr.fit(X=np.array(X).reshape(len(X),1), y=y)
#print(l_regr.intercept_)
#print(l_regr.coef_)                   
#def sigmoid(x):
    #return (1 / (1 + np.exp(-(l_regr.intercept_[0] + (l_regr.coef_[0][0] * x)))))
#x1 = np.arange(0,30,0.01)
#y1 = [sigmoid(n) for n in x1]
#edgecolors = np.array(['red', 'blue'])[y]
#plt.scatter(X,y, facecolors='none', edgecolors=edgecolors)
#plt.plot(x1,y1)
#plt.xlabel("mean radius")
#plt.ylabel("probability")
#plt.show()
#print(l_regr.predict_proba([[8]]))
#print(l_regr.predict([[8]])[0])
train_set, test_set, train_labels, test_labels = train_test_split(
    cancer.data,
    cancer.target,
    test_size=0.25,
    random_state=1,
    stratify=cancer.target
)
x = train_set[:,0:30]
y = train_labels
model = Pipeline([
    ('scalar', StandardScaler()),
    ("logreg", LogisticRegression(
        solver="lbfgs",
        max_iter=500
        ))
    ]) 

model.fit(X=x,y=y)
print("Coefficients:", model.named_steps["logreg"].coef_)
print("Intercept:", model.named_steps["logreg"].intercept_)
pre_prob = pd.DataFrame(model.predict_proba(X=test_set))
pre_prob.columns = ["Malignant","Benign"]
preds = model.predict(X=test_set)
pre_class = pd.DataFrame(preds)
pre_class.columns = ["prediction"]
original_res = pd.DataFrame(test_labels)
original_res.columns = ["original"]
result = pd.concat([pre_prob,pre_class,original_res], axis=1)
#print(result.head())
metric = metrics.confusion_matrix(y_true=test_labels, y_pred=preds)
#print(metric)
report = metrics.classification_report(
    y_true=test_labels,
    y_pred=preds,
    output_dict=True
)
report.pop("macro avg", None)
report.pop("weighted avg", None)
df = pd.DataFrame(report).transpose().round(2)
#print(df)
probs = model.predict_proba(test_set)
preds = probs[:,1]
fpr, tpr, threshold = roc_curve(test_labels, preds)
#print(fpr)
#print(tpr)
#print(threshold)
roc = auc(fpr,tpr)
plt.plot(fpr, tpr, label='AUC = %0.2f' %roc)
plt.plot([0,1],[0,1],'r--')
plt.xlim([0,1])
plt.ylim([0,1])
plt.ylabel('TPR')
plt.xlabel('FPR')
plt.title('ROC')
plt.legend(loc='lower right')
#plt.show()
print(model.classes_)
plt.savefig("roc_curve.png", dpi=300)