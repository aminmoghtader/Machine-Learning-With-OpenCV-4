import os
#os.environ["QT_QPA_PLATFORM"] = "xcb"
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import make_circles
from sklearn import preprocessing

#X, y = make_circles(n_samples=100, noise=0.09)
#rgb = np.array(['r','g','b'])
#plt.scatter(X[:,0], X[:,1], color=rgb[y])
#plt.show()
rss = 5.4444
print("rss:%.2f" %rss)
f = "/home/am/Documents/AI/anaconda3/envs/ML/img/results/output_rgbf1.csv"
df = pd.read_csv(f)
#df.isnull().sum()
#df.R = df.R.fillna(df.R.mean())
#df = df.dropna()
#df = df.reset_index(drop=True)
a = bool("true")
print(a)
df = df[df.duplicated(keep=False)]
df.drop_duplicates(keep='first', inplace=True)
df.drop_duplicates(subset=['A'], keep='last', inplace=True)

x = df.values.astype(float)
scaler = preprocessing.MinMaxScaler()
x_scaled = scaler.fit_transform(x)
df = pd.DataFrame(x_scaled, columns=df.columns)

def outliers_iqr(data):
    q1, q3 = np.percentile(data, [25,75])
    iqr = q3 - q1
    lower = q1 - (iqr * 1.5)
    upper = q3 - (iqr * 1.5)
    return np.where((data > upper) | (data < lower))

for i in outliers_iqr(df.height)[0]:
    print(df[i:i+1])

def outliers_z(data):
    threshold = 3
    mean = np.mean(data)
    std = np.std(data)
    z = [(y - mean) / std for y in data]
    return np.where(np.abs(z) > threshold)

for i in outliers_iqr(df.height)[0]:
    print(df[i:i+1])
