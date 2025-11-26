import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from sklearn import metrics
import sys

# ============================================
# 1. Read CSV
# ============================================
file = "/home/am/Downloads/Body.csv"
df = pd.read_csv(file)
# ============================================
# 2. Clean column names
# ============================================
df.columns = df.columns.str.strip()
# ============================================
# 3. Select features for clustering
# ============================================
features = ['Belly', 'Waist', 'LegLength']
# Check missing columns
for f in features:
    if f not in df.columns:
        raise ValueError(f"Column missing: {f}")
# ============================================
# 4. Remove outliers
#    (anything > 3 standard deviations from mean)
# ============================================
df_clean = df.copy()
for f in features:
    mean = df_clean[f].mean()
    std = df_clean[f].std()
    df_clean = df_clean[(df_clean[f] > mean - 3*std) & (df_clean[f] < mean + 3*std)]

print("Data points after removing outliers:", len(df_clean))
# ============================================
# 5. Extract data matrix
# ============================================
X = df_clean[features].values
# ============================================
# 6. Standardize features
# ============================================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
# ============================================
# 7. K-Means
# ============================================
silhouettle_avg = []
min_k = 3
for k in range(min_k, 10):    
    kmeans = KMeans(n_clusters=k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    score = metrics.silhouette_score(X_scaled, labels)
    silhouettle_avg.append(score)
opt_k = silhouettle_avg.index(max(silhouettle_avg)) + min_k
print(f"optimal K = {opt_k}")
if opt_k:
    kmeans = KMeans(n_clusters=opt_k, random_state=42)
    labels = kmeans.fit_predict(X_scaled)
    print("\nCluster Centers (after scaling):")
    center = kmeans.cluster_centers_
    print(center)
    print("original size:")
    print(scaler.inverse_transform(center))
else:
    print("optimal k not valid")
    sys.exit()
# ============================================
# 9. Plot 3D
# ============================================
fig = plt.figure(figsize=(7, 6))
ax = fig.add_subplot(111, projection='3d')

ax.scatter(X_scaled[:, 0], X_scaled[:, 1], X_scaled[:, 2],
           c=labels, s=20)
#ax.scatter(center[:,0],center[:,1],center[:,2], marker='*', s=100, c='black')
ax.set_xlabel("Belly (scaled)")
ax.set_ylabel("Waist (scaled)")
ax.set_zlabel("LegLength (scaled)")
plt.title("K-Means Clustering (3D)")
plt.savefig('k.png')