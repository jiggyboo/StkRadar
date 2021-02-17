import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tslearn.clustering import TimeSeriesKMeans
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

# Making ML Models to compare future stocks with.

seed = 0
cl = 1 # number of clusters
gbw = 2

input = [pd.read_csv('Data/best_time.csv'),pd.read_csv('Data/bad_time.csv'),pd.read_csv('Data/worst_time.csv')] # preparing Data
input[gbw] = input[gbw].iloc[:,1:]
input[gbw] = input[gbw].to_numpy()
result = np.transpose(input[gbw])
# result = TimeSeriesScalerMeanVariance(1,.5).fit_transform(result) 

km = TimeSeriesKMeans(n_clusters = cl, verbose=True, random_state=seed)
y_pred = km.fit_predict(result)
# y_pred.to_json('MLModels/swkm.json')
# print(y_pred.transform(result))
print(result)
plt.figure()
for yi in range(cl):
    plt.subplot(3, 3, yi + 1)
    for xx in result[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, 15)
    plt.ylim(0, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    plt.title("Euclidean $k$-means")

dba_km = TimeSeriesKMeans(n_clusters=cl,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10,
                          random_state=seed)
y_pred = dba_km.fit_predict(result)
# y_pred.to_json('MLModels/swdba_km.json')

for yi in range(cl):
    plt.subplot(3, 3, 4 + yi)
    for xx in result[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, 15)
    plt.ylim(0, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    plt.title("DBA $k$-means")

sdtw_km = TimeSeriesKMeans(n_clusters=cl,
                           metric="softdtw",
                           metric_params={"gamma": .01},
                           verbose=True,
                           random_state=seed)
y_pred = sdtw_km.fit_predict(result)
# y_pred.to_json('MLModels/swsdtw_km.json')

for yi in range(cl):
    plt.subplot(3, 3, 7 + yi)
    for xx in result[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, 15)
    plt.ylim(0, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    plt.title("Soft-DTW $k$-means")

plt.tight_layout()
plt.show()