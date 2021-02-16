import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from numpy import genfromtxt

from tslearn.clustering import TimeSeriesKMeans
from tslearn.datasets import CachedDatasets
from tslearn.preprocessing import TimeSeriesScalerMeanVariance, \
    TimeSeriesResampler

seed = 0
inputd = pd.read_csv('dataset.csv')
inputd = inputd.to_numpy()
inputd = inputd.transpose()
inputd = inputd[1:]
inputd = np.concatenate((inputd[::3],inputd[2::3]))
km = TimeSeriesKMeans(n_clusters = 2, verbose=True, random_state=seed)
y_pred = km.fit_predict(inputd)
model = km.fit(inputd)
model.to_json('km.json')
print(inputd)
plt.figure()
for yi in range(2):
    plt.subplot(3, 3, yi + 1)
    for xx in inputd[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, 15)
    plt.ylim(0, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    plt.title("Euclidean $k$-means")

dba_km = TimeSeriesKMeans(n_clusters=2,
                          n_init=2,
                          metric="dtw",
                          verbose=True,
                          max_iter_barycenter=10,
                          random_state=seed)
y_pred = dba_km.fit_predict(inputd)
model = dba_km.fit(inputd)
model.to_json('dba_km.json')

for yi in range(2):
    plt.subplot(3, 3, 4 + yi)
    for xx in inputd[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(dba_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, 15)
    plt.ylim(0, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    plt.title("DBA $k$-means")

sdtw_km = TimeSeriesKMeans(n_clusters=2,
                           metric="softdtw",
                           metric_params={"gamma": .01},
                           verbose=True,
                           random_state=seed)
y_pred = sdtw_km.fit_predict(inputd)
model = sdtw_km.fit(inputd)
model.to_json('sdtw_km.json')

for yi in range(2):
    plt.subplot(3, 3, 7 + yi)
    for xx in inputd[y_pred == yi]:
        plt.plot(xx.ravel(), "k-", alpha=.2)
    plt.plot(sdtw_km.cluster_centers_[yi].ravel(), "r-")
    plt.xlim(0, 15)
    plt.ylim(0, 4)
    plt.text(0.55, 0.85,'Cluster %d' % (yi + 1),
             transform=plt.gca().transAxes)
    plt.title("Soft-DTW $k$-means")

plt.tight_layout()
plt.show()