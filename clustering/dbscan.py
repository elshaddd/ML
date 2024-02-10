# Import of necessary libraries
from sklearn.datasets import make_moons
from sklearn.datasets import make_blobs
from sklearn.cluster import DBSCAN
import numpy as np
# Generate data
X, y = make_blobs(n_samples=300,
                  centers=[[0, 0], [0, 5], [5, 5]],
                  cluster_std=[1.0, 0.5, 1.0])
# Compute DBSCAN
dbscan = DBSCAN(eps=0.50, min_samples=5)
labels = dbscan.fit_predict(X)
# Definition noise points
no_noise = np.sum(np.array(labels) == -1, axis=0)


# Import of necessary libraries
# Generate data
n_samples = 200
X, _ = make_moons(n_samples=n_samples, noise=0.05, random_state=0)
# Compute DBSCAN
dbscan = DBSCAN(eps=0.3, min_samples=5)
labels = dbscan.fit_predict(X)
