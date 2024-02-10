# Import the necessary libraries
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
# Generate random data with 3 clusters
X, y_true = make_blobs(n_samples=300,
                       centers=3,
                       cluster_std=0.60,
                       random_state=0)
# Classification of data (three clusters)
kmeans = KMeans(n_clusters=3)
kmeans.fit(X)
# Determining the centroids
centers = kmeans.cluster_centers_
