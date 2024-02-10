from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from yellowbrick.cluster import KElbowVisualizer
X, y = make_blobs(n_samples=1000, n_features=12, centers=8,
                  shuffle=True, random_state=42)
model = KMeans()

visualizer = KElbowVisualizer(model, k=(4, 12))
visualizer.fit(X)
visualizer.show()

visualizer = KElbowVisualizer(model, k=(4, 12), metric='silhouette')
visualizer.fit(X)
visualizer.show()

visualizer = KElbowVisualizer(model, k=(4, 12), metric='calinski_harabasz')
visualizer.fit(X)
visualizer.show()
