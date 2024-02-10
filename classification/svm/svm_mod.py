from sklearn import svm
from sklearn.datasets import make_gaussian_quantiles
from sklearn.datasets import make_circles
X1, y1 = make_circles(n_samples=500, noise=0.06, random_state=42)
X2, y2 = make_gaussian_quantiles(n_features=2, n_classes=2,
                                 n_samples=1000, mean=(2, 3))
# Используем линейное ядро
linear_svc = svm.SVC(kernel='linear').fit(X, y)
# Используем RBF-ядро
rbf_svc = svm.SVC(kernel='rbf', gamma=0.7, C=1).fit(X, y)
# Используем полиномиальное ядро, степени 2
poly_svc = svm.SVC(kernel='poly', degree=2, C=1).fit(X, y)
