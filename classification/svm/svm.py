# Connecting the necessary libraries
import numpy as np
from sklearn import svm
# Creating a data set
np.random.seed(0)
X = np.r_[np.random.randn(20, 2) - [2, 2],
          np.random.randn(20, 2) + [2, 2]]
Y = [0] * 20 + [1] * 20
# Fit the model
clf = svm.SVC(kernel='linear')
clf.fit(X, Y)
# Draw the separating hypoplane
w = clf.coef_[0]
a = -w[0] / w[1]
xx = np.linspace(-5, 5)
yy = a * xx - (clf.intercept_[0]) / w[1]
# Construct lines through the reference vectors
b = clf.support_vectors_[0]
yy_down = a * xx + (b[1] - a * b[0])
b = clf.support_vectors_[-1]
yy_up = a * xx + (b[1] - a * b[0])
