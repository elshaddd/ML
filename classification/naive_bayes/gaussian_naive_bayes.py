# Connecting the necessary libraries
from sklearn import datasets
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
# Load IRIS dataset. Take the first two features
iris = datasets.load_iris()
X = iris.data[:, [0, 1]]
y = iris.target
# Create train and test split
X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=0.3,
                                        random_state=42, stratify=y)
# Fit the model
gnb = GaussianNB()
y_pred = gnb.fit(X_tr, y_tr).predict(X_t)
