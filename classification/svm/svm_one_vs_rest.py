# Connecting the necessary libraries
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Creating a data set
iris = datasets.load_iris()
X = iris.data
y = iris.target
# Create train and test split
X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=0.3,
                                        random_state=42, stratify=y)
# Feature Scaling using StandardScaler
sc = StandardScaler().fit(X_tr)
X_tr_std = sc.transform(X_tr)
X_t_std = sc.transform(X_t)
# Fit the model and define strategy and fit model
ovr = OneVsRestClassifier(SVC()).fit(X_tr_std, y_tr)
# make predictions
y_ovr = ovr.predict(X_t_std)
