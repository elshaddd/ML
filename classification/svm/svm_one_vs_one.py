# Connecting the necessary libraries
from sklearn.svm import SVC
from sklearn.multiclass import OneVsOneClassifier
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
# define strategy and fit model
ovo = OneVsOneClassifier(SVC()).fit(X_tr_std, y_tr)
# make predictions
y_ovo = ovo.predict(X_t_std)
