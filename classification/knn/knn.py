# Connecting the necessary libraries
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.model_selection import train_test_split
# Load IRIS dataset
iris = datasets.load_iris()
X = iris.data[:, [0, 1]]
y = iris.target
# Create train and test split
X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=0.3,
                                        random_state=42, stratify=y)
# Feature Scaling using StandardScaler
sc = StandardScaler()
sc.fit(X_tr)
X_tr_std = sc.transform(X_tr)
X_t_std = sc.transform(X_t)
# Fit the model
knn = KNeighborsClassifier(n_neighbors=5, p=2, weights='uniform',
                           algorithm='auto').fit(X_tr_std, y_tr)
predicted = knn.predict(X_t_std)
