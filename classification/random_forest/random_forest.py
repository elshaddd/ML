# Connecting the necessary libraries
from sklearn import datasets
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load IRIS dataset. Take the first two features
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
# Fit the model (n_estimators - Number of trees)
clf = RandomForestClassifier(n_estimators=100)
# Train the model using the training sets
clf.fit(X_tr_std, y_tr)
y_pred = clf.predict(X_t_std)
