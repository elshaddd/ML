# Load dataset
from sklearn.datasets import load_iris
iris = load_iris(); X = iris.data; y = iris.target
# Load model
from sklearn.ensemble import RandomForestClassifier
rfm = RandomForestClassifier(n_estimators=100)
# 1. Hold -out cross -validation
from sklearn.model_selection import train_test_split
X_train , X_test , y_train , y_test = train_test_split(X, y, test_size=0.3, random_state=1)
rfm.fit(X_train ,y_train)
# 2. K-fold cross -validation
from sklearn.model_selection import cross_val_score , KFold
kf = KFold(n_splits=5)
score = cross_val_score(rfm, X, y, cv=kf)
# 3. Stratified k-fold cross -validation
from sklearn.model_selection import StratifiedKFold
skf = StratifiedKFold(n_splits = 3)
score = cross_val_score(rfm, X, y, cv=skf)
# 4. Leave -One-Out Cross -Validation
from sklearn.model_selection import LeaveOneOut
LOOCV=LeaveOneOut()
score = cross_val_score(rfm, X, y, cv=LOOCV)
