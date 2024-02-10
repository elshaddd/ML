from sklearn.tree import DecisionTreeClassifier
from yellowbrick.classifier import ClassificationReport
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
# Load IRIS dataset. Take the first two features
iris = datasets.load_iris()
X = iris.data[:, [0, 1]]
y = iris.target
# Create train and test split
X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=0.3,
                                        random_state=42, stratify=y)
#########
classes = ["setosa", "virginica", "versicolor"]
visualizer = ClassificationReport(
    DecisionTreeClassifier(), classes=classes, support=False)
visualizer.fit(X_tr, y_tr)
visualizer.score(X_t, y_t)
visualizer.show()
