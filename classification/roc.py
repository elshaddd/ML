# Connecting the necessary libraries
from sklearn.metrics import roc_curve, auc
from sklearn.datasets import make_classification
from sklearn.naive_bayes import GaussianNB
X, y = make_classification(n_samples=500, random_state=0)
# Fit the model
model = GaussianNB()
model.fit(X, y)
y_score = model.predict_proba(X)[:, 1]
# Calculation fpr = False Positive Rate , tpr = True Positive Rate and AUC
fpr, tpr, thresholds = roc_curve(y, y_score)
AUC = auc(fpr, tpr)


from sklearn.ensemble import RandomForestClassifier
from yellowbrick.classifier import ROCAUC
from yellowbrick.datasets import load_spam
# Load dataset
X, y = load_spam()
X_tr, X_t, y_tr, y_t = train_test_split(X, y, test_size=0.3,
                                        random_state=42, stratify=y)
classes = ["not_spam", "is_spam"]
model = RandomForestClassifier(random_state=42, n_estimators=20)
visualizer = ROCAUC(model , classes=classes , micro=False , macro=False)
visualizer.fit(X_tr , y_tr)
visualizer.score(X_t , y_t)
visualizer.show()