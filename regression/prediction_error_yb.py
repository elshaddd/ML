from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNetCV
from yellowbrick.regressor import PredictionError
model = LinearRegression()
viz = PredictionError(model)
viz.fit(X_train, y_train)
viz.score(X_test, y_test)
viz.show()
