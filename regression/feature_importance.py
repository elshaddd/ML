from sklearn.linear_model import LinearRegression, ElasticNet
from yellowbrick.model_selection import FeatureImportances

housing = fetch_california_housing()
data = pd.DataFrame(housing.data, columns=housing.feature_names)
data['AveCostHouse'] = housing.target.tolist()


X = data[['MedInc', 'HouseAge', 'AveRooms', 'AveBedrms',
          'Population', 'AveOccup', 'Latitude', 'Longitude']]
viz = FeatureImportances(LinearRegression(), labels=classes)
viz.fit(X, housing.target)
viz.show()
viz = FeatureImportances(ElasticNet(alpha=1.0, l1_ratio=0.5), labels=classes)
viz.fit(X, housing.target)
viz.show()
