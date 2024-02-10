# Connecting the necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
# Generate a non-linear dataset with a quadratic relationship
np.random.seed(42)
x = np.linspace(-10, 10, 100).reshape(-1, 1)
y = 0.5 * x ** 2 + np.random.normal(0, 20, size=100)
y = y.reshape(-1, 1)
# Fit linear regression model
linear_model = LinearRegression()
linear_model.fit(x, y)
y_pred_linear = linear_model.predict(x)
# Fit polynomial regression model (degree = 2)
poly_features = PolynomialFeatures(degree=2, include_bias=False)
x_poly = poly_features.fit_transform(x)
poly_model = LinearRegression()
poly_model.fit(x_poly, y)
y_pred_poly = poly_model.predict(x_poly)
