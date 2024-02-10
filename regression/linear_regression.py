# Connecting the necessary libraries
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn import datasets
# Load dataset. Take the first two features
diabetes = datasets.load_diabetes()
diabetes_X = diabetes.data[:, np.newaxis, 2]
# Create train and test split
X_train = diabetes_X[:-40]
X_test = diabetes_X[-40:]
y_train = diabetes.target[:-40]
y_test = diabetes.target[-40:]
# Fit the model
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
