import numpy as np
from sklearn import datasets, linear_model
from sklearn.metrics import mean_squared_error

daibetic = datasets.load_diabetes()
X = daibetic.data[:, np.newaxis, 2]
X_train = X[:30]
X_test = X[:-20]
Y_train = daibetic.target[:30]
Y_test = daibetic.target[:-20]
model = linear_model.LinearRegression()
model.fit(X_train, Y_train)
Y_predict = model.predict(X_test)
print("Error is:", mean_squared_error(Y_test, Y_predict))
print("Weights ", model.coef_)
print("Intercept", model.intercept_)
