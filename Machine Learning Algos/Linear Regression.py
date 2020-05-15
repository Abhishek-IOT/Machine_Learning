import numpy as np
from sklearn import datasets

daibetic = datasets.load_diabetes()
X = daibetic.data[:, np.newaxis, 2]
X_train = X[:30]
X_test = X[:-20]
