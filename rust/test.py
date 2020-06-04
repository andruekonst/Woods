import numpy as np
import woods
from sklearn.datasets import make_regression, make_friedman2
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, RegressorMixin
from time import time


n = 10
x = np.arange(n * 4).reshape((n, -1)).astype(np.double)
y = np.arange(n).astype(np.double)
x[:, [0, 2]] = 0.0
x[:2, 1] = 1000.0
y[:n // 2] = 0.0
y[n // 2:] = 1.0
print(x)
print(y)
print(f"x shape: {x.shape}\ny shape: {y.shape}")
# print("Woods:", woods.fit(x, y))
dr = woods.DecisionRule()
dr.fit(x, y)
# print("woods:", dr.get_split())
print("woods predictions:", dr.predict(x))
