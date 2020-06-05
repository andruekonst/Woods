import numpy as np
import woods
from sklearn.datasets import make_regression, make_friedman2
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
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

X = x.copy()
y = np.arange(n).astype(np.double)
dt = woods.DecisionTree(depth=5, min_samples_split=2)
dt.fit(X, y)
print(y)
print("woods tree predictions:", dt.predict(X))
print(mean_squared_error(y, dt.predict(X)))

params = dict(depth=5,
              min_samples_split=2,
              n_estimators=100,
              learning_rate=0.1)
gbm = woods.GradientBoosting(**params)
gbm.fit(X, y)
print(y)
print("woods gbm predictions:", gbm.predict(X))
print(mean_squared_error(y, gbm.predict(X)))

n_experiments = 10
n_repeats = 10

X, y = make_friedman2(250, noise=1.0, random_state=0)
X = X.astype(np.double)
y = y.astype(np.double)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

gbm = woods.GradientBoosting(**params)

gbm_times = []
for j in range(n_experiments):
    start = time()
    for i in range(n_repeats):
        gbm.fit(X_train, y_train)
    end = time()
    gbm_times.append(end - start)
print(f"time: {np.mean(gbm_times)} +- {np.std(gbm_times)}")
print("woods gradient boosting predictions:") # , gbm.predict(X))
# print(np.mean(np.linalg.norm(y_test - gbm.predict(X_test))))
print(mean_squared_error(y_test, gbm.predict(X_test)))
print("woods gbm predict times:")
gbm_times = []
for j in range(n_experiments):
    start = time()
    for i in range(n_repeats):
        gbm.predict(X_train)
    end = time()
    gbm_times.append(end - start)
print(f"  {np.mean(gbm_times)} +- {np.std(gbm_times)}")