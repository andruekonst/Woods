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
dt.save("tmp_models/test.json")

loaded_dt = woods.DecisionTree()
loaded_dt.load("tmp_models/test.json")
print("woods loaded tree predictions:", loaded_dt.predict(X))
print(mean_squared_error(y, loaded_dt.predict(X)))

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


class WoodsGBMRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, n_estimators=100, max_depth=3, min_samples_split=2, learning_rate=0.1):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.learning_rate = learning_rate

    def fit(self, X, y):
        params = dict(
            depth=self.max_depth,
            min_samples_split=self.min_samples_split,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
        )
        self.est_ = woods.GradientBoosting(**params)
        self.est_.fit(X, y)

    def predict(self, X):
        return self.est_.predict(X)

grid = {
    'n_estimators': [100, 1000],
    'max_depth': [2, 3, 4, 5, 7, 8, 9],
    'learning_rate': [0.1, 0.01]
}

model = GridSearchCV(WoodsGBMRegressor(), grid,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1)
model.fit(X_train, y_train)
print("best woods randomized gradient boosting predictions:")
print(mean_squared_error(y_test, model.predict(X_test)))
print(f"best parameters: {model.best_params_}")

est = GradientBoostingRegressor(max_depth=params["depth"], n_estimators=params["n_estimators"],
                                learning_rate=params["learning_rate"])
est_times = []
for j in range(n_experiments):
    start = time()
    for i in range(n_repeats):
        est.fit(X_train, y_train)
    end = time()
    est_times.append(end - start)
print(f"time: {np.mean(est_times)} +- {np.std(est_times)}")
print("sklearn gradient boosting predictions:")
print(mean_squared_error(y_test, est.predict(X_test)))
model = GridSearchCV(est, grid,
                        scoring='neg_mean_squared_error',
                        n_jobs=-1)
model.fit(X_train, y_train)
print("best sklearn gradient boosting predictions:")
print(mean_squared_error(y_test, model.predict(X_test)))
print(f"best parameters: {model.best_params_}")

print("Done")