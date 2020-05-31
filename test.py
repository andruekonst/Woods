import numpy as np
import woods
from sklearn.datasets import make_regression, make_friedman1
from time import time

a = np.arange(10)
print(a)
print("Woods mean:", woods.mean(a.astype(np.double)))
print("NumPy mean:", np.mean(a))

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
dr = woods.RandomDecisionRule()
dr.fit(x, y, 0)
print("woods:", dr.get_split())
print("woods predictions:", dr.predict(x))

# X, y = make_regression(random_state=0)
# X = X.astype(np.double)
# y = y.astype(np.double)
X = x.copy()
y = np.arange(n).astype(np.double)
dt = woods.RandomizedDecisionTree()
dt.set_depth(4)
dt.fit(X, y, 0)
print(y)
print("woods tree predictions:", dt.predict(X))
print(np.mean(np.linalg.norm(y - dt.predict(X))))

from sklearn.model_selection import train_test_split
X, y = make_friedman1(275, noise=1.0, random_state=0)
X = X.astype(np.double)
y = y.astype(np.double)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)

rdt = woods.RandomizedDecisionTree()
rdt.set_depth(5)
rdt.fit(X_train, y_train, 0)
print("woods randomized decision tree predictions:") # , rdt.predict(X))
print(np.mean(np.linalg.norm(y_test - rdt.predict(X_test))))

gbm = woods.RandomizedGradientBoosting()
gbm.set_depth(3)
gbm.set_iterations(100)
gbm.set_learning_rate(0.2)
start = time()
for i in range(100):
    gbm.fit(X_train, y_train, 0)
end = time()
print(f"time: {end - start}")
print("woods gradient boosting predictions:") # , gbm.predict(X))
print(np.mean(np.linalg.norm(y_test - gbm.predict(X_test))))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
est = GradientBoostingRegressor(max_depth=3)
# est = DecisionTreeRegressor(max_depth=5, splitter="random", random_state=0)
start = time()
for i in range(100):
    est.fit(X_train, y_train)
end = time()
print(f"time: {end - start}")
print("sklearn gradient boosting predictions:")
print(np.mean(np.linalg.norm(y_test - est.predict(X_test))))

print("Done")
# print(woods.mean())