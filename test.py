import numpy as np
import woods
from sklearn.datasets import make_regression

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
greedydecisionrule = woods.GreedyDecisionRule()
greedydecisionrule.fit(x, y, 0)
print("woods:", greedydecisionrule.get_split())
print("woods predictions:", greedydecisionrule.predict(x))

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

X, y = make_regression(random_state=0)
X = X.astype(np.double)
y = y.astype(np.double)
print("MEAN")
print(np.mean(y))
rdt = woods.RandomizedDecisionTree()
rdt.set_depth(5)
rdt.fit(X, y, 0)
print("woods randomized decision tree predictions:") # , rdt.predict(X))
print(np.mean(np.linalg.norm(y - rdt.predict(X))))

gbm = woods.RandomizedGradientBoosting()
gbm.set_depth(5)
gbm.set_iterations(100)
gbm.set_learning_rate(0.1)
gbm.fit(X, y, 0)
print("woods gradient boosting predictions:") # , gbm.predict(X))
print(np.mean(np.linalg.norm(y - gbm.predict(X))))

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
# est = GradientBoostingRegressor()
est = DecisionTreeRegressor(max_depth=5, splitter="random", random_state=0)
est.fit(X, y)
print(np.mean(np.linalg.norm(y - est.predict(X))))

print("Done")
# print(woods.mean())