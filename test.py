import numpy as np
import woods

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
greedyDecisionRule = woods.GreedyDecisionRule()
print("Woods:", greedyDecisionRule.fit(x, y, 0))
print("NumPy:", np.sum(x[:, 0] * x[:, 1]))

print("Done")
# print(woods.mean())