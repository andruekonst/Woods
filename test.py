import numpy as np
import woods

a = np.arange(10)
print(a)
print("Woods mean:", woods.mean(a.astype(np.double)))
print("NumPy mean:", np.mean(a))
print("Done")
# print(woods.mean())