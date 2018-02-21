import numpy as np
from numpy import linalg as LA

w = np.array([0, 0, 0]).T
# print(w)
X = np.array([[1, 0.3], [0.4, 0.5], [0.7, 0.8]])
X = np.hstack((np.ones(X.shape[0]).reshape(X.shape[0], 1), X))
# print(X)
Y = np.array([[1, 1, 0]]).T

i = 0
for i in range(0, X.shape[0]):
    x = X[i, :]
    y = 1 if w.dot(x) > 0 else 0
    w = w + (Y[i] - y) * x.T
    i += 1
print(w)


