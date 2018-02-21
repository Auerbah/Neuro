# numpy.linalg
import numpy as np
from numpy import linalg as LA
a = np.arange(1, 5).reshape((2, 2))
print("a = ")
print(a)

print("a*a = ")
print(a.dot(a))

print("a^2 = ")
print(np.linalg.matrix_power(a, 2))

print("a^T = ")
print(a.T)

print("a^-1 = ")
print(np.linalg.inv(a))

print("a^-1 (pseudo) = ")
print(LA.pinv(a))


v = np.array([1, 2, 3])

print("v = ")
print(v)
print(LA.norm(v))
print(v/LA.norm(v))

