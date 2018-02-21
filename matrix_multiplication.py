import numpy as np

"""x = (1, 3)"""
x = np.array([[1, 2, 3]])
w = np.array([[3, 2, 1]])
b = 1.23
print(x)
print(x.shape)
print(w.dot(x.T) + b)

"""x = (3, 1)"""
x = np.array([[1], [2], [3]])
w = np.array([[3], [2], [1]])
b = 1.23
print(x)
print(x.shape)
print(w.T.dot(x) + b)

"""x = (3, )"""
x = np.array([1, 2, 3])
w = np.array([3, 2, 1])
b = 1.23
print(x)
print(x.shape)
print(w.T.dot(x) + b)
print(w.dot(x.T) + b)
print(np.dot(x, w) + b)
