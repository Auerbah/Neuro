import numpy as np

# x1_shape = (2, 2)
#
# X1 = np.ones(x1_shape)
#
# print(X1)
#
# print(x1_shape)
#
# print(x1_shape[1])


x_shape = tuple(map(int, input().split()))
X = np.fromiter(map(int, input().split()), np.int).reshape(x_shape)

y_shape = tuple(map(int, input().split()))
Y = np.fromiter(map(int, input().split()), np.int).reshape(y_shape)

if x_shape[1] == y_shape[1]:
    print(X.dot(Y.T))
else:
    print("matrix shapes do not match")
