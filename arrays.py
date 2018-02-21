import numpy as np
a = np.array([[1,2],[3,4]])
print(a)
b = np.eye(3,4,1)
print(b)
c = np.zeros(shape=b.shape)
print(c)
aa = np.array([1, 2, 3, 4, 5])
print(aa)
print(aa[0])
print(aa[:])
print(aa[1:])
M = np.array([[1, 2], [3, 4]])
print(M[:, 1])
print(M[1, :])

mat = np.array([[2, 1, 0, 0], [0, 2, 1, 0], [0, 0, 2, 1]])
print(mat)
print(mat.reshape(12, 1))

import random
w = np.array(random.sample(range(1000), 12))  # одномерный массив из 12 случайных чисел от 1 до 1000
print(w)
w = w.reshape((3,4))  # превратим w в трёхмерную матрицу
print(w)

v = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
print(v)
print(v.mean(axis=0))  # вдоль столбцов
print(v.mean(axis=1))  # вдоль строк
print(v.mean(axis=None))  # вдоль всего массива
print(v.mean())
print(a.argmin())
print(a.argmax())
print(np.unravel_index(v.argmax(), v.shape))

M1 = np.arange(6).reshape((2,3))
print(M1)
print(M1.cumsum())


