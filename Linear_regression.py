import numpy as np
from numpy import linalg as LA
from urllib.request import urlopen
'''
# X - матрица наблюдений n x m (n - кол-во наблюдений, m - число переменных
# b - вектор коэффициентов m x 1
# y - вектор значений целевой переменной n x 1
# e - вектор нормально распределенных ошибок n x 1

# b1 - вектор оценок коэффициентов m x 1
# y1 - вектор предсказаний целевой переменной n x 1
# e1 = y - y1 - вектор ошибок предсказания n x 1
'''

'''
1ое задание со stepic
'''

# M = np.array([[10, 60], [7, 50], [12, 75]])
# print(M)
# print(M[:, 0])
# print(M[:, 1:])
# y = M[:, 0]
# X = M[:, 1:]
# X = np.hstack((np.ones(X.shape[0]).reshape(X.shape[0], 1), X))
# print(X)
# y1 = LA.inv(X.T.dot(X)).dot(X.T).dot(y)
# print(y1)

'''
2ое задание со stepic
'''

fname = input()  # read file name from stdin
# fname = "https://stepic.org/media/attachments/lesson/16462/boston_houses.csv"
f = urlopen(fname)  # open file from URL
data = np.loadtxt(f, delimiter=',', skiprows=1)  # load data to work with

y = data[:, 0]
X = data[:, 1:]
X = np.hstack((np.ones(X.shape[0]).reshape(X.shape[0], 1), X))
# print(X)
y1 = LA.inv(X.T.dot(X)).dot(X.T).dot(y)

# print(y1)

y1 = y1.astype(np.str)
# print(y1)
str = " ".join(y1)
print(str)
# str = " ".join(y1)
# m = np.array(map(int, y1))


# np.hstack((array1, array2, ...))  # склеивает по строкам массивы, являющиеся компонентами кортежа, поданного на вход; массивы должны совпадать по всем измерениям, кроме второго
# np.ones_like(array)  # создаёт массив, состоящий из единиц, идентичный по форме массиву array
# "delim".join(array)  # возвращает строку, состоящую из элементов array, разделённых символами "delim"
# map(str, array)  # применяет функцию str к каждому элементу array