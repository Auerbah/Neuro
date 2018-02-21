import numpy as np


class Perceptron:

    def __init__(self, w, b):
        """
        Инициализируем наш объект - перцептрон.
        w - вектор весов размера (m, 1), где m - количество переменных
        b - число
        """

        self.w = w
        self.b = b

    def forward_pass(self, single_input):
        """
        Метод рассчитывает ответ перцептрона при предъявлении одного примера
        single_input - вектор примера размера (m, 1).
        Метод возвращает число (0 или 1) или boolean (True/False)
        """

        result = 0
        for i in range(0, len(self.w)):
            result += self.w[i] * single_input[i]
        result += self.b

        return result > 0

    def vectorized_forward_pass(self, input_matrix):
        """
        Метод рассчитывает ответ перцептрона при предъявлении набора примеров
        input_matrix - матрица примеров размера (n, m), каждая строка - отдельный пример,
        n - количество примеров, m - количество переменных
        Возвращает вертикальный вектор размера (n, 1) с ответами перцептрона
        (элементы вектора - boolean или целые числа (0 или 1))
        """

        result = input_matrix.dot(self.w) + self.b
        return result > 0

    def train_on_single_example(self, example, y):
        """
        принимает вектор активации входов example формы (m, 1)
        и правильный ответ для него (число 0 или 1 или boolean),
        обновляет значения весов перцептрона в соответствии с этим примером
        и возвращает размер ошибки, которая случилась на этом примере до изменения весов (0 или 1)
        (на её основании мы потом построим интересный график)
        """

        result = 1 if self.w.T.dot(example) + self.b > 0 else 0
        self.w = self.w + (y - result) * example
        self.b = self.b + (y - result)
        return abs(result - y)


w = np.array([0, 0]).T
p = Perceptron(w, 0)
X = np.array([[1, 0.3], [0.4, 0.5], [0.7, 0.8]])
print(p.forward_pass(X[0, :].T))
print(p.vectorized_forward_pass(X))
print(p.train_on_single_example(X[0, :].T, 1))
print(p.train_on_single_example(X[1, :].T, 1))
print(p.train_on_single_example(X[2, :].T, 0))
print(p.b, p.w)

"""
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
"""