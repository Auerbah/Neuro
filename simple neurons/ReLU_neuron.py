from pylab import *
from matplotlib.pyplot import figure


def relu(x):
    return np.maximum(x, np.zeros(x.shape))


x = linspace(-5, 5, 100)
y = relu(x)

plot(x, y, 'r')
xlabel('x')
ylabel('y')
title('Rectified linear unit')
show()
