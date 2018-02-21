from pylab import *
from matplotlib.pylab import figure

x = linspace(-5, 5, 100)
y = log(1+exp(x))

plot(x, y, 'r')
xlabel('x')
ylabel('y')
title('softplus')
show()
