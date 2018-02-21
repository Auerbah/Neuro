from pylab import *
from matplotlib.pyplot import figure

x = linspace(-10, 10, 100)
y1 = 1 / (1 + exp(-x))
figure()
plot(x, y1, 'r')
xlabel('x')
ylabel('y')
title('Sigmoid')
show()

"""(exp(x)-exp(-x))/(exp(x)+exp(-x))"""
y2 = tanh(x)
figure()
plot(x, y2, 'b')
xlabel('x')
ylabel('y')
title('Hyperbolic tan')
show()

plot(x, y1, 'r')
plot(x, y2, 'b')
xlabel('x')
ylabel('y')
title('Hyperbolic tan and sigmoid')
show()


