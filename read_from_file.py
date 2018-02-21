import numpy as np
from urllib.request import urlopen

f = urlopen('https://stepic.org/media/attachments/lesson/16462/boston_houses.csv')
#
# print(f.read())

# sbux = np.loadtxt(f, skiprows=1, delimiter=",")


sbux = np.loadtxt(f, skiprows=1, delimiter=',')
sbux2 = sbux.mean(0)

print(sbux2)
# print(sbux.shape)
