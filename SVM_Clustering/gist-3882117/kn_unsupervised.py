import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

x1 = np.genfromtxt("class1.csv", delimiter = ",")
x2 = np.genfromtxt("class2.csv", delimiter = ",")
x3 = np.genfromtxt("class3.csv", delimiter = ",")

y1 = np.zerosimport numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import NearestNeighbors

x1 = np.genfromtxt("class1.csv", delimiter = ",")
x2 = np.genfromtxt("class2.csv", delimiter = ",")
x3 = np.genfromtxt("class3.csv", delimiter = ",")

y1 = np.zeros(x1.shape[0])
y2 = np.ones(x2.shape[0])
y3 = np.arange(x3.shape[0])
y3.fill(2)

x = np.concatenate((x1, x2, x3), axis = 0)
y = np.concatenate((y1, y2, y3))

xmin, xmax = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
ymin, ymax = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1

neigh = NearestNeighbors(n_neighbors = 11)
neigh.fit(x)

neighbors = neigh.kneighbors(x, return_distance = False)
print neighbors
(x1.shape[0])
y2 = np.ones(x2.shape[0])
y3 = np.arange(x3.shape[0])
y3.fill(2)

x = np.concatenate((x1, x2, x3), axis = 0)
y = np.concatenate((y1, y2, y3))

xmin, xmax = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
ymin, ymax = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1

neigh = NearestNeighbors(n_neighbors = 11)
neigh.fit(x)

neighbors = neigh.kneighbors(x, return_distance = False)
print neighbors
