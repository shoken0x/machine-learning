import numpy as np
from sklearn.neighbors import KNeighborsClassifier

def iris_index(s):
    names = {'Iris-setosa':1.0,
             'Iris-versicolor':2.0,
             'Iris-virginica':3.0}
    if names.has_key(s): return names[s]
    else: return -1.0

iris = np.genfromtxt("iris.data",
                     delimiter = ",",
                     dtype = float,
                     converters = {4: iris_index})
x = iris[:, 0:4]
y = iris[:, 4]

#clf = KNeighborsClassifier(n_neighbors = 9)
#y_pred = clf.fit(x, y).predict(x)
#print "Number of mislabeled points: %d" % (y != y_pred).sum()

for n in range(1,10):
  clf = KNeighborsClassifier(n_neighbors = n)
  y_pred = clf.fit(x, y).predict(x)
  print "n_neighbors = %d" % n
  print "Number of mislabeled points: %d" % (y != y_pred).sum()

