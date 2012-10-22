import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree

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

for d in range(1,6):
  clf = tree.DecisionTreeClassifier(max_depth = d)
  y_pred = clf.fit(x, y).predict(x)
  print "max_depth = %d" % d
  print "Number of mislabeled points: %d" % (y != y_pred).sum()
  print "mean accuracy: %f" % clf.fit(x, y).score(x, y)

