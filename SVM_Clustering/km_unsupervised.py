import numpy as np
from sklearn import cluster

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

clf = cluster.KMeans(n_clusters = 3)

y_pred = clf.fit(x).predict(x)
print "Number of mislabeled points: %d" % (y != y_pred).sum()
