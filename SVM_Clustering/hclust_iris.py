import numpy as np
from sklearn import cluster

def hclust(n_clusters, arrests, states):
    ward = cluster.Ward(n_clusters = n_clusters)
    pred = ward.fit_predict(arrests)
    for i in range(0, len(arrests)):
        print states[i], pred[i]

def iris_index(s):
    names = {'Iris-setosa':1.0,
             'Iris-versicolor':2.0,
             'Iris-virginica':3.0}
    if names.has_key(s): return names[s]
    else: return -1.0

x = np.genfromtxt(fname = "iris.data",
                  dtype = float,
                  delimiter = ",",
                  usecols = (1,2,3,4))
y = np.genfromtxt(fname = "iris.data",
                  dtype = str,
                  delimiter = ",",
                  usecols = (4))

print "----------n_clusters=%d-----------", 3 
hclust(3, x, y)

