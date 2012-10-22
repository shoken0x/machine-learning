import numpy as np
from sklearn import cluster

def hclust(n_clusters, arrests, states):
    ward = cluster.Ward(n_clusters = n_clusters)
    pred =  ward.fit_predict(arrests)
    for i in range(0, len(arrests)):
        print states[i], pred[i]

arrests = np.genfromtxt(fname = "usarrests.csv",
                        delimiter = ",",
                        skip_header = 1,
                        usecols = (1, 2, 3, 4))

states =  np.genfromtxt(fname = "usarrests.csv",
                        dtype = str,
                        delimiter = ",",
                        skip_header = 1,
                        usecols = (0))

for i in range(2, 5):
    print "----------n_clusters=%d-----------", i
    hclust(i, arrests, states)