import numpy as np
from sklearn import svm

wine = np.genfromtxt("wine.data",
                     delimiter = ",",
                     dtype = float
                    )   
x = wine[:, 1:14]
y = wine[:, 0]

clf = svm.SVC(kernel='linear',C=1)
y_pred = clf.fit(x, y).predict(x)
print "Number of mislabeled points: %d" % (y != y_pred).sum()

