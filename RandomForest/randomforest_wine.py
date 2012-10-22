import numpy as np
from sklearn.ensemble import RandomForestClassifier

wine = np.genfromtxt("wine.data",
                     delimiter = ",",
                     dtype = float
                    )
x = wine[:, 1:14]
y = wine[:, 0]


for d in range(1,6):
  clf = RandomForestClassifier(max_depth = 3)
  y_pred = clf.fit(x, y).predict(x)
  print "max_depth = %d" % d
  print "Number of mislabeled points: %d" % (y != y_pred).sum()
  print "mean accuracy: %f" % clf.fit(x, y).score(x, y) 
