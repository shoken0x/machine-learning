import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

wine = np.genfromtxt("wine.data",
                     delimiter = ",",
                     dtype = float
                     )
x = wine[:, 1:14]
y = wine[:, 0]

gnb = GaussianNB()
y_pred = gnb.fit(x, y).predict(x)
print "Number of mislabeled points: %d" % (y != y_pred).sum()
print "mean accuracy: %f" % gnb.fit(x, y).score(x, y)
#print y
#print y_pred
