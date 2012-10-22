import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

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

gnb = GaussianNB()
y_pred = gnb.fit(x, y).predict(x)
print "Number of mislabeled points: %d" % (y != y_pred).sum()
print "mean accuracy: %f" % gnb.fit(x, y).score(x, y)
print y
print y_pred
