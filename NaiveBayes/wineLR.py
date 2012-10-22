import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

wine = np.genfromtxt("wine.data",
                     delimiter = ",",
                     dtype = float
                    )

X = wine[:, 1:14]
Y = wine[:, 0]

regr = linear_model.LinearRegression()
regr.fit(X, Y)

y_pred = regr.fit(X, Y).predict(X).astype(np.int64)
print y_pred
print "Number of mislabeled points: %d" % (Y != y_pred).sum()
