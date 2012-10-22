import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

array = np.genfromtxt("source.csv", delimiter=",")

X = array[:, 0:1]
Y = array[:, 1]

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print 'Coefficients: %f' % regr.coef_
print 'Residual sum of squares: %f' % np.mean((regr.predict(X) - Y) ** 2)
print 'Variance score %f' % regr.score(X, Y)

plt.scatter(X, Y, c=X, s=35)
plt.colorbar()

plt.plot(X, regr.predict(X), color = 'red', linewidth = 2)

plt.savefig("readfromtxt2.png")
