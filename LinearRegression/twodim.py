import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

array = np.array([[0, 1.1], [1, 3.1], [2, 4.8], [3, 2.7]])
print array

X = array[:, 0:1]
Y = array[:, 1]

print X
print Y

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print 'Coefficients: %f' % regr.coef_
print 'Residual sum of squares: %f' % np.mean((regr.predict(X) - Y) ** 2)
print 'Variance score %f' % regr.score(X, Y)

plt.scatter(X, Y)
plt.plot(X, regr.predict(X), color = 'pink', linewidth = 3)
plt.savefig("twodim.png")
