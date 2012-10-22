import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt

X = np.array([[0], [1], [2], [3]])
Y = np.array([0.1, 2.1, 3.8, 2.7])

regr = linear_model.LinearRegression()
regr.fit(X, Y)

print 'Coefficients: %f' % regr.coef_
print 'Residual sum of squares: %f' % np.mean((regr.predict(X) - Y) ** 2)
print 'Variance score %f' % regr.score(X, Y)

plt.scatter(X, Y)
plt.plot(X, regr.predict(X), color = 'pink', linewidth = 3)
plt.savefig("simple2.png")
