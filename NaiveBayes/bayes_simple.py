import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB

x1 = np.genfromtxt("class1.csv", delimiter = ",")
x2 = np.genfromtxt("class2.csv", delimiter = ",")

y1 = np.zeros(x1.shape[0])
y2 = np.ones(x2.shape[0])

x = np.concatenate((x1, x2), axis = 0)
y = np.concatenate((y1, y2))

xmin, xmax = x[:, 0].min() - 0.1, x[:, 0].max() + 0.1
ymin, ymax = x[:, 1].min() - 0.1, x[:, 1].max() + 0.1


gnb = GaussianNB()
y_pred = gnb.fit(x, y).predict(x)
print "Number of mislabeled points: %d" % (y != y_pred).sum()

xx, yy = np.meshgrid(np.arange(xmin, xmax, 0.01), np.arange(ymin, ymax, 0.01))
xnew = np.c_[xx.ravel(), yy.ravel()]
ynew = gnb.fit(x, y).predict(xnew).reshape(xx.shape)

fig = plt.figure(1)
#plt.set_cmap(plt.cm.Paired)
plt.pcolormesh(xx, yy, ynew)
plt.plot(x1[:, 0], x1[:, 1], 'ob', x2[:, 0], x2[:, 1], 'or')
plt.savefig("bayes_simple.png")
