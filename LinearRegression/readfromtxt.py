import numpy as np
import matplotlib.pyplot as plt
from sklearn import naive_bayes

x = np.genfromtxt("source.csv", delimiter=",")

plt.scatter(x[:,0], x[:,1],s=100)
plt.savefig("readfromtxt.png")
