import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)

from sklearn import datasets

X, y = datasets.make_classification(n_informative=2,n_redundant=0,n_samples = 100, n_features = 2)

print(X,y)

X[:,0] = np.abs(X[:,0] * 0.5 + 5)
X[:,1] = np.abs(X[:,1] * 30 + 160)

fig, ax = plt.subplots()


ax.scatter(X[y == 0, 0], X[y == 0, 1])

ax.scatter(X[y == 1, 0], X[y == 1, 1])

x = np.linspace(3,7)
y = -260*x + 1400
ax.plot(x,y)
plt.show()