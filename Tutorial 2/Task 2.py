from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
import matplotlib.pyplot as plt
import numpy as np

np.random.seed(5)

X , y = datasets.make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0)

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.33)

print(y_train)

clf = GaussianNB()
clf.fit(X_train, y_train)

# function to generate a grid 
def gen_sample_grid(npx, npy, limit):
  x1line = np.linspace(-limit, limit, npx)
  x2line = np.linspace(-limit, limit, npy)
  x1grid, x2grid = np.meshgrid(x1line, x2line)
  Xgrid = np.array([x1grid, x2grid]).reshape([2,npx*npy]).T
  return Xgrid, x1line, x2line


Xgrid, x1line, x2line = gen_sample_grid(200,200,3)

classVals = clf.predict(Xgrid)
classVals = np.reshape(classVals, [200,200])

fig, ax = plt.subplots()
plt.contourf(x1line,x2line,classVals)
ax.scatter(X_train[y_train == 0, 0], X_train[y_train == 0, 1])
ax.scatter(X_train[y_train == 1, 0], X_train[y_train == 1, 1])

plt.xlim(-3,3)
plt.ylim(-3,3)


y_test_model = clf.predict(X_test)

ntot = len(y_test)
nMatch = 0
for i in range(0,ntot):
  if y_test[i] == y_test_model[i]:
    nMatch = nMatch + 1

print(np.round(nMatch/ntot*100,2))
plt.show()