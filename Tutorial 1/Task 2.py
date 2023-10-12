import numpy as np
import matplotlib.pyplot as plt

# function to return a covariance matrix rotated to a certain angle
def get_cov(sdx, sdy, rotandeg):
  covar = np.array([[sdx**2,0], [0, sdy**2]])
  rot_ang = rotandeg /360 * 2 * np.pi
  # defining the matrix rotation 
  rot_mat = np.array([
                [np.cos(rotandeg), -np.sin(rotandeg)],
                [np.sin(rotandeg),  np.cos(rotandeg)]
            ])
  covar = np.matmul(np.matmul(rot_mat, covar), rot_mat.T)
  return covar

# function to genrate a grid 
def gen_sample_grid(npx=200, npy=200, limit = 1):
  x1line = np.linspace(-limit, limit, npx)
  x2line = np.linspace(-limit, limit, npy)
  x1grid, x2grid = np.meshgrid(x1line, x2line)
  Xgrid = np.array([x1grid, x2grid]).reshape([2,npx*npy]).T
  return Xgrid, x1line, x2line

Xgrid, x1line, x2line = gen_sample_grid()
covar = get_cov(1,0.3,30)

p = 1 / (2 * np.pi * np.sqrt(np.linalg.det(covar))) * np.exp(
    -1 / 2 * (np.matmul(Xgrid, np.linalg.inv(covar)) * Xgrid).sum(-1))

pgrid = np.reshape(p, [200,200])
plt.contourf(x1line, x2line, pgrid)
#plt.scatter(x1line,x2line)
#plt.show()

distvals = np.random.multivariate_normal([0,0],covar,100)

fig, ax = plt.subplots()

ax.scatter(distvals[:, 0], distvals[:, 1])
#ax.scatter(distvals[:, 0], distvals[:, 1], s = 1) # to compare with distribution
plt.ylim(-1, 1)
plt.xlim(-1, 1)

plt.show()