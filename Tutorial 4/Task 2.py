import numpy as np
import matplotlib.pyplot as plt

w = (np.array([[-1,-3]]))
w_0 = 1

# function to generate the grid 
def gen_sample_grid(npx=200, npy=200, limit = 1):
  x1line = np.linspace(0, limit, npx)
  x2line = np.linspace(0, limit, npy)
  x1grid, x2grid = np.meshgrid(x1line, x2line)
  Xgrid = np.array([x1grid, x2grid]).reshape([2,npx*npy]).T
  return Xgrid, x1line, x2line

Xgrid , x1line, x2line = gen_sample_grid()

g_x = np.matmul(Xgrid,w.T) + w_0

g_x = np.reshape(g_x, [200,200])

# plotting the line corresponding with g_x = 0
M = -1/3
C = 1/3
y_l = M*x1line+C

plt.contourf(x1line,x2line,g_x)
plt.plot(x1line,y_l)
plt.colorbar()

plt.show()