import numpy as np
import matplotlib.pyplot as plt

def get_cov(sdx=1, sdy=1, rotandeg=0):
  covar = np.array([[sdx**2,0], [0, sdy**2]])
  print(covar)
  rot_ang = rotandeg /360 * 2 * np.pi
  rot_mat = np.array([[np.cos(rotandeg), -np.sin(rotandeg)],
                      [np.sin(rotandeg),  np.cos(rotandeg)]])
  covar = np.matmul(np.matmul(rot_mat, covar), rot_mat.T)
  return covar

print(get_cov())
def gen_sample_grid(npx=200, npy=200, limit = 1):
  x1line = np.linspace(-limit, limit, npx)
  x2line = np.linspace(-limit, limit, npy)
  x1grid, x2grid = np.meshgrid(x1line, x2line)
  Xgrid = np.array([x1grid, x2grid]).reshape([2,npx*npy]).T
  return Xgrid, x1line, x2line