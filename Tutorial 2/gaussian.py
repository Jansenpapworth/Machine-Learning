### function to calculate the gaussian function
import numpy as np

def gaussian(x,u,sig):
    g = (2*np.pi)**(-1/2)*1/sig*np.exp((-((x-u)/sig)**2)/2)
    return g

print(gaussian(1,0,0.5))