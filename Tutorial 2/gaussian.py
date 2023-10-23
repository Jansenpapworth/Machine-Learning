### function to calculate the gaussian function
import numpy as np
import matplotlib.pyplot as plt

def gaussian(x,u,sig):
    g = (2*np.pi)**(-1/2)*1/sig*np.exp((-((x-u)/sig)**2)/2)
    return g