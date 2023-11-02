import pandas
import numpy as np
import matplotlib.pyplot as plt

df = pandas.read_csv('http://pogo.software/me4ml/tensile_strength.csv')

print(df)

# extract the frequency values (and scale since they are MHz)
temp = np.array(df['Temperature (deg C)'][:])
UTS = np.array(df['Ultimate tensile strength (Pa)'][:])

# temp std and mean
t_mean = np.mean(temp)
t_std = np.std(temp)

# UTS std and mean
s_mean = np.mean(UTS)
s_std = np.std(UTS)

# scaling the data
t_scale = (temp-t_mean)/t_std
s_scale = (UTS-s_mean)/s_std

# saving the scaling parameters
scArray = np.array([[t_mean, s_mean],[t_std, s_std]])
np.savetxt('scaleParams.txt',scArray)

# plotting the histogram
fig, ax = plt.subplots()
plt.hist(s_scale)
plt.show()
