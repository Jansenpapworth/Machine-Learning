import pandas
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures

df = pandas.read_csv('http://pogo.software/me4ml/hdpeVel.csv')

# set the index column as the one containing the temperatures values
df = df.set_index('T/C f/MHz')

# extract the frequency values (and scale since they are MHz)
freq = df.columns.values.astype(np.float16)*1e6
# extract the temperature values
temp = df.index.values.astype(np.float16)
# extract the main part - the velocity values
vel = df.to_numpy()

# calculate the total number of values
tot_values = len(freq)*len(temp)

# forming the data in a grid with output value for velocity for every combination
x1grid, x2grid = np.meshgrid(freq, temp) 
Xgrid = np.concatenate([x1grid.reshape([tot_values, 1]), 
	x2grid.reshape([tot_values, 1])], axis=1) 
ygrid = vel.reshape([tot_values, 1])

reg = LinearRegression()
# uses training data X_poly to fit a function ygrid
reg.fit(Xgrid, ygrid)
y_lin = reg.predict(Xgrid)

# plotting both the training data and the fitted curve on a scatter graph
fig = plt.figure() 
ax = fig.add_subplot(111, projection='3d') 
ax.scatter(Xgrid[:, 0], Xgrid[:, 1], ygrid, marker='x', color='#000000') 
ax.scatter(Xgrid[:, 0], Xgrid[:, 1], y_lin, marker='o', color='#ff0000')
ax.set_xlabel('Frequency MHz')
ax.set_ylabel('Temperature C')
ax.set_zlabel('Wave Speed')

poly = PolynomialFeatures(degree = 2)
# generate teh new feature vector curve
X_poly = poly.fit_transform(Xgrid)

print(X_poly.shape)
print(poly.powers_)


###### need to finish this up as this is not working properly
reg_poly = LinearRegression()
reg_poly.fit(X_poly,ygrid)
y_poly = reg_poly.predict(Xgrid)

ax.scatter(Xgrid[:,0], Xgrid[:,1],y_poly, marker = '^', color = '#00ff00') 

plt.show()