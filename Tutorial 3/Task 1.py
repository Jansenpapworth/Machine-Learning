import pandas
import numpy as np
import matplotlib.pyplot as plt

df = pandas.read_csv('http://pogo.software/me4ml/xray.csv')

# assigning the data to x and y dimensions
x = np.array(df['Distance (mm)'][:])
y = np.array(df['Total absorption'][:])

# Finding the matrix coefficients
m = len(x)
x_sum = np.sum(x)
x_sum_square = np.sum(x**2)

y_x_sum = np.sum(x*y)
y_sum = np.sum(y)

# forming the matrixes
b = [[y_x_sum],[y_sum]]

A = [[x_sum,x_sum_square],
     [  m  ,    x_sum   ]]

# Solve the matrix
B = np.linalg.solve(A,b)

# find y values
x_reg = np.linspace(0,6,200)
y_linear_predict = B[0] + B[1]*x_reg
plt.plot(x_reg,y_linear_predict,color='black')

# adding quadratic regression line
x_sum_cubed = np.sum(x**3)
x_sum_4ed = np.sum(x**4)

x_square_y_sum = np.sum((x**2)*y)

A_2 = [[x_sum_square, x_sum_cubed, x_sum_4ed],
       [x_sum ,      x_sum_square, x_sum_cubed],
       [m     ,      x_sum       , x_sum_square]]

b_2 = [x_square_y_sum,y_x_sum,y_sum]

B_2 = np.linalg.solve(A_2,b_2)

# finding the y predicted values
y_quad_predict = B_2[0] + B_2[1]*x_reg + B_2[2]*x_reg**2

plt.plot(x_reg,y_quad_predict,color='red')

plt.scatter(x,y)
plt.title('X-ray Data')
plt.text(0.5,100,'Jansen Papworth',size=20,zorder=0.,color='grey')
plt.show()