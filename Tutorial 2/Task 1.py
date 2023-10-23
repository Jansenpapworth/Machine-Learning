import numpy as np
import matplotlib.pyplot as plt

from gaussian import gaussian

x = np.linspace(-10,20,200)

# calculating the distributions
p_1 = gaussian(x,2,1.5)
p_2 = gaussian(x,7,0.5)
p_3 = gaussian(x,8,2.5)
p_4 = gaussian(x,3.5,1)

p_x_w1 = p_1 + p_2
p_x_w2 = p_3 + p_4

p_x_w1 = p_x_w1/np.trapz(p_x_w1,x)
p_x_w2 = p_x_w2/np.trapz(p_x_w2,x)

fig, ax = plt.subplots()
plt.title('Likelihood')
plt.plot(x,p_x_w1)
plt.plot(x,p_x_w2)

# prior probabilities
p_prior1 = 0.5
P_prior2 = 0.5


# calculating the evidence terms
evidence_1 = (p_prior1*p_x_w1) + (p_prior1*p_x_w2)
evidence_2 = (P_prior2*p_x_w1) + (P_prior2*p_x_w2)

P_pos1 = (p_x_w1*0.5)/evidence_1
P_pos2 = (p_x_w2*0.5)/evidence_2

fig, ax = plt.subplots()
plt.title('Posterior')
plt.xlim([-3, 15])
plt.plot(x,P_pos1)
plt.plot(x,P_pos2)
plt.show()

