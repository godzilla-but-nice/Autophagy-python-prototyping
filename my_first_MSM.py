import pdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

# initial parameters: (states???)
t = np.linspace(0, 30, 100) # times in (s)
y0 = (0.0, 0.0) # starting conditions in (m) and (m/s)

# problem parameters:
parameters = {'f' : 1.0, # external force (N)
              'm' : 2.0, # mass in (kg)
              'k' : 1.0, # spring constant (N/m)
              'b' : 0.8, # damping constant
             }

# function to pass to odeint
def F(y, t, p = parameters):
    """
    Return derivatives for second order ODE y'' = y
    """
    dy = [0, 0]
    dy[0] = y[1]
    dy[1] = ((p['f'] / p['m']) -
            ((p['b'] / p['m']) * y[1]) -
            ((p['k'] / p['m']) * y[0]))

    return dy

# solve ODE
out = odeint(F, y0, t)

plt.figure()
plt.plot(t, out[:, 0], 'r')
plt.plot(t, out[:, 1], 'b')

plt.savefig('plot.png')
