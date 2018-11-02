import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends import _macosx
from scipy.integrate import odeint

# times and states:
t = np.linspace(0, 120, 500)
y0 = (0.0, # (m) starting position for mass 1
      0.0, # (m/s) starting velocity for mass 1
      0.0, # (m) starting position for mass 2
      0.0) # (m/s) starting velocity for mass 2

# problem parameters:
pars = {'m1' : 3.0, # (kg) mass 1
        'k1' : 1.0, # (N/m) spring 1
        'b1' : 1.5, # (Ns/m) damping 1
        'f1' : -10.0, # (N) external force on mass 1
        'm2' : 5.5, # (kg) mass 2
        'k2' : 1.0, # (N/m) spring 2
        'b2' : 1.7, # (Ns/m) damping 2
        'f2' : 5.0} # (N) external force on mass 2

# function to pass to odeint
def F(y, t, p = pars):
    """
    Calculate two differential equations describing a system with one mass
    connected to a fixed surface by a spring and damper and another mass
    connected to the first mass by a spring and damper
    """
    # return array
    dF = np.zeros((4))

    # velocities
    dF[0] = y[1]
    dF[2] = y[3]

    # positions
    dF[1] = ((p['f1'] / p['m1']) +
            ((p['b2'] / p['m1']) * (y[3] - y[1])) +
            ((p['k2'] / p['m1']) * (y[2] - y[0])) -
            ((p['b1'] / p['m1']) * y[1]) -
            ((p['k1'] / p['m1']) * y[0]))
    dF[3] = ((p['f2'] / p['m2']) +
            ((p['b2'] / p['m2']) * (y[3] - y[1])) -
            ((p['k2'] / p['m2']) * (y[2] - y[0])))

    return dF

pdb.set_trace()

out = odeint(F, y0, t)

pdb.set_trace()

plt.figure()
plt.plot(t, out[:, 1], 'r') # position of mass 1
plt.plot(t, out[:, 3], 'b') # position of mass 2
plt.savefig('plot.png')
