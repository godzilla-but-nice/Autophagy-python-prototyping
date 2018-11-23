# this script simulates an arbitrary number of masses connected linearly by
# springs and dampers

import pdb
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends import _macosx
from scipy.integrate import odeint
import numpy.random as rng

# set seed for consistancy
#np.random.seed(12345)

# set number of masses here:
num_m = 6
mass_mult = 10

# number of springs and dampers and forces are based on the number of masses:
num_k = num_m + 1
num_b = num_m + 1
num_f = num_m

# use random values for all of our parameters
k = rng.random(num_k)
b = rng.random(num_b)
f = rng.uniform(-10, 10, num_f)
m = rng.random(num_m)
m = m * mass_mult

# times to check mass poitions:
t = np.linspace(0, 250, 500)

# populate matrices for our coefficients:
k_mat = b_mat = np.zeros((num_m, num_m))
for i in np.arange(num_m):
    for j in np.arange(num_m):
        if i == j:
            k_mat[i, j] = (k[i] + k[i+1]) / m[i]
            b_mat[i, j] = (b[i] + b[i+1]) / m[i]
        elif i - j == 1:
            k_mat[i, j] = -k[i] / m[i]
            b_mat[i, j] = -b[i] / m[i]
        elif i - j == -1:
            k_mat[i, j] = -k[i+1] / m[i]
            b_mat[i, j] = -b[i+1] / m[i]

# initial values array, first num_m terms refer to starting positions, second,
# velocity
init = np.zeros(2 * num_m)

# differential function
def linear_springs(y, t, p = {'k' : k_mat, 'b' : b_mat, 'f' : f}):
    '''
    find positions and velocities for each mass after time t
    '''
    dF = y[num_m:]
    dG = p['f'] - (p['b'] @ y[num_m:]) - (p['k'] @ y[:num_m])

    return np.hstack((dF, dG))

# run simulation
output, report = odeint(linear_springs, init, t, full_output = 1)
plt.plot(t, output[:, :num_m], alpha = 0.5)
#plt.savefig('plot.png')
plt.show()
