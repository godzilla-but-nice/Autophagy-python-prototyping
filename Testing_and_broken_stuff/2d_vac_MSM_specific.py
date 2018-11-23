import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from scipy.integrate import odeint
import numpy.random as rng
import pdb
plt.style.use('ggplot')

# read data
vac = pd.read_csv('one_vac_2d.csv')

vac = vac.assign(id=vac.index.values)

# fig, ax = plt.subplots(figsize=(6, 6))
# ax.set(xlim=(-1043, 1043), ylim=(-1043, 1043))
# plt.scatter(vac['x'], vac['y'], color = 'gbrcy')
# plt.savefig('body_center.png')

# we have a total of 4 springs:
# 1-5
# 1-2
# 2-3
# 3-4

# set up some spring and damping constants
k = rng.random(4)
b = rng.random(4)

# we need to find the x and y components for each of these springs
# first we need to find the angle between each of these bodies
angles = np.zeros(4)
euc = np.zeros((4, 2))
euc[0, 0] = (vac.loc[vac.index[4], 'x'] - vac.loc[vac.index[0], 'x'])
euc[0, 1] = (vac.loc[vac.index[4], 'y'] - vac.loc[vac.index[0], 'y'])
euc[1, 0] = (vac.loc[vac.index[1], 'x'] - vac.loc[vac.index[0], 'x'])
euc[1, 1] = (vac.loc[vac.index[1], 'y'] - vac.loc[vac.index[0], 'y'])
euc[2, 0] = (vac.loc[vac.index[2], 'y'] - vac.loc[vac.index[1], 'x'])
euc[2, 1] = (vac.loc[vac.index[2], 'y'] - vac.loc[vac.index[1], 'y'])
euc[3, 0] = (vac.loc[vac.index[3], 'x'] - vac.loc[vac.index[2], 'x'])
euc[3, 1] = (vac.loc[vac.index[3], 'y'] - vac.loc[vac.index[2], 'y'])

angles = abs(np.arctan(euc[:, 1] / euc[:, 0]))

# im going to build the coefficient matrices a mix of by hand to begin with
# to show that this works
kx_mat = np.zeros((5, 5))
ky_mat = np.zeros((5, 5))
bx_mat = np.zeros((5, 5))
by_mat = np.zeros((5, 5))

kx_mat[0, :] = np.array([(k[0]*np.cos(angles[0]))+(k[1]*np.cos(angles[1])),
                        -k[1]*np.cos(angles[1]), 0, 0,
                        -k[0]*np.cos(angles[0])])
kx_mat[1, :] = np.array([-k[1]*np.cos((np.pi/2) - angles[1]),
                         (k[1]*np.cos((np.pi/2) - angles[1])) +
                         (k[2]*np.cos(angles[2])),
                         -k[2]*np.cos(angles[2]), 0, 0])
kx_mat[2, :] = np.array([0, -k[2]*np.cos((np.pi/2) - angles[2]),
                         (k[2]*np.cos((np.pi/2) - angles[2])) +
                         (k[3]*np.cos(angles[3])),
                         -k[3]*np.cos(angles[3]), 0])
kx_mat[3, :] = np.array([0, 0, 0,
                         -k[3]*np.cos((np.pi/2) - angles[3]),
                         k[3]*np.cos((np.pi/2) - angles[3])])
kx_mat[4, :] = np.array([-k[0]*np.cos((np.pi/2) - angles[0]),
                         0, 0, 0, -k[0]*np.cos((np.pi/2) - angles[0])])


ky_mat[0, :] = np.array([(k[0]*np.sin(angles[0]))+(k[1]*np.sin(angles[1])),
                        -k[1]*np.sin(angles[1]), 0, 0,
                        -k[0]*np.sin(angles[0])])
ky_mat[1, :] = np.array([-k[1]*np.sin((np.pi/2) - angles[1]),
                         (k[1]*np.sin((np.pi/2) - angles[1])) +
                         (k[2]*np.sin(angles[2])),
                         -k[2]*np.sin(angles[2]), 0, 0])
ky_mat[2, :] = np.array([0, -k[2]*np.sin((np.pi/2) - angles[2]),
                         (k[2]*np.sin((np.pi/2) - angles[2])) +
                         (k[3]*np.sin(angles[3])),
                         -k[3]*np.sin(angles[3]), 0])
ky_mat[3, :] = np.array([0, 0, 0,
                         -k[3]*np.sin((np.pi/2) - angles[3]),
                         k[3]*np.sin((np.pi/2) - angles[3])])
ky_mat[4, :] = np.array([-k[0]*np.sin((np.pi/2) - angles[0]),
                         0, 0, 0, -k[0]*np.sin((np.pi/2) - angles[0])])

# this ODE describes the position and velocity of the autophagic bodies
def harmonic(y, t, p = {'kx' : kx_mat, 'ky' : ky_mat,
                        'bx' : bx_mat, 'by' : ky_mat}):
    '''
    find positions and velocities for each mass at time t
    '''
    dFx = y[10:15]
    dFy = y[15:]

    #dGx = -(p['kx'] @ y[:5])
    #dGy = -(p['ky'] @ y[5:10])

    return np.hstack((dFx, dFy))#, dGx, dGy))

# starting values for position and velocity (length 20):
# [:5]    starting x
# [5:10]  starting y
# [10:15] starting vx
# [15:]   starting vy
init = np.zeros(15)
init[:5] = 0.3
init[5:10] = 0.3

# times where we will evaluate the funcion
t = np.linspace(0, 10, 100)

output, report = odeint(harmonic, init, t, full_output = 1)

plt.plot(t, output[:, :5])
plt.show()
pdb.set_trace()
