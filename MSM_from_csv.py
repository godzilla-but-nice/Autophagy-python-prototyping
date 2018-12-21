import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance_matrix
from ThreeDimMSM import *
import pdb

mass_coeff = 0.0000333

df = pd.read_csv('one_vacuole_t2.csv')
n_mass = df.shape[0]

# calculate parameters to pass to ThreeDimMSM
positions = df.loc[:,['x', 'y', 'z']].values

volumes = df.loc[:, ['radius']].values**3 * np.pi
masses = volumes * mass_coeff
#masses = np.repeat(np.mean(volumes) * mass_coeff, 15)


start_pos = np.zeros(n_mass*3)
start_vel = np.zeros(n_mass*3)
init = np.hstack((start_pos, start_vel))

times = np.linspace(0, 100, 1000)

# set up msm object
msm = ThreeDimMSM(positions, masses, init, times)
output, com = msm.runRK2()

x_pos = output[:, :n_mass] + positions[:,0]
y_pos = output[:, n_mass:2*n_mass] + positions[:,1]
z_pos = output[:, 2*n_mass:3*n_mass] + positions[:,2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
before = ax.scatter(x_pos[0,:], y_pos[0,:], z_pos[0,:], label = 'Before')
one = ax.scatter(x_pos[99,:], y_pos[99,:], z_pos[99,:], label = 'one')
two = ax.scatter(x_pos[199,:], y_pos[199,:], z_pos[199,:], label = 'two')
thr = ax.scatter(x_pos[299,:], y_pos[299,:], z_pos[299,:], label = 'three')
fou = ax.scatter(x_pos[399,:], y_pos[399,:], z_pos[399,:], label = 'four')
fiv = ax.scatter(x_pos[499,:], y_pos[499,:], z_pos[499,:], label = 'five')
com = ax.scatter(com[:, 0], com[:, 1], com[:, 2], c='black', s=100)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(loc='best')

#plt.show()

# find average separation between bodies
avg_sep = np.zeros(times.shape[0])
for time in np.arange(times.shape[0]):
    position_at_time = np.asarray([x_pos[time,:], y_pos[time,:], z_pos[time,:]])
    dist_mat = distance_matrix(position_at_time.T, position_at_time.T)
    i_lower = np.tril_indices(dist_mat.shape[0])
    dist_mat[i_lower] = 0
    avg_sep[time] = np.mean(dist_mat)

plt.figure()
fig, ax = plt.subplots()
before_sep = ax.plot(times, avg_sep)
plt.show()
pdb.set_trace()
