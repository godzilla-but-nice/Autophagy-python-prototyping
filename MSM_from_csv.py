import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
from scipy.spatial import distance_matrix
from ThreeDimMSM import *
import pdb

mass_coeff = 0.0000333

df = pd.read_csv('in_data/one_vacuole_t2.csv')
n_mass = df.shape[0]

# calculate parameters to pass to ThreeDimMSM
positions = df.loc[:,['x', 'y', 'z']].values

volumes = df.loc[:, ['radius']].values**3 * np.pi
masses = volumes * mass_coeff

times = np.linspace(0, 1000, 5000)

# set up msm object
msm = ThreeDimMSM(positions, masses, times, k_mean = 4, k_sd = 2,
                    force_coeff = 7e7)
output = msm.runRK4()

x_pos = output[:, :n_mass] + positions[:,0]
y_pos = output[:, n_mass:2*n_mass] + positions[:,1]
z_pos = output[:, 2*n_mass:3*n_mass] + positions[:,2]

x_vel = output[:, 3*n_mass:4*n_mass]
y_vel = output[:, 4*n_mass:5*n_mass]
z_vel = output[:, 5*n_mass:6*n_mass]
total_vel = np.sqrt(x_vel**2 + y_vel**2 + z_vel**2)

# 3d scatter plot showing overall movement of the masses
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
before = ax.scatter(x_pos[0,:], y_pos[0,:], z_pos[0,:], label = 'Before')
after = ax.scatter(x_pos[-1,:], y_pos[-1,:], z_pos[-1,:], label = 'After')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_zlabel('z')
ax.legend(loc='best')
fig.savefig('out_data/positions.png')

# find average serarations between bodies
avg_sep = np.zeros(times.shape[0])
for time in np.arange(times.shape[0]):
    position_at_time = np.asarray([x_pos[time,:], y_pos[time,:], z_pos[time,:]])
    dist_mat = distance_matrix(position_at_time.T, position_at_time.T)
    i_lower = np.tril_indices(dist_mat.shape[0])
    avg_sep[time] = np.mean(dist_mat[i_lower])

# supplemental plot velocities and average seperation over time
fig, ax = plt.subplots(nrows = 1, ncols = 2, figsize=(10,4))
# velocities
ax[0].plot(times, total_vel)
ax[0].set_xlabel('time')
ax[0].set_ylabel('velocity')

ax[1].plot(times, avg_sep)
ax[1].set_xlabel('time')
ax[1].set_ylabel('average separation')
fig.savefig('out_data/supplemental.png')


out_df = pd.DataFrame({'x': x_pos[-1, :],
                       'y': y_pos[-1, :],
                       'z': z_pos[-1, :],
                       'radius': df.loc[:, 'radius'].values})
out_df.to_csv('out_data/threedim.csv', index = False)
