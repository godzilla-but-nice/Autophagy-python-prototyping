import numpy as np
import matplotlib.pyplot as plt
import pdb
from TwoDimMSM import *
from matplotlib.animation import FuncAnimation
import pandas as pd

x_pts = np.zeros(6)
y_pts = np.zeros(6)

# these calculations are pentagon specific. They give me a pentagon inscribed
# in a circle of radius 300 centered around the origin
x_pts[1] = 300 * np.cos(np.pi/10.0)
x_pts[2] = 300 * np.cos(0.942478)
x_pts[3] = -x_pts[2]
x_pts[4] = -x_pts[1]

y_pts[0] = 300
y_pts[1] = 300 * np.sin(np.pi/10.0)
y_pts[2] = -300 * np.sin(0.942478)
y_pts[3] = y_pts[2]
y_pts[4] = y_pts[1]

# radii of the masses
radii = np.ones(len(x_pts)) * 150
vols = (4./3.) * np.pi * radii**3
masses = (1./1e5) * vols

#start_pos = np.random.normal(0, 40, size=12)
start_pos = np.zeros(12)
start_vel = np.zeros(12)
initials = np.hstack((start_pos, start_vel))

times = np.linspace(0, 300, 500)

feat = np.vstack((x_pts, y_pts))

msm = TwoDimMSM(feat, masses, initials, times)
output = msm.runRK2()

#################
# BUILD ANIMATION
#################
# set up scatter arrays:
anim_x = np.zeros(output[:, :6].shape)
anim_y = np.zeros(output[:, 6:12].shape)
center_of_mass = np.zeros((output.shape[0], 2))

for i in range(len(x_pts)):
    anim_x[:, i] = output[:, i] + x_pts[i] #- initials[i]
    anim_y[:, i] = output[:, 6+i] + y_pts[i] #- initials[6+i]
    momentum = (output[:, 12+i]**2 + output[:, 18+i]**2)**0.5

center_of_mass[:, 0] = np.sum(output[:, :6] * masses) / np.sum(masses)
center_of_mass[:, 1] = np.sum(output[:, 6:12] * masses) / np.sum(masses)

plt.figure()
fig, ax = plt.subplots(figsize = (6, 6))
ax.set(xlim=(-500, 500), ylim =(-500, 500))
scat = ax.scatter(anim_x[0,:], anim_y[0,:], marker = 'o')
com = ax.scatter(center_of_mass[0,0], center_of_mass[0,1], c='r', s=100)

# set up a text box that will display momentum at all times
def p_template(momentum):
    return 'Momentum: {}'.format(round(momentum, 2))

def time_template(time):
    return 'Time: {}'.format(round(time, 2))

p_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
time_text = ax.text(0.05, 0.85, '', transform=ax.transAxes)

def animate(i):
    scat.set_offsets(np.c_[anim_x[i,:],anim_y[i,:]])
    com.set_offsets(np.c_[center_of_mass[i, 0], center_of_mass[i, 1]])

    tot_momentum = momentum[i]
    p_text.set_text(p_template(tot_momentum))

    time = times[i]
    time_text.set_text(time_template(time))

    return scat, p_text

anim = FuncAnimation(fig, animate, interval=10, frames=len(times)-1)

plt.draw()
plt.show()

plt.figure()
plt.plot(times, output[:, 6:12])
plt.show()

# assemble this stuff into a dataframe so I can easily visualize it in R
df = pd.DataFrame({'x':anim_x[-1, :],
                   'y':anim_y[-1, :],
                   'radius':np.repeat(150, 6)})
df.to_csv('pentagon.csv', index = False)
