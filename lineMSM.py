import numpy as np
import matplotlib.pyplot as plt
import pdb
from TwoDimMSM import *
from matplotlib.animation import FuncAnimation

x_pts = np.zeros(2)
y_pts = np.zeros(2)

# these calculations are pentagon specific. They give me a pentagon inscribed
# in a circle of radius 300 centered around the origin
x_pts[0] = -100
x_pts[1] = 100

# radii of the masses
radii = np.ones(len(x_pts)) * 150
vols = (4./3.) * np.pi * radii**3
masses = (1./1e5) * vols

# start_pos = np.random.normal(0, 40, size=4)
start_pos = np.zeros(4)
start_vel = np.zeros(4)
initials = np.hstack((start_pos, start_vel))

times = np.linspace(0, 100, 200)

feat = np.vstack((x_pts, y_pts))

msm = TwoDimMSM(feat, masses, initials, times)
output = msm.runRK2()

# set up animation:
anim_x = np.zeros(output[:, :2].shape)
anim_y = np.zeros(output[:, 2:4].shape)

for i in range(len(x_pts)):
    anim_x[:, i] = output[:, i] + x_pts[i]
    anim_y[:, i] = output[:, 2+i] + y_pts[i]
    momentum = (output[:, 4+i]**2 + output[:, 6+i]**2)**0.5

plt.figure()
fig, ax = plt.subplots(figsize = (6, 6))
ax.set(xlim=(-500, 500), ylim =(-500, 500))
scat = ax.scatter(anim_x[0,:], anim_y[0,:], marker = 'o')

# set up a text box that will display momentum at all times
def p_template(momentum):
    return 'Momentum: {}'.format(round(momentum, 2))

def time_template(time):
    return 'Time: {}'.format(round(time, 2))

p_text = ax.text(0.05, 0.9, '', transform=ax.transAxes)
time_text = ax.text(0.05, 0.85, '', transform=ax.transAxes)

def animate(i):
    scat.set_offsets(np.c_[anim_x[i,:],anim_y[i,:]])

    tot_momentum = momentum[i]
    p_text.set_text(p_template(tot_momentum))

    time = times[i]
    time_text.set_text(time_template(time))

    return scat, p_text

anim = FuncAnimation(fig, animate, interval=100, frames=len(times)-1)

plt.draw()
plt.show()

plt.figure()
plt.plot(times, output[:, :4])
plt.show()
