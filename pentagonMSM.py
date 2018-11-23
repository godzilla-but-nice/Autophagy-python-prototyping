import numpy as np
import matplotlib.pyplot as plt
import pdb
from MSMsolver import *
from matplotlib.animation import FuncAnimation

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
masses = (1./100000.) * vols

start_pos = np.random.uniform(-100, 100, size=12)
#start_pos = np.zeros(12)
start_vel = np.zeros(12)
initials = np.hstack((start_pos, start_vel))

times = np.linspace(0, 100, 500)

feat = np.vstack((x_pts, y_pts))

msm = TwoDimMSM(feat, masses, initials, times)
output, report = msm.runSimulation()

# set up animation:
anim_x = np.zeros(output[:, :6].shape)
anim_y = np.zeros(output[:, 6:12].shape)

for i in range(len(x_pts)):
    anim_x[:, i] = output[:, i] + x_pts[i] #- initials[i]
    anim_y[:, i] = output[:, 6+i] + y_pts[i] #- initials[6+i]

plt.figure()
fig, ax = plt.subplots(figsize = (6, 6))
ax.set(xlim=(-500, 500), ylim =(-500, 500))
scat = ax.scatter(anim_x[0,:], anim_y[0,:], marker = 'o')

def animate(i):
    scat.set_offsets(np.c_[anim_x[i,:],anim_y[i,:]])

anim = FuncAnimation(fig, animate, interval=100, frames=len(times)-1)

plt.draw()
plt.show()

plt.figure()
plt.plot(times, output[:, 6:12])
plt.show()

# plt.scatter(x_pts, y_pts)
# plt.show()
