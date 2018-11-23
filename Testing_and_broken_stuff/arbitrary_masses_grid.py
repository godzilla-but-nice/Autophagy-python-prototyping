# this script simulates an arbitrary number of masses connected in a grid
# over two dimensions.
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
plt.style.use('ggplot')
#from matplotlib.backends import _macosx
from scipy.integrate import odeint
import numpy.random as rng
import pdb

# set seed for consistancy
np.random.seed(12345)

# set number of masses here (only works for squares right now):
num_m_x = num_m_y = 4
############################
tot_m = num_m_x * num_m_y
mass_mult = 10

# use random values for all of our parameters, store them as arrays
kx = rng.random((num_m_x, num_m_y + 1)).T
ky = rng.random((num_m_x + 1, num_m_y))
bx = rng.random((num_m_x, num_m_y + 1)).T
by = rng.random((num_m_x + 1, num_m_y))
f = rng.uniform(-10, 10, (num_m_x, num_m_y))
m = rng.random((num_m_x, num_m_y)).reshape(tot_m)
m = m * mass_mult
bx = bx * 2
by=by * 2
# kx = kx * 10
# ky = ky * 10

# weird iterators that help in our mapping of the coefficient (lowercase)
# matrices to our main calculation matrices (uppercase). It could be easier
# to store our coefficients one dimensionally?
col_x = 0
col_y = 0

# In each matrix we essentially have a set of independent matrices because
# some of our springs connect to a fixed surface instead of a previous mass in
# a series of masses. Each row corresponds to a body and each column is
# its direct relationship to the position or velocity of each other body
Kx, Bx = np.zeros((tot_m, tot_m)), np.zeros((tot_m, tot_m))
Ky, By = np.zeros((tot_m, tot_m)), np.zeros((tot_m, tot_m))
for row_num in np.arange(tot_m):
    for col_num in np.arange(tot_m):

        # these conditionals allow us to modify which column of the coefficient
        # matrices we are using to build our main matrix
        col_x = int((row_num+1) / (kx.shape[0]))
        col_y = int((row_num+1) / (kx.shape[0]))

        # pdb.set_trace()
        # this statement appropriately maps row_num (of our main matrix) to the
        # row of the coefficient matrices
        row_x = row_num % (kx.shape[0] - 1)
        row_y = row_num % (ky.shape[0] - 1)

        #pdb.set_trace()
        if row_num == col_num:
            Kx[row_num, col_num] = ((kx[row_x, col_x]+kx[row_x+1, col_x])
                                        /m[row_num])
            Bx[row_num, col_num] = ((bx[row_x, col_x]+bx[row_x+1, col_x])
                                        /m[row_num])
            Ky[row_num, col_num] = ((ky[row_y, col_y]+ky[row_y+1, col_y])
                                        /m[row_num])
            By[row_num, col_num] = ((by[row_y, col_y]+by[row_y+1, col_y])
                                        /m[row_num])

        if row_num - col_num == -1 and row_x != (kx.shape[0] - 2):
            Kx[row_num, col_num] = (-kx[row_x+1, col_x]/m[row_num])
            Bx[row_num, col_num] = -bx[row_x+1, col_x]/m[row_num]

        if row_num - col_num == -1 and row_y != (ky.shape[0] - 2):
            Ky[row_num, col_num] = -ky[row_y+1, col_y]/m[row_num]
            By[row_num, col_num] = -by[row_y+1, col_y]/m[row_num]

        if row_num - col_num == 1 and row_x != 0:
            Kx[row_num, col_num] = -kx[row_x, col_x]/m[row_num]
            Bx[row_num, col_num] = -bx[row_x, col_x]/m[row_num]

        if row_num - col_num == 1 and row_y != 0:
            Ky[row_num, col_num] = -ky[row_y, col_y]/m[row_num]
            By[row_num, col_num] = -by[row_y, col_y]/m[row_num]


# test so I dont have to do this manually over and over
print('X Top left verified:', Kx[0,0] == kx[0,0] + kx[1,0])
print('X Next one verified:', Kx[1,1] == kx[1,0] + kx[2,0])
print('X second to last:', Kx[-2,-2] == kx[0, -2] + kx[1, -2])
print('Y Top left verified:', Ky[0,0] == ky[0,0] + ky[1,0])
print('X bottom right verified:', Kx[-1, -1] == kx[-2, -1] + kx[-1, -1])
print('Y bottom right verified:', Ky[-1, -1] == ky[-2, -1] + ky[-1, -1])


#pdb.set_trace()

# times to check mass poitions:
t = np.linspace(0, 100, 300)

# starting values:
# [:tot_m]           x positions
# [tot_m:2*tot_m]      y positions
# [2*tot_m:3*tot_m]  x velocities
# [3*tot_m:]         y velocities
ini_pos = rng.uniform(-11, 11, 2 * tot_m)
#ini_pos = np.zeros(2 * tot_m)
ini_vel = np.zeros(2 * tot_m)
init = np.hstack((ini_pos, ini_vel))

# differential function
def linear_springs(y, t, p = {'kx' : Kx, 'ky' : Ky, 'bx' : Bx, 'by' : By,
                              'fx' : np.zeros(Ky.shape[0]),
                              'fy' : np.zeros(Ky.shape[0])}):
    '''
    find positions and velocities for each mass at time t
    '''
    # if t > 2.0 and t < 6.0:
    #     p['fy'] = np.where(np.arange(p['fy'].shape[0]) % 5 == 0, .1, -.1)
    #     p['fx'] = np.where(np.arange(p['fx'].shape[0]) % 2 == 0, .1, -.1)
    # else:
    #     p['fy'][0] = p['fy'][4] = p['fy'][8] = p['fy'][12] = 0.0

    dFx = y[2*tot_m:3*tot_m]
    dFy = y[3*tot_m:]

    dGx = (p['fx']-(p['bx'] @ dFx) - (p['kx'] @ y[:tot_m]))
    dGy = (p['fy']-(p['by'] @ dFy) - (p['ky'] @ y[tot_m:2*tot_m]))

    return np.hstack((dFx, dFy, dGx, dGy))

# run simulation
output, report = odeint(linear_springs, init, t, full_output = 1)


# plt.plot(t, output[:, :4], alpha = 0.5)
# #plt.savefig('plot.png')
# plt.show()

moves = output[:, :8]
plot_x = np.zeros(output[:,:16].shape)
plot_y = np.zeros(output[:,:16].shape)
starting_x = np.array([-6, -6, -6, -6,
                        -2, -2, -2, -2,
                        2, 2, 2, 2,
                        6, 6, 6, 6])
starting_y = np.array([6, 2, -2, -6,
                       6, 2, -2, -6,
                       6, 2, -2, -6,
                       6, 2, -2, -6])
for i in range(len(starting_x)):
    plot_x[:, i] = output[:, i] + starting_x[i]
    plot_y[:, i] = output[:, 16+i] + starting_y[i]

plt.figure()
fig, ax = plt.subplots(figsize=(6, 6))
ax.set(xlim=(-12, 12), ylim =(-12, 12))
#line = ax.plot(plot_x[0,:], plot_y[0,:])
scat = ax.scatter(plot_x[0,:], plot_y[0,:], marker = 'o')

def animate(i):
    # line.set_ydata(plot_x[i,:], plot_y[i,:])
    scat.set_offsets(np.c_[plot_x[i,:],plot_y[i,:]])

anim = FuncAnimation(fig, animate, interval=100, frames=len(t)-1)

plt.draw()
plt.show()
