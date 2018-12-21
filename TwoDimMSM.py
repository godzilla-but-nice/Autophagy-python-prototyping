import numpy as np
from scipy.integrate import odeint
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import pdb

class TwoDimMSM:
    """
    This class contains functions for solving 2d mass-spring systems.
    It will determine which masses are interacting based on proximity,
    set up coeffienct matrices for springs and dampers, and use them to
    evaluate the ODEs

    Parameters:
    -----------
    dat = 2d array of floats
        #mass x (radius, x position, y position) describes the masses
    starts = 1d array of floats
        [:masses.shape[0]]                    starting x positions
        [masses.shape[0]:2*masses.shape[0]]   starting y positions
        [2*masses.shape[0]:3*masses.shape[0]] starting x velocities
        [3*masses.shape[0]:]                  starting y velocities
    times = 1d array of floats
        times at which we evaluate the function
    randomseed = int
        used to set RandomState which is relevant in setting spring
        and damper coefficients and setting the anchor mass
    """

    def __init__(self, pos, mass, starts, times, randomseed = 12345):
        self.orig_pos = pos
        self.pos = np.copy(self.orig_pos)
        self.masses = mass
        self.starts = starts
        self.times = times
        self.random_seed = randomseed
        self.updateDistances()

    def setAnchor(self, rand):
        """
        Locks a single mass in place to prevent system migrations

        Parameters:
        ----------

        rand = np.random.RandomState object
        """
        # this is what is actually needs to do:
        #anchor_idx = rand.randint(0, self.pos.shape[0])
        anchor_idx = 5
        self.masses[anchor_idx] = 1e15

        return self

    def updateDistances(self, delta_x = 0.0, delta_y = 0.0):
        """
        Function to track distances for the forcing function

        Parameters:
        ----------
        delta_x = 1-D array, changes in x in nm
        delta_y = 1-D array, changes in y in nm
        """
        self.pos[0,:] = self.orig_pos[0,:] + delta_x
        self.pos[1,:] = self.orig_pos[1,:] + delta_y
        self.d_mat_ = distance_matrix(self.pos.T, self.pos.T)
        return self

    def allConnections(self, rand, k_mean = 3, k_sd = 1.5,
                                   c_mean = 10, c_sd = 0):
        """
        Set up spring and damper matrices, establishing all of the connections
        between masses in the model
        """
        # generate k values. They should be consistant whether we are looking
        # at either mass at the ends of the springs.
        k_vals = np.abs(rand.normal(loc=k_mean, scale=k_sd, size=(self.d_mat_.shape)))
        i_lower = np.tril_indices(k_vals.shape[0])
        k_vals[i_lower] = k_vals.T[i_lower]

        # We want c = 2*sqrt(k*m) so that will be a couple of calculations
        m_mat = np.add.outer(self.masses, self.masses)
        c_vals = 2 * (k_vals * m_mat)**0.5

        pdb.set_trace()
        pdb.set_trace()
        self.kx_ = np.zeros(self.d_mat_.shape)
        self.ky_ = np.zeros(self.d_mat_.shape)
        self.cx_ = np.zeros(self.d_mat_.shape)
        self.cy_ = np.zeros(self.d_mat_.shape)

        for row in np.arange(self.d_mat_.shape[1]):
            # we can use the angle between this (row) body and each other body
            # to make our spring constants act the correct amount in the x and
            # y directions
            angles = np.arctan2(self.pos[1,:] - self.pos[1, row],
                                self.pos[0,:] - self.pos[0, row])

            # we need to set the entry in the row corresponding to this body
            # equal to zero. Springs dont connect masses to themselves. We also
            # need to redo the springs so they care about the angles
            k_x_row = np.absolute(k_vals[row,:] * np.cos(angles))
            k_x_row[row] = 0.0
            k_y_row = np.absolute(k_vals[row,:] * np.sin(angles))
            k_y_row[row] = 0.0

            # now we need to do the same thing for the damping constants
            c_x_row = np.absolute(c_vals[row,:] * np.cos(angles))
            c_x_row[row] = 0.0
            c_y_row = np.absolute(c_vals[row,:] * np.sin(angles))
            c_y_row[row] = 0.0

            for col in np.arange(self.d_mat_.shape[1]):
                #pdb.set_trace()
                if row == col:
                    self.kx_[row, col] = np.sum(k_x_row)
                    self.ky_[row, col] = np.sum(k_y_row)

                    self.cx_[row, col] = np.sum(c_x_row)
                    self.cy_[row, col] = np.sum(c_y_row)
                else:
                    self.kx_[row, col] = -k_x_row[col]
                    self.ky_[row, col] = -k_y_row[col]

                    self.cx_[row, col] = -c_x_row[col]
                    self.cy_[row, col] = -c_y_row[col]

        return self

    def simpleAttraction(self, coeff = 1e10):
        """
        Forcing function that provides a simple attractive force using the
        expressions:

        Fx = k / (x2 - x1)
        Fy = k / (x2 - x1)

        where k is some coefficient

        Parameters:
        -----------
        coeff = scalar, determines the strength of the attractive forces
        """
        f = np.zeros(self.pos.shape)

        # get correct shape for distance_matrix: (n,) to (n,1)
        xs = self.pos[0,:].reshape(self.pos.shape[1], 1)
        ys = self.pos[1,:].reshape(self.pos.shape[1], 1)

        dist_x = distance_matrix(xs, xs)
        dist_y = distance_matrix(ys, ys)

        # working on a more computationally efficient way to generate directions
        more_x = np.greater.outer(self.pos[0, :], self.pos[0, :])
        more_y = np.greater.outer(self.pos[1, :], self.pos[1, :])
        dir_x = np.where(more_x == True, -1, 1)
        dir_y = np.where(more_y == True, -1, 1)

        # make this direction coefficient matrix symmetrical
        # i_lower = np.tril_indices(self.d_mat_.shape[0])
        # dir_x[i_lower] = dir_x.T[i_lower]
        # dir_y[i_lower] = dir_y.T[i_lower]

        dist_x = dist_x * dir_x
        dist_y = dist_y * dir_y

        # print(dist_x)
        # print(dir_x)
        # print(self.pos)
        # pdb.set_trace()

        # make division by zero return zero
        dist_x = np.where(dist_x == 0, 1e-5, dist_x)
        dist_y = np.where(dist_y == 0, 1e-5, dist_x)

        fx_arr = coeff / dist_x**2
        fy_arr = coeff / dist_y**2

        # don't count the infinate terms
        f[0,:] = np.sum(fx_arr, axis = 1)
        f[1,:] = np.sum(fy_arr, axis = 1)

        #pdb.set_trace()

        # trying some weird stuff
        #f[0,:] = np.sum(dir_x, axis = 1) * 100
        #f[1,:] = np.sum(dir_y, axis = 1) * 100

        return f

    def pseudoAttraction(self):
        """
        Returns a foce vector that directs force toward the center of mass at
        each time point
        """
        # find center of mass
        CMx = np.sum(self.masses * self.pos[0,:])
        CMy = np.sum(self.masses * self.pos[1,:])
        tot_mass = np.sum(self.masses)

        center_of_mass = np.asarray([CMx/tot_mass, CMy/tot_mass])

        # find direction vectors to the center of mass from each body
        offset_x = center_of_mass[0] - self.pos[0,:]
        offset_y = center_of_mass[1] - self.pos[1,:]
        center_offset = np.vstack((offset_x, offset_y))
        center_dist = (center_offset[0,:]**2 + center_offset[1,:]**2)**0.5
        norm_x = offset_x / center_dist
        norm_y = offset_y / center_dist

        # protect myself from division by zero
        dist_zero = np.logical_and(center_dist < 1e-9, center_dist > -1e-9)
        center_dist[dist_zero] = 1e15

        f_mag = 3e7 / center_dist**2

        fx = f_mag * norm_x
        fy = f_mag * norm_y
        f = np.vstack((fx, fy))

        # plt.figure()
        # plt.scatter(self.pos[0], self.pos[1])
        # plt.scatter(center_of_mass[0], center_of_mass[1], c='r')
        # plt.quiver(self.pos[0], self.pos[1], fx, fy)
        # plt.show()
        # pdb.set_trace()

        return f

    def zeroForce(self):
        """
        returns a force vector of aprropriate length for ODE system with zero
        force
        """
        return np.zeros(self.pos.shape)

    def msmSys(self, init, t):
        """
        This is our system of ODEs. The first two equations are just
        dx = v
        and the second is the big one
        ddx = f(x)-[c]v-[k]x
        """
        n_mass = self.pos.shape[1]

        dx = init[2*n_mass:3*n_mass]
        dy = init[3*n_mass:]

        forces = self.pseudoAttraction()

        ddx = (forces[0] -
               (self.cx_ @ dx) -
               (self.kx_ @ init[:n_mass])) / self.masses
        ddy = (forces[1] -
                (self.cy_ @ dy) -
                (self.ky_ @ init[n_mass : 2*n_mass])) / self.masses

        return np.hstack((dx, dy, ddx, ddy))

    def runSimulation(self):
        """
        Ideally the only thing we need to run in our script
        """
        rng = np.random.RandomState()
        #self.setAnchor(rng)
        self.allConnections(rng)
        return odeint(self.msmSys, self.starts, self.times,
                        full_output = 1)

    def solveRK2(self, fun, t):
        """
        Second-order Runge-Kutta solver for systems of ODEs.

        Parameters
        ----------

        """
        step = t[1] - t[0]
        output = np.zeros((self.times.shape[0], 4*self.masses.shape[0]))
        pdb.set_trace()
        for time in np.arange(self.times.shape[0])[1:]:
            k1 = fun(output[time-1,:], self.times[time]) * step
            k2 = fun(output[time-1,:] + k1/2., 0)/2 * step
            output[time, :] = output[time-1, :] + k2
            self.updateDistances(output[time, :self.masses.shape[0]])
            # print('dx: {}'.format(output[time, :self.masses.shape[0]]))
            # print('x pos: {}'.format(self.pos[0,:]))
            # pdb.set_trace()
        return output

    def runRK2(self):
        rng = np.random.RandomState()
        self.allConnections(rng)
        self.setAnchor(rng)
        return self.solveRK2(self.msmSys, self.times)
