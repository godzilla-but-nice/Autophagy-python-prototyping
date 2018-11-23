import numpy as np
from scipy.integrate import odeint
from scipy.spatial import distance_matrix
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
        and damper coefficients
    """

    def __init__(self, pos, mass, starts, times, randomseed = 12345):
        self.pos = pos
        self.masses = mass
        self.starts = starts
        self.times = times
        self.random_seed = randomseed
        self.updateDistances()

    def setAnchor(self, rand):
        """
        locks a single mass in place to prevent system migrations
        """
        # this is what is actually needs to do:

        # anchor_idx = rand.randint(0, self.pos.shape[0])
        # self.masses[anchor_idx] = 1e15

        # this is for testing
        self.masses[2] = 1e15
        return self

    def updateDistances(self, delta_x = 0.0, delta_y = 0.0):
        """
        Function to track distances for the forcing function

        Arguments:
        ----------
        delta_x = 1-D array, changes in x in nm
        delta_y = 1-D array, changes in y in nm
        """
        self.pos[0,:] = self.pos[0,:] + delta_x
        self.pos[1,:] = self.pos[1,:] + delta_y
        self.d_mat_ = distance_matrix(self.pos.T, self.pos.T)
        return self

    def makeConnections(self, rand, mean = 10, sd = 1e-10):
        """
        Set up spring and damper matrices, establishing all of the connections
        between masses in the model
        """
        k_vals = rand.normal(loc=mean*(1), scale=sd, size=(self.d_mat_.shape))
        c_vals = rand.normal(loc=mean*(1), scale=sd, size=(self.d_mat_.shape))

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

        self.cx_ = self.cx_ * 1
        self.cy_ = self.cy_ * 1

        return self

    def simpleAttraction(self, coeff = 1e4):
        f = np.zeros(self.pos.shape)
        xs = self.pos[0,:].reshape(self.pos.shape[1], 1)
        ys = self.pos[1,:].reshape(self.pos.shape[1], 1)
        dist_x = distance_matrix(xs, xs)
        fx_arr = coeff / dist_x
        dist_y = distance_matrix(ys, ys)
        fy_arr = coeff / dist_y
        f[0,:] = np.sum(np.where(fx_arr > 1e10, 0, fx_arr), axis = 1)
        f[1,:] = np.sum(np.where(fy_arr > 1e10, 0, fy_arr), axis = 1)
        return f
        # return np.sum(coeff / self.d_mat_, axis = 1)

    def zeroForce(self):
        """
        returns a force vector of aprropriate length for ODE system with zero
        force
        """
        return np.zeroes(self.pos.shape)

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

        if (t > 10 and t < 20):
            force_x = np.zeros(6)
            force_x[0] = 2e3
            force_y = np.zeros(6)
            force_y[0] = -1.2e3
        else:
            force_x = np.zeros(6)
            force_y = np.zeros(6)

        #self.updateDistances(dx, dy)

        ddx = (force_x -
               (self.cx_ @ dx) -
               (self.kx_ @ init[:n_mass])) / self.masses
        ddy = (force_y -
                (self.cy_ @ dy) -
                (self.ky_ @ init[n_mass : 2*n_mass])) / self.masses

        self.updateDistances(dx, dy)

        return np.hstack((dx, dy, ddx, ddy))

    def runSimulation(self):
        """
        Ideally the only thing we need to run in our script
        """
        rng = np.random.RandomState(self.random_seed)
        self.setAnchor(rng)
        self.makeConnections(rng)
        return odeint(self.msmSys, self.starts, self.times, full_output = 1)
