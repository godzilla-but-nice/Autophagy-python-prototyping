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
        and damper coefficients and setting the anchor mass
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
        Locks a single mass in place to prevent system migrations

        Parameters:
        ----------

        rand = np.random.RandomState object
        """
        # this is what is actually needs to do:
        anchor_idx = rand.randint(0, self.pos.shape[0])
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
        self.pos[0,:] = self.pos[0,:] + delta_x
        self.pos[1,:] = self.pos[1,:] + delta_y
        self.d_mat_ = distance_matrix(self.pos.T, self.pos.T)
        return self

    def allConnections(self, rand, mean = 3, sd = 1e-10):
        """
        Set up spring and damper matrices, establishing all of the connections
        between masses in the model
        """
        k_vals = rand.normal(loc=mean*(10), scale=sd, size=(self.d_mat_.shape))
        c_vals = rand.normal(loc=mean*(20), scale=sd, size=(self.d_mat_.shape))

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

    def simpleAttraction(self, coeff = 1e4):
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

        # calculate coefficient matrix to give direction to forces
        dir_x = np.ones(self.d_mat_.shape)
        dir_y = np.ones(self.d_mat_.shape)
        for i in np.arange(self.pos.shape[1]):
            for j in np.arange(self.pos.shape[1]):
                if self.pos[0, j] - self.pos[0, i] > 0:
                    dir_x[i, j] = dir_x[i, j] * -1
                if self.pos[1, j] - self.pos[1, i] > 0:
                    dir_y[i, j] = dir_y[i, j] * -1

        # make this coefficient matrix symmetrical
        i_lower = np.tril_indices(self.d_mat_.shape[0])
        dir_x[i_lower] = dir_x.T[i_lower]
        dir_y[i_lower] = dir_y.T[i_lower]

        dist_x = dist_x * dir_x
        dist_y = dist_y * dir_y

        # make division by zero return zero
        dist_x = np.where(dist_x == 0, float('inf'), dist_x)
        dist_y = np.where(dist_y == 0, float('inf'), dist_x)

        fx_arr = coeff / dist_x**2
        fy_arr = coeff / dist_y**2

        #pdb.set_trace()

        # don't count the infinate terms
        f[0,:] = np.sum(fx_arr, axis = 1)
        f[1,:] = np.sum(fy_arr, axis = 1)

        #pdb.set_trace()

        # trying some weird stuff
        f[0,:] = np.sum(dir_x, axis = 1) * 800
        f[1,:] = np.sum(dir_y, axis = 1) * 800

        return f

    def zeroForce(self):
        """
        returns a force vector of aprropriate length for ODE system with zero
        force
        """
        return np.zeros(self.pos.shape[1])

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

        ddx = (self.simpleAttraction()[0] -
               (self.cx_ @ dx) -
               (self.kx_ @ init[:n_mass])) / self.masses
        ddy = (self.simpleAttraction()[1] -
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
        self.allConnections(rng)
        return odeint(self.msmSys, self.starts, self.times, full_output = 1)
