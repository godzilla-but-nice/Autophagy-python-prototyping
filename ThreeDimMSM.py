import numpy as np
from scipy.integrate import odeint
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import pdb

class ThreeDimMSM:
    """
    This class contains functions for solving n-dimensional mass-spring systems.
    It will determine which masses are interacting based on proximity,
    set up coeffienct matrices for springs and dampers, and use them to
    evaluate the ODEs

    Parameters:
    -----------
    dat = 2d array of floats
        mass# by x, y, z position, describes the positions of the masses
    starts = 1d array of floats
        [:pos.shape[0]]                    starting x positions
        [pos.shape[0]:2*pos.shape[0]]      starting y positions
        [2*pos.shape[0]:3*pos.shape[0]]    starting z positions
        [3*pos.shape[0]:4*pos.shape[0]]    starting x velocities
        [4*pos.shape[0]:5*pos.shape[0]]    starting y velocities
        [5*pos.shape[0]:6*pos.shape[0]]    starting z velocities
    masses = 1d array of floats
        actual masses of the masses in the model
    times = tuple (t0, tf)
        contains start and end time
    randomseed = int
        used to set RandomState which is relevant in setting spring
        and damper coefficients and setting the anchor mass
    """

    def __init__(self, pos, masses, starts, times, randomseed = 12345):
        self.orig_pos = pos
        self.pos = np.copy(self.orig_pos)
        self.masses = masses.reshape((masses.shape[0],))
        self.starts = starts
        self.times = times
        self.random_seed = randomseed
        self.com_ = np.zeros((times.shape[0], 3))
        self.updateDistances()

    def setAnchor(self, rand):
        """
        Locks a single mass in place to prevent system migrations

        Parameters:
        ----------
        rand = np.random.RandomState object
        """
        self.anchor_idx_ = rand.randint(0, self.pos.shape[0])
        #self.masses[self.anchor_idx_] = 1e15

        return self

    def updateDistances(self, delta_x = 0.0, delta_y = 0.0, delta_z = 0.0):
        """
        Function to track distances for the forcing function

        Parameters:
        ----------
        delta_x = 1-D array, changes in x in nm
        delta_y = 1-D array, changes in y in nm
        delta_z = 1-D array, changes in z in nm
        """
        self.pos[:,0] = self.orig_pos[:,0] + delta_x
        self.pos[:,1] = self.orig_pos[:,1] + delta_y
        self.pos[:,2] = self.orig_pos[:,2] + delta_z
        self.d_mat_ = distance_matrix(self.pos, self.pos)
        return self

    def allConnections(self, rand, k_mean = 3, k_sd = 0):
        """
        Set up spring and damper matrices, establishing connections between
        all masses in the model

        Parameters
        ----------
        rand = numpy.random.RandomState object
            state for setting randomized coefficients
        k_mean = scalar
            mean for normally distributed spring coefficients
        k_sd = scalar
            standard deviation for normally distributed spring coefficients
        c_mean = scalar
            mean for normally distributed damper coefficients
        c_sd = scalar
            standard deviation for normally distributed damper coefficients
        """
        # k-values will be diagonally symetrical so that springs provide equal
        # forces regardless of how we look at them
        k_vals = np.abs(rand.normal(loc=k_mean, scale=k_sd, size=(self.d_mat_.shape)))
        i_lower = np.tril_indices(k_vals.shape[0])
        k_vals[i_lower] = k_vals.T[i_lower]

        # these calculations set up critical damping in our springs
        mass1d = self.masses.reshape((self.masses.shape[0],))
        m_mat = np.add.outer(mass1d, mass1d)
        c_vals = 2 * (k_vals * m_mat)**0.5

        # initialize all of these final coeffient matrices
        self.kx_ = np.zeros(self.d_mat_.shape)
        self.ky_ = np.zeros(self.d_mat_.shape)
        self.kz_ = np.zeros(self.d_mat_.shape)
        self.cx_ = np.zeros(self.d_mat_.shape)
        self.cy_ = np.zeros(self.d_mat_.shape)
        self.cz_ = np.zeros(self.d_mat_.shape)

        for row in np.arange(self.d_mat_.shape[1]):
            # we can use the angles between this (row) body and each other body
            # to make our spring constants act the correct amount in the x and
            # y directions. theta = azimuthal, phi = polar angle
            phi = np.arctan2(self.pos[:,1] - self.pos[row, 1],
                             self.pos[:,0] - self.pos[row, 0])

            # theta = np.arccos((self.pos[:, 2] - self.pos[row, 2]) /
            #     (np.sqrt(self.pos[:,0]**2 + self.pos[:,1]**2 + self.pos[:,2]**2)) -
            #     (np.sqrt(self.pos[row,0]**2+self.pos[row,1]**2+self.pos[row,2]**2)))

            theta = np.arccos((self.pos[:, 2] - self.pos[row, 2]) /
                (np.sqrt((self.pos[:, 0] - self.pos[row, 0])**2 +
                          (self.pos[:, 1] - self.pos[row, 1])**2 +
                          (self.pos[:, 2] - self.pos[row, 2])**2)))
            theta = np.where(np.isnan(theta), 0, theta)

            # we need to set the entry in the row corresponding to this body
            # equal to zero. Springs dont connect masses to themselves. We also
            # need to redo the springs so they care about the angles
            k_x_row = np.absolute(k_vals[row,:] * np.sin(theta) * np.cos(phi))
            k_x_row[row] = 0.0
            k_y_row = np.absolute(k_vals[row,:] * np.sin(theta) * np.sin(phi))
            k_y_row[row] = 0.0
            k_z_row = np.absolute(k_vals[row,:] * np.cos(theta))
            k_z_row[row] = 0.0
            # now we need to do the same thing for the damping constants
            c_x_row = np.absolute(c_vals[row,:] * np.sin(theta) * np.cos(phi))
            c_x_row[row] = 0.0
            c_y_row = np.absolute(c_vals[row,:] * np.sin(theta) * np.cos(phi))
            c_y_row[row] = 0.0
            c_z_row = np.absolute(c_vals[row,:] * np.cos(theta))
            c_z_row[row] = 0.0

            for col in np.arange(self.d_mat_.shape[1]):
                if row == col:
                    self.kx_[row, col] = np.sum(k_x_row)
                    self.ky_[row, col] = np.sum(k_y_row)
                    self.kz_[row, col] = np.sum(k_z_row)

                    self.cx_[row, col] = np.sum(c_x_row)
                    self.cy_[row, col] = np.sum(c_y_row)
                    self.cz_[row, col] = np.sum(c_z_row)
                else:
                    self.kx_[row, col] = -k_x_row[col]
                    self.ky_[row, col] = -k_y_row[col]
                    self.kz_[row, col] = -k_z_row[col]

                    self.cx_[row, col] = -c_x_row[col]
                    self.cy_[row, col] = -c_y_row[col]
                    self.cz_[row, col] = -c_z_row[col]

        return self

    def pseudoAttraction(self):
        """
        Returns a foce vector that directs force toward the center of mass at
        each time point
        """
        # find center of mass
        CMx = np.sum(self.masses * self.pos[:,0])
        CMy = np.sum(self.masses * self.pos[:,1])
        CMz = np.sum(self.masses * self.pos[:,2])
        tot_mass = np.sum(self.masses)

        center_of_mass = np.asarray([CMx/tot_mass, CMy/tot_mass, CMz/tot_mass])

        # find direction vectors to the center of mass from each body
        offset_x = center_of_mass[0] - self.pos[:,0]
        offset_y = center_of_mass[1] - self.pos[:,1]
        offset_z = center_of_mass[2] - self.pos[:,2]
        center_offset = np.vstack((offset_x, offset_y, offset_z))
        center_dist = (center_offset[0,:]**2 + center_offset[1,:]**2
                            + center_offset[2,:]**2)**0.5
        norm_x = offset_x / center_dist
        norm_y = offset_y / center_dist
        norm_z = offset_z / center_dist

        # protect myself from division by zero
        dist_zero = np.logical_and(center_dist < 1e-12, center_dist > -1e-12)
        center_dist[dist_zero] = 1e15

        f_mag = 3e4 #/ center_dist**2

        fx = f_mag * norm_x
        fx[self.anchor_idx_] = 0
        fy = f_mag * norm_y
        fy[self.anchor_idx_] = 0
        fz = f_mag * norm_z
        fz[self.anchor_idx_] = 0
        f = np.vstack((fx, fy, fz))

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection = '3d')
        # ax.scatter(self.pos[:,0], self.pos[:,1], self.pos[:,2])
        # ax.scatter(center_of_mass[0], center_of_mass[1], center_of_mass[2], c='r', s=100)
        # ax.quiver(self.pos[:,0], self.pos[:,1], self.pos[:,2], fx/200, fy/200, fz/200, alpha=0.6)
        # ax.set_xlim((-1059, 1059))
        # ax.set_ylim((-1059, 1059))
        # ax.set_zlim((-1059, 1059))
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.show()
        # pdb.set_trace()


        return f

    def zeroForce(self):
        """
        returns a force vector of aprropriate length for ODE system with zero
        external force
        """
        return np.zeros(self.pos.shape[0])

    def msmSys(self, init, t):
        """
        This is our system of ODEs. The first two equations are just
        dx = v
        and the second is the big one
        ddx = f(x)-[c]v-[k]x

        Variables in these equations:
        v (n,) velocity of each mass
        x (n,) position of each mass
        f(x) (n,) external forces acting on each body
        [c] (n,n) damping coefficients
        [k] (n,n) spring coefficients

        Parameters:
        -----------
        init = 1-D array, positions and velocities before this iteration
        t = time of evaluation
        """
        n_mass = self.pos.shape[0]

        dx = init[3*n_mass:4*n_mass]
        dy = init[4*n_mass:5*n_mass]
        dz = init[5*n_mass:]

        force = self.pseudoAttraction()

        ddx = (force[0] -
               (self.cx_ @ dx) -
               (self.kx_ @ init[:n_mass])) / self.masses
        ddy = (force[1] -
                (self.cy_ @ dy) -
                (self.ky_ @ init[n_mass:2*n_mass])) / self.masses
        ddz = (force[2] -
                (self.cz_ @ dz) -
                (self.kz_ @ init[2*n_mass:3*n_mass])) / self.masses

        #ddx[self.anchor_idx_] = 0.
        #ddy[self.anchor_idx_] = 0.
        #ddz[self.anchor_idx_] = 0.

        return np.hstack((dx, dy, dz, ddx, ddy, ddz))

    def runSimulation(self):
        """
        Do all of the work to set up and run the model making this the only
        function we need to call in our script
        """
        rng = np.random.RandomState(self.random_seed)
        self.setAnchor(rng)
        self.allConnections(rng)
        return odeint(self.msmSys, self.starts, self.times, full_output = 1)

    def solveRK2(self, fun, t):
        """
        Second-order Runge-Kutta solver for systems of ODEs.

        Parameters
        ----------

        """
        step = t[1] - t[0]
        output = np.zeros((self.times.shape[0], 6*self.masses.shape[0]))
        for time in np.arange(self.times.shape[0])[1:]:
            k1 = fun(output[time-1,:], self.times[time]) * step
            k2 = fun(output[time-1,:] + k1/2., 0)/2 * step
            output[time, :] = output[time-1, :] + k2
            self.updateDistances(output[time, :self.masses.shape[0]])

            # center of mass output for debug
            CMx = np.sum(self.masses * self.pos[:,0])
            CMy = np.sum(self.masses * self.pos[:,1])
            CMz = np.sum(self.masses * self.pos[:,2])
            tot_mass = np.sum(self.masses)
            self.com_[time, :] = np.asarray([CMx/tot_mass, CMy/tot_mass, CMz/tot_mass])
        return output, self.com_

    def runRK2(self):
        rng = np.random.RandomState()
        self.allConnections(rng)
        self.setAnchor(rng)
        return self.solveRK2(self.msmSys, self.times)
