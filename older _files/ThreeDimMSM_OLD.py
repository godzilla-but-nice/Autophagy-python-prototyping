import numpy as np
from scipy.integrate import odeint
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from numpy import inf
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

        CMx = np.sum(self.masses * self.pos[:,0])
        CMy = np.sum(self.masses * self.pos[:,1])
        CMz = np.sum(self.masses * self.pos[:,2])
        tot_mass = np.sum(self.masses)

        self.com_ = np.asarray([CMx/tot_mass, CMy/tot_mass, CMz/tot_mass])
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

    def establishSprings(self, rand, k_mean = 3, k_sd = 1, max_dist = None):
        """
        Assign spring constants for the springs between masses in the system
        using a normal distribution. By default establishes springs connecting
        each mass to each other mass but this behavior can be changed by passing
        a value to max_dist.

        k_mean: scalar
            mean value for generation of spring constants
        k_sd: scalar
            standard deviation for generation of spring constants
        max_dist: scalar
            maximum distance threshhold for connected masses. closer to one
            another than this value will be connected by springs
        """
        k_vals = np.abs(rand.normal(loc=k_mean, scale=k_sd,
                        size=((self.masses.shape[0], self.masses.shape[0]))))
        i_lower = np.tril_indices(k_vals.shape[0])
        k_vals[i_lower] = k_vals.T[i_lower]
        i_diag = np.diag_indices(k_vals.shape[0])
        k_vals[i_diag] = 0.

        # set critical damping
        m_mat = np.add.outer(self.masses, self.masses)
        c_vals = 2 * m_mat * (k_vals / m_mat)**0.5

        if max_dist != None:
            k_vals[np.where(self.d_mat_ > max_dist)] = 0.
            c_vals[np.where(self.d_mat_ > max_dist)] = 0.

        self.k_vals_ = k_vals
        self.c_vals_ = c_vals

        self.updateSprings()

        return self

    def updateSprings(self):
        """
        Update the spring values to reflect the changing relative positions
        of the masses.

        self.phi_: polar angle (radians)
        self.theta_: azimuthal angle (radians)
        """
        # We want to find vectors between each body
        delta_x = np.subtract.outer(self.pos[:, 0], self.pos[:, 0])
        delta_y = np.subtract.outer(self.pos[:, 1], self.pos[:, 1])
        delta_z = np.subtract.outer(self.pos[:, 2], self.pos[:, 2])

        # normalize these to unit length
        mag = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)
        norm_dx = np.abs(delta_x / mag)
        norm_dy = np.abs(delta_y / mag)
        norm_dz = np.abs(delta_z / mag)

        # we need to remove the NaN values along the diagonal because physically
        # we know they should be represented by zeros
        diagonals = np.diag_indices(norm_dx.shape[0], ndim=2)
        norm_dx[diagonals] = 0
        norm_dy[diagonals] = 0
        norm_dz[diagonals] = 0

        # the values for individual springs between each pair of bodies
        raw_kx = self.k_vals_ * norm_dx
        raw_ky = self.k_vals_ * norm_dy
        raw_kz = self.k_vals_ * norm_dz

        # anything that isnt on the diagonal is just the - of the appropriate value
        self.kx_ = -raw_kx
        self.ky_ = -raw_ky
        self.kz_ = -raw_kz

        # same proceedure for the damping
        raw_cx = self.c_vals_ * norm_dx
        raw_cy = self.c_vals_ * norm_dy
        raw_cz = self.c_vals_ * norm_dz

        self.cx_ = -raw_cx
        self.cy_ = -raw_cy
        self.cz_ = -raw_cz

        # The diagonal positions should be the positive of the sum of the row
        for mass in np.arange(self.masses.shape[0]):
            self.kx_[mass, mass] = np.sum(raw_kx[mass,:])
            self.ky_[mass, mass] = np.sum(raw_ky[mass,:])
            self.kz_[mass, mass] = np.sum(raw_kz[mass,:])
            self.cx_[mass, mass] = np.sum(raw_cx[mass,:])
            self.cy_[mass, mass] = np.sum(raw_cy[mass,:])
            self.cz_[mass, mass] = np.sum(raw_cz[mass,:])

        return self

    def pseudoAttraction(self):
        """
        Returns a foce vector that directs force toward the center of mass at
        each time point
        """
        # find direction vectors to the center of mass from each body
        offset_x = self.com_[0] - self.pos[:,0]
        offset_y = self.com_[1] - self.pos[:,1]
        offset_z = self.com_[2] - self.pos[:,2]
        center_offset = np.vstack((offset_x, offset_y, offset_z))
        center_dist = (center_offset[0,:]**2 + center_offset[1,:]**2
                            + center_offset[2,:]**2)**0.5

        # protect myself from division by zero
        dist_zero = np.logical_and(center_dist < 1e-15, center_dist > -1e-15)
        center_dist[dist_zero] = 1e3

        norm_x = offset_x / center_dist
        norm_y = offset_y / center_dist
        norm_z = offset_z / center_dist

        f_mag = 6e2 #/ center_dist**2


        fx = f_mag * norm_x
        fy = f_mag * norm_y
        fz = f_mag * norm_z
        f = np.vstack((fx, fy, fz))

        return f

    def simpleAttraction(self, force_constant = 5e7):
        """
        Attractive force between the masses
        """
        # find offsets between every possible pair of bodies
        delta_x = np.subtract.outer(self.pos[:, 0], self.pos[:, 0])
        delta_y = np.subtract.outer(self.pos[:, 1], self.pos[:, 1])
        delta_z = np.subtract.outer(self.pos[:, 2], self.pos[:, 2])

        dist = distance_matrix(self.pos, self.pos)
        mag = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

        # standardize these distances
        norm_dx = delta_x / dist
        norm_dy = delta_y / dist
        norm_dz = delta_z / dist

        # diagonals should be zero not NaN
        diagonals = np.diag_indices(norm_dx.shape[0], ndim=2)
        norm_dx[diagonals] = 0
        norm_dy[diagonals] = 0
        norm_dz[diagonals] = 0

        total_force = force_constant / dist**2
        total_force[total_force == inf] = 0
        fx_mat = norm_dx * total_force
        fy_mat = norm_dy * total_force
        fz_mat = norm_dz * total_force

        fx = np.sum(fx_mat, axis=0)
        fy = np.sum(fy_mat, axis=0)
        fz = np.sum(fz_mat, axis=0)
        forces = np.vstack((fx, fy, fz))

        # fig = plt.figure()
        # ax = fig.add_subplot(111, projection = '3d')
        # ax.scatter(self.pos[:,0], self.pos[:,1], self.pos[:,2])
        # ax.scatter(self.com_[0], self.com_[1], self.com_[2], c='r', s=100)
        # ax.quiver(self.pos[:,0], self.pos[:,1], self.pos[:,2],
        #                             fx_mat, fy_mat, fz_mat, alpha=0.6)
        # ax.set_xlim((-1059, 1059))
        # ax.set_ylim((-1059, 1059))
        # ax.set_zlim((-1059, 1059))
        # ax.set_xlabel('x')
        # ax.set_ylabel('y')
        # ax.set_zlabel('z')
        # plt.show()
        # # print('com x: {0}, y: {1}, z:{2}'.format(round(self.com_[0],2),
        # #                                          round(self.com_[1],2),
        # #                                          round(self.com_[2],2)))
        # pdb.set_trace()

        return forces

    def oneForce(self):
        """
        returns a force vector of aprropriate length for ODE system with one
        external force acting on a single body
        """
        f0 = np.zeros(self.pos.shape[0])
        f1 = np.copy(f0)
        f1[0] = 10000
        return np.vstack((f1, f0, f0))

    def zeroForce(self):
        """
        returns a force vector of aprropriate length for ODE system with zero
        external force
        """
        f0 = np.zeros(self.pos.shape[0])

        return np.vstack((f0, f0, f0))

    def parallelForces(self):
        """
        Returns forces with a constant value for every dimension
        """
        f = np.ones(self.pos.shape[0]) * 3e2

        return [f, f, f]

    def msmSys(self, init, t):
        """
        This is our system of ODEs. The first two equations are
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

        x = init[:n_mass]
        y = init[n_mass:2*n_mass]
        z = init[2*n_mass:3*n_mass]

        vx = init[3*n_mass:4*n_mass]
        vy = init[4*n_mass:5*n_mass]
        vz = init[5*n_mass:]

        if t > 0:
            force = self.simpleAttraction(6e7)
        else:
            force = self.zeroForce()

        ddx = (force[0] -
               (self.cx_ @ vx) -
               (self.kx_ @ x)) / self.masses
        ddy = (force[1] -
                (self.cy_ @ vy) -
                (self.ky_ @ y)) / self.masses
        ddz = (force[2] -
                (self.cz_ @ vz) -
                (self.kz_ @ z)) / self.masses

        # ddx[self.anchor_idx_] = 0.
        # ddy[self.anchor_idx_] = 0.
        # ddz[self.anchor_idx_] = 0.

        return np.hstack((vx, vy, vz, ddx, ddy, ddz))

    def solveRK2(self, fun, t):
        """
        Second-order Runge-Kutta solver for systems of ODEs.

        Parameters
        ----------

        """
        step = t[1] - t[0]
        offsets = np.zeros((self.times.shape[0], 6*self.masses.shape[0]))
        for time in np.arange(self.times.shape[0])[1:]:
            k1 = fun(offsets[time-1,:], self.times[time]) * step
            k2 = fun(offsets[time-1,:] + k1/2., self.times[time])/2 * step
            offsets[time, :] = offsets[time-1, :] + k2
            self.updateDistances(offsets[time, :self.masses.shape[0]])
            self.updateSprings()

        return offsets, self.com_

    def runRK2(self):
        rng = np.random.RandomState()
        self.establishSprings(rng)
        self.setAnchor(rng)
        return self.solveRK2(self.msmSys, self.times)
