import numpy as np
from numpy import inf
from tqdm import tqdm

class ThreeDimMSM:
    """
    This class contains functions for solving n-dimensional mass-spring systems.
    It will set up coeffienct matrices for springs and dampers and use them to
    evaluate the ODEs

    Parameters:
    -----------
    dat = 2d array of floats
        mass# by x, y, z position, describes the positions of the masses
    pos = 1d array of floats
        [:pos.shape[0]]                    starting x positions
        [pos.shape[0]:2*pos.shape[0]]      starting y positions
        [2*pos.shape[0]:3*pos.shape[0]]    starting z positions
        [3*pos.shape[0]:4*pos.shape[0]]    starting x velocities
        [4*pos.shape[0]:5*pos.shape[0]]    starting y velocities
        [5*pos.shape[0]:6*pos.shape[0]]    starting z velocities
    masses = 1d array of floats
        actual masses of the masses in the model
    times = 1d array of floats
        times at which to evaluate the model
    k_mean = scalar
        mean for normal distribution of spring stiffness
    k_sd = scalar
        standard deviation for normal distribution of spring stiffness
    force_coeff = scalar
        value for the numerator of the attractive force magnitude (over r**2)
    random_seed = int
        used to set RandomState for reproducability
    """

    def __init__(self, pos, masses, times, k_mean = 4, k_sd = 2,
                 force_coeff = 5e7, random_seed = None):
        self.orig_pos = pos
        self.pos = np.copy(self.orig_pos)
        self.masses = masses.reshape((masses.shape[0],))
        self.times = times
        self.k_mean = k_mean
        self.k_sd = k_sd
        self.force_coeff = force_coeff
        self.randomseed = random_seed

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
        return self

    def establishSprings(self, rand, max_dist = None):
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
            maximum distance threshhold for connected masses. If the masses are
            initially within this distance threshold they will be connected by
            springs
        """
        k_vals = np.abs(rand.normal(loc=self.k_mean, scale=self.k_sd,
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

    def simpleAttraction(self, force_constant):
        """
        Attractive force between the masses which takes the form of:

        k/r^2

        where k is the force constant and r is the distance between two masses
        in a pair. This force is split into cartesian components and each mass
        is assigned a net force based on all of the components acting on it.

        Parameters:
        -----------
        force_constant = scalar
            Determines the overall strength of the attractive forces

        Return:
        -------
        forces = 2d array of floats (3, len(self.masses))
            array that describes x, y and z, components of the net force acting
            on each mass
        """
        # find offsets between every possible pair of bodies
        delta_x = np.subtract.outer(self.pos[:, 0], self.pos[:, 0])
        delta_y = np.subtract.outer(self.pos[:, 1], self.pos[:, 1])
        delta_z = np.subtract.outer(self.pos[:, 2], self.pos[:, 2])

        #dist = distance_matrix(self.pos, self.pos)
        dist = np.sqrt(delta_x**2 + delta_y**2 + delta_z**2)

        # standardize these distances
        norm_dx = delta_x / dist
        norm_dy = delta_y / dist
        norm_dz = delta_z / dist

        # diagonals should be zero not NaN
        diagonals = np.diag_indices(norm_dx.shape[0], ndim=2)
        norm_dx[diagonals] = 0
        norm_dy[diagonals] = 0
        norm_dz[diagonals] = 0

        # find magnitude of force acting between each pair of bodies
        total_force = force_constant / dist**2
        total_force[total_force == inf] = 0

        # find component forces between each pair of bodies according to magnitude
        fx_mat = norm_dx * total_force
        fy_mat = norm_dy * total_force
        fz_mat = norm_dz * total_force

        # find net component forces
        fx = np.sum(fx_mat, axis=0)
        fy = np.sum(fy_mat, axis=0)
        fz = np.sum(fz_mat, axis=0)
        forces = np.vstack((fx, fy, fz))

        return forces

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
        init = 1-D array
            positions and velocities before this iteration
        t = scalar
            time of evaluation
        """
        n_mass = self.pos.shape[0]

        x = init[:n_mass]
        y = init[n_mass:2*n_mass]
        z = init[2*n_mass:3*n_mass]

        vx = init[3*n_mass:4*n_mass]
        vy = init[4*n_mass:5*n_mass]
        vz = init[5*n_mass:]

        force = self.simpleAttraction(self.force_coeff)

        ddx = (force[0] -
               (self.cx_ @ vx) -
               (self.kx_ @ x)) / self.masses
        ddy = (force[1] -
                (self.cy_ @ vy) -
                (self.ky_ @ y)) / self.masses
        ddz = (force[2] -
                (self.cz_ @ vz) -
                (self.kz_ @ z)) / self.masses

        return np.hstack((vx, vy, vz, ddx, ddy, ddz))

    def solveRK4(self, fun, t):
        """
        forth-order Runge-Kutta solver for systems of ODEs.

        Parameters
        ----------
        fun = function(array of positions and velocities, scalar for time)
            This is the function that contains our discretized ODEs
        t = 1d array of floats
            This contains our times where we would like to evaluate our ODEs.
            assumes constant step size

        Returns:
        --------
        offsets = 2d array of floats (len(t), 6 * len(self.masses))
            for each mass this array contains x, y, and z offsets from the start
            position and x, y, and z velocites at each time point.
        """
        step = t[1] - t[0]
        offsets = np.zeros((self.times.shape[0], 6*self.masses.shape[0]))
        for time in tqdm(np.arange(self.times.shape[0])[1:]):
            k1 = fun(offsets[time-1,:], self.times[time]) * step
            k2 = fun(offsets[time-1,:] + k1/2., self.times[time])/2 * step
            k3 = fun(offsets[time-1,:] + k2/2., self.times[time])/2 * step
            k4 = fun(offsets[time-1,:] + k3, self.times[time]) * step
            offsets[time, :] = offsets[time-1, :] + k1/6 + k2/3 + k3/3 + k4/6
            self.updateDistances(offsets[time, :self.masses.shape[0]])
            self.updateSprings()

        return offsets

    def runRK4(self):
        """
        The only non initialization function we call in our script. This
        function establishes the springs and runs the simulation using a
        forth-order Runge-Kutta solver.
        """
        rng = np.random.RandomState()
        self.establishSprings(rng)
        return self.solveRK4(self.msmSys, self.times)
