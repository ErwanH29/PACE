import numpy as np
import scipy.interpolate as si

from amuse.lab import units

G0 = 1.6e-3 * units.erg/units.s/units.cm**2.


class FRIED_interp(object):
    """
    FRIED interpolator object.
    As there is a degeneracy between disk radius, disk mass
    and disk outer density, we neglect disk outer density.
    """
    def __init__ (
        self, 
        data_path='disk_data/', 
        logMstar=True, 
        logF=True, 
        logMdisk=True, 
        logRdisk=True, 
        verbosity=False
    ):
        """
        Initialise FRIED_interp object
        Args:
            data_path (str):   Directory of FRIED grid data file
            logMstar (bool):   Make log grid of host star mass (else linear)
            logF (bool):       Make log grid of incident FUV field (else linear)
            logMdisk (bool):   Make log grid of disk mass (else linear)
            logRdisk (bool):   Make log grid of disk radius (else linear)
            verbosity (bool):  Print warning message if points are outside domain
        """

        mstar_grid, f_grid, mdisk_grid, rdisk_grid = np.loadtxt(
            data_path, usecols=(0,1,2,4), unpack=True
            )
        self._logMdot_grid = np.loadtxt(data_path, usecols=(5,))

        self._logMstar = logMstar
        self._logF = logF
        self._logMdisk = logMdisk
        self._logRdisk = logRdisk
        self.verbosity = verbosity
        self.backup_counter = 0

        if self._logMstar:
            mstar_grid = np.log10(mstar_grid)
        if self._logF:
            f_grid = np.log10(f_grid)
        if self._logMdisk:
            mdisk_grid = np.log10(mdisk_grid)
        if self._logRdisk:
            rdisk_grid = np.log10(rdisk_grid)

        self._grid = np.array([
            mstar_grid,
            f_grid,
            mdisk_grid,
            rdisk_grid,
        ]).T

        self._interp = si.LinearNDInterpolator(
            self._grid, self._logMdot_grid
            )
        self._backup_interp = si.NearestNDInterpolator(
            self._grid, self._logMdot_grid
            )

    def interp(self, Mstar, F, Mdisk, Rdisk):
        """
        Compute mass loss rate at N positions on the grid.
        
        Args:
            Mstar (array):  Host star mass
            F (array):      Incident FUV field
            Mdisk (array):  Disk mass
            Rdisk (array):  Disk radius
        Returns:
            Mdot (array): Mass loss rate for each set of parameters
        """
        def _setup_input(x, log_flag):
            arr = np.asarray(x, dtype=np.float64)
            return np.log10(arr) if log_flag else arr

        try:
            n = len(Mstar)
        except:
            n = 1

        Mstar = _setup_input(Mstar, self._logMstar)
        F = _setup_input(F, self._logF)
        Mdisk = _setup_input(Mdisk, self._logMdisk)
        Rdisk = _setup_input(Rdisk, self._logRdisk)

        empty_arr = np.ones(n)
        query = np.array([
            Mstar * empty_arr,
            F * empty_arr,
            Mdisk * empty_arr,
            Rdisk * empty_arr
        ]).T

        logMdot = self._interp(query)
        mask_nan = np.isnan(logMdot)
        logMdot[mask_nan] = self._backup_interp(query[mask_nan])
        self.backup_counter += np.sum(mask_nan)
        if self.verbosity:
            print(
                f"[WARNING] {int(np.sum(mask_nan))} points outside interpolation domain. "
                "Using nearest neighbour fallback.",
                flush=True,
            )

        return 10.**logMdot

    def interp_amuse(self, Mstar, F, Mdisk, Rdisk):
        """
        Compute mass loss rate at N positions on the grid.
        
        Args:
            Mstar (array): Host star mass
            F (array):     Incident FUV field
            Mdisk (array): Disk mass
            Rdisk (array): Disk radius
        Returns:
            Mdot (array): Mass loss rate for each set of parameters
        """
        return self.interp(
            Mstar.value_in(units.MSun),
            F.value_in(G0),
            Mdisk.value_in(units.MJupiter),
            Rdisk.value_in(units.AU),
        ) | units.MSun / units.yr