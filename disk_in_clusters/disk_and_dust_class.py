import numpy as np
import queue

from amuse.units import units, constants

from shared_src.params import (
    FRAGMENT_V, SUBL_TEMP, CFL, A0, RHO_DUST,
    G0, SIGMA_FLOOR
)


code_queue = queue.Queue()

DustParameters = {  # arXiv:2302.03721
    'uf': FRAGMENT_V | units.cm/units.s,
    'rho_s': RHO_DUST,
    'a0': A0,
    'Tsubl': SUBL_TEMP,
    'CFL': CFL
}


class Disk:
    def __init__(
        self,
        disk_radius,
        disk_gas_mass,
        central_mass,
        grid,
        alpha,
        mu=2.33,
        Tm=None,
        delta=1e-2,
        rho_g=1.|units.g/units.cm**3,
        a_min=1e-8|units.m,
        ipe_flag=True,
        epe_flag=True,
        critical_radius=None
    ):
        """
        Initialise a protoplanetary disk.
        
        Args:
            disk_radius (float):      Outer radius of the disk.
            disk_gas_mass (float):    Total gas mass in the disk.
            central_mass (float):     Mass of the central star.
            grid (Grid):              VADER grid object defining the radial structure of the disk.
            alpha (float):            Viscosity parameter for calculating the viscous timescale.
            mu (float):               Mean molecular weight of the gas.
            Tm (float):               Mid-plane temperature at 1 au.
            delta (float):            Dust-to-gas mass ratio.
            rho_g (float):            Internal density of dust grains.
            a_min (float):            Minimum grain size for dust evolution.
            ipe_flag (bool):          Whether to include internal photoevaporation.
            epe_flag (bool):          Whether to include external photoevaporation.
            critical_radius (float):  Characteristic radius for the exponential cutoff in the column density profile.
        """
        self.model_time = 0.0 | units.Myr
        self.central_mass = central_mass
        self.mu = mu
        self.delta = delta
        self.rho_g = rho_g
        self.a_min = a_min
        self.accreted_mass = 0.0 | units.MSun
        self.truncation_mass_loss = 0.0 | units.MSun
        self.fuv_ambient_flux = 0.0 | G0
        self.outer_photoevap_rate = 0.0 | units.MSun / units.yr
        self.epe_mass_loss = 0.0 | units.MSun
        self.ipe_mass_loss = 0.0 | units.MSun
        self.ipe_flag = ipe_flag
        self.epe_flag = epe_flag
        self.viscous = None
        self.disk_dispersed = False
        self.disk_convergence_failure = False
        self.disk_active = True

        if Tm is None:
            Tm = (270. | units.K) * central_mass.value_in(units.MSun)
        self.Tm = Tm
        temp_profile = Tm / np.sqrt(grid.r.value_in(units.au))

        R1 = disk_radius if critical_radius is None else critical_radius
        nu = alpha * constants.kB / constants.u * self.Tm / np.sqrt(R1.value_in(units.au))
        nu *= (R1**3 / (constants.G * central_mass))**0.5
        self.t_viscous = R1 * R1 / (3.0 * nu)

        self.grid = grid.copy()
        self.grid.column_density = self.column_density(
            disk_radius, disk_gas_mass, rcut=critical_radius
        )
        self.grid.pressure = (
            self.grid.column_density
            * constants.kB
            * temp_profile
            / (mu * 1.008 * constants.u)
        )

        self._disk_dust_mass = self.delta * self.disk_gas_mass

    def assign_code(self, viscous_code):
        self.viscous = viscous_code

    def _get_disk_scale_height(self, Tm, disk_radius, central_mass):
        """
        Calculate the disk scale height at a given radius.
        
        Args:
            Tm (float):           Mid-plane temperature at 1 au (in K).
            disk_radius (float):  Radius at which to calculate the scale height (in au).
            central_mass (float): Mass of the central star (in solar masses).
        Returns:
            scale_height (float): Disk scale height at the given radius (in au).
        """
        rscale = 1 | units.au
        return (
            constants.kB * Tm * (rscale)**0.5 * disk_radius**(2.5)
            / (self.mu * 1.008 * constants.u * central_mass * constants.G)
        )**0.5

    def _get_thermal_speed(self, Tm, disk_radius):
        """
        Calculate the thermal speed at a given disk radius.
        
        Args:
            Tm (float):           Mid-plane temperature at 1 au (in K).
            disk_radius (float):  Radius at which to calculate the thermal speed (in au).
        Returns:
            thermal_speed (float): Thermal speed at the given radius (in cm/s).
        """
        disk_rad_au = disk_radius.value_in(units.au)
        return (
            8 * constants.kB * Tm / np.sqrt(disk_rad_au)
            / (np.pi * self.mu * 1.008 * constants.u)
        )**0.5

    def _set_cell_state(self, idx, sigma):
        self.grid[idx].column_density = sigma
        temperature = self.Tm / np.sqrt(self.grid[idx].r.value_in(units.au))
        self.grid[idx].pressure = (
            sigma * constants.kB * temperature / (self.mu * 1.008 * constants.u)
        )

    def column_density(self, rdisk, disk_gas_mass, lower_density=SIGMA_FLOOR, rcut=None):
        """
        Calculate the disk column density based on 1974MNRAS.168..603L
        
        Args:
            rdisk (float):         Outer radius of the disk.
            disk_gas_mass (float): Total gas mass in the disk.
            lower_density (float): Minimum column density to return.
            rcut (float):          Characteristic radius for the exponential cutoff.
        """
        if rcut is None:
            rcut = rdisk
        r = self.grid.r.copy()
        Sigma_0 = disk_gas_mass / (
            2.0 * np.pi * rcut**2 * (1.0 - np.exp(-rdisk / rcut))
        )
        return Sigma_0 * (rcut / r) * np.exp(-r / rcut) * (r <= rdisk) + lower_density

    def evaporate_mass(self, mass_to_remove):
        """
        Evaporate mass from the disk by reducing the column density of the outermost cells.
        
        Args:
            mass_to_remove (float): The total mass to evaporate from the disk.
        """
        N = len(self.grid.r)
        removed_mass = 0.0 | units.MSun

        for i in range(N):
            idx = N - 1 - i
            cell = self.grid[idx]
            if cell.column_density <= SIGMA_FLOOR:
                continue

            removable_mass = cell.area * (cell.column_density - SIGMA_FLOOR)
            if removed_mass + removable_mass > mass_to_remove:
                dsigma = (mass_to_remove - removed_mass) / cell.area
                sigma = cell.column_density - dsigma
                self._set_cell_state(idx, sigma)
                return

            removed_mass += removable_mass
            self._set_cell_state(idx, SIGMA_FLOOR)

    def truncate_disk(self, new_radius, vader_mode):
        """
        Truncate the disk to a new radius due to a close encounter.
        
        Args:
            new_radius (float): The new outer radius of the disk after truncation
            vader_mode (str):   VADER mode being used ("pedisk_dusty", "pedisk", or "pedisk_nataccr")
        Returns:
            removed_mass (float): The mass lost due to truncation (in solar masses)
        """
        if new_radius >= self.disk_radius:
            return 0.0 | units.MSun

        mask = self.grid.r > new_radius
        if not np.any(mask):
            return 0.0 | units.MSun
        
        print(f"Truncation: {self.grid.r.max().value_in(units.au)} au -> {new_radius.value_in(units.au)} au", flush=True)

        removable_sigma = self.grid[mask].column_density - SIGMA_FLOOR
        removed_mass = (self.grid[mask].area * removable_sigma).sum()
        self.grid[mask].column_density = SIGMA_FLOOR

        temperature = self.Tm / np.sqrt(self.grid[mask].r.value_in(units.au))
        self.grid[mask].pressure = (
            SIGMA_FLOOR * constants.kB * temperature / (self.mu * 1.008 * constants.u)
        )

        if vader_mode == "pedisk_dusty":
            self.grid_user[0, mask].value = 1e-14
            self.grid_user[1, mask].value = DustParameters["a0"].value_in(units.cm)

        if removed_mass > (0.0 | units.MSun):
            self.truncation_mass_loss += removed_mass
        return removed_mass

    @property
    def accretion_rate(self):
        """Calculate mass-dependent accretion rate of T-Tauri stars (arXiv:1310.2069)."""
        return (
            10.**(1.81 * np.log10(self.central_mass.value_in(units.MSun)) - 8.25)
            * (1. + self.model_time / self.t_viscous)**(-1.5)
            | units.MSun / units.yr
        )

    @property
    def inner_photoevap_rate(self):
        """Calculate the inner photoevaporation rate. Mass-scale via arXiv:1112.1087, rate via arXiv:1904.02752"""
        Lx = self.xray_luminosity.value_in(units.erg / units.s)
        return (
            10.**(
                -2.7326 * np.exp(
                    -(np.log(np.log10(Lx)) - 3.3307) ** 2 / 2.9868e-3
                ) - 7.2580
            )
            * (self.central_mass / (0.7 | units.MSun)) ** -0.068
            | units.MSun / units.yr
        )

    @property
    def xray_luminosity(self):
        """Calculate mass-dependenc x-ray luminosity of classical T-Tauri stars (arXiv:1210.6770)"""
        return 10.**(1.7 * np.log10(self.central_mass.value_in(units.MSun)) + 30.0) | units.erg / units.s

    @property
    def disk_radius(self, f=0.999):
        """
        Calcualte the gas radius of the disk.
        
        Args:
            f (float): Fraction of the total disk mass to include within the radius.
        Returns:
            radius (float): The radius containing fraction f of the total disk mass (in au).
        """
        cml_mass = (self.grid.area * self.grid.column_density).cumsum()
        edge = np.argmax(cml_mass >= cml_mass[-1] * f)
        return self.grid.r[edge]

    @property
    def disk_gas_mass(self):
        """Get the total gas mass in the disk"""
        return (self.grid.area * self.grid.column_density).sum()

    @property
    def disk_dust_mass(self):
        """Get the total dust mass in the disk"""
        if hasattr(self, 'grid_user'):
            return (self.grid.area * (self.grid_user[0].value | units.g / units.cm**2)).sum()
        return self._disk_dust_mass

    @property
    def disk_mass(self):
        """Get total disk mass"""
        return self.disk_dust_mass + self.disk_gas_mass

    @property
    def age(self):
        return self.model_time