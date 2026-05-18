import numpy as np
from amuse.units import units
from amuse.datamodel import Particles, Particle, new_regular_grid

from shared_src.extra_funcs import dynamical_mass, Rhills
from shared_src.migration_map_paadekooper import cal_tau_I
from shared_src.params import GAMMA


def check_resonance(planets, planetsmass, j_values):
    """
    Ensure that no adjacent planets cross resonance locations.
    """
    for i in range(len(planets) - 1):
        inner = planets[i]
        outer = planets[i+1]
        mass_inner = planetsmass[i]
        mass_outer = planetsmass[i+1]
        j = j_values[i]  # Resonance value (j+1):j

        # Minimum ratio of semi-major axes to maintain resonance
        min_ratio = ((j + 1) / j) ** (2 / 3)

        # If the planets are about to cross the resonance, adjust the two planets
        if outer / inner < min_ratio:
            inner05 = (mass_inner*inner**0.5+mass_outer*outer**0.5)/(mass_inner+mass_outer*((j+1)/j)**(1/3))
            planets[i] = inner05**2
            planets[i+1] = planets[i] * ((j+1)/j)**(2/3)
    return planets


class nonisothermal_Migration:
    def __init__ (self, timestep, ndisk_cells):
        """Class to handle non-isothermal Type I migration using Paardekooper et al. (2011)."""
        self.planets = Particles()
        self.planets.add_calculated_attribute('dynamical_mass', dynamical_mass)
        
        # DEBUG - Why is it fixed for Solar mass star?
        self.star = Particle(mass=1|units.MSun)
        self.star_teff = 5775 | units.K
        self.model_time = 0. | units.Myr

        self.disk = new_regular_grid(([int(ndisk_cells)]),[1])
        self.disk.surface_gas = 100 | units.g/units.cm**2
        self.disk.surface_solid = 10 | units.g/units.cm**2
        self.disk.temperature = 10 | units.K
        self.disk.scale_height = 0.03 * self.disk.position
        self.disk.alpha = 2e-3

        self.gamma = GAMMA
        self.dt = timestep
        self.eta = 0.1 # control the timestep
        
    def set_time_step(self, tau_I, model_time_i, end_time):
        ratio = (
            Rhills(
                self.planets.dynamical_mass,
                self.star.mass,
                self.planets.semimajor_axis
            ) / self.planets.semimajor_axis
        )
        ratio_min = ratio.flatten().min()
        dt_hill = np.log(1 + ratio_min) * tau_I
        dt_min = self.eta* tau_I
        dt = min(abs(dt_hill), abs(end_time-model_time_i), abs(dt_min))
        return dt

    def access_migration_map(self, rp, Mp):
        Ms, gamma, sigmag, sigmad, tempd, rgrid = self.star.mass, self.gamma, self.disk.surface_gas, self.disk.surface_solid, self.disk.temperature, self.disk.position
        M_planet, M_star, sigma_g, sigma_d, r_grid = Mp.value_in(units.g), Ms.value_in(units.g), sigmag.value_in(units.g/units.cm**2), sigmad.value_in(units.g/units.cm**2), rgrid.value_in(units.cm)
        rpj = rp.value_in(units.cm)
        temp_d = tempd.value_in(units.K)
        alpha = self.disk.alpha[0] # WARNING: change 0 to ip if alpha is not uniform
        Z, Mig_ratej = cal_tau_I(np.array([rpj]), M_planet, M_star, gamma, sigma_g, sigma_d, temp_d, r_grid, alpha)

        return Z, Mig_ratej|units.yr**-1

    def evolve_model (self, end_time):
        model_time_i = self.model_time
        Rdisk = self.disk.position

        # Update temperature analytically, now move it to disk evolution model (some problems).
        # dtgr = self.disk.surface_solid/self.disk.surface_gas
        # temp_d = np.array(cal_temperature(self.disk.position.value_in(units.cm),self.star.mass.value_in(units.g),self.star.radius.value_in(units.cm),
        #                                   self.star_teff.value_in(units.K), self.disk.alpha[0], self.disk.surface_gas.value_in(units.g/units.cm**2), dtgr)) |units.K
        # self.disk.temperature = temp_d.reshape((pre_ndisk,1))

        while model_time_i < end_time:
            tau_a = np.zeros(len(self.planets)) | units.kyr
            for i in range(len(self.planets)):
                ap = self.planets[i].semimajor_axis
                #Type I & Type II
                if (ap<=Rdisk[0]) or (ap>=Rdisk[-1]):
                    tau_a[i] = np.inf | units.kyr
                    print("WARNING: Planet(s) is out of the grids")
                else:
                    _, rate = self.access_migration_map(ap, self.planets[i].dynamical_mass)
                    tau_a[i] = -rate**-1

            dt = self.set_time_step(min(abs(tau_a)), model_time_i, end_time)
            model_time_i += dt

            for i in range(len(self.planets)):
                ap = self.planets[i].semimajor_axis
                a_dot = -ap/tau_a[i]
                self.planets[i].semimajor_axis += a_dot * dt
            j_values = np.ceil(((self.planets.semimajor_axis[1:]/self.planets.semimajor_axis[:-1])**(3/2)-1+0.01)**(-1))
            self.planets.semimajor_axis = check_resonance(
                self.planets.semimajor_axis, 
                self.planets.dynamical_mass, 
                j_values
                )

            if dt == 0|units.s:
                break
        
        self.model_time = end_time
