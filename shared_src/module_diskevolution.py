"""Evolve gas and dust disk using Vader (Krumholz+2015) and pedisk (Birnstiel+2012, Wilhelm+2023)"""
import numpy as np

from amuse.community.vader.interface import Vader
from amuse.units import units, constants
from amuse.datamodel import Particle, new_regular_grid

from shared_src.params import (
    A0, MU, STOKES_NUMBER, ALPHA, SIGMA_FLOOR, 
    SUBL_TEMP, CFL, RHO_DUST
)


class DiskGasDustEvolution:
    """Class to handle the evolution of gas and dust in the protoplanetary disk using VADER and pedisk."""
    def __init__(self, ndisk_cells):
        self.code = Vader(mode='pedisk_dusty')
        self.code.module_time = 0|units.Myr
        self.model_time = 0 | units.Myr
        
        # DEBUG - Why is this fixed?
        self.star = Particle(mass=1|units.MSun)
        self.star_teff = 5775 | units.K
        self.mu = MU

        self.disk = new_regular_grid(([int(ndisk_cells)]), [1]|units.au)
        self.disk.surface_gas = 1e-20 | units.g/units.cm**2
        self.disk.surface_solid = 1e-20 | units.g/units.cm**2
        self.disk.vd = 0. | units.cm/units.s
        self.disk.st = STOKES_NUMBER
        self.disk.alpha = ALPHA

    @property
    def inner_photoevap_rate(self):
        '''
        Internal photoevaporation rate of protoplanetary disks from Picogna et al. 
        2019, with mass scaling following Owen et al. 2012
        '''
        Lx = self.xray_luminosity.value_in( units.erg / units.s )

        return 10.**( -2.7326*np.exp(
            -( np.log(np.log10( Lx )) - 3.3307 )**2/2.9868e-3 ) - 7.2580) \
            * (self.star.mass/(0.7 | units.MSun))**-0.068 | units.MSun / units.yr

    @property
    def xray_luminosity(self):
        '''
        Mass-dependent X-ray luminosity of classical T-Tauri stars according to 
        Flaccomio et al. 2012 (typical luminosities)
        '''
        star_mass = self.star.mass.value_in(units.MSun)
        time_Myr  = self.model_time.value_in(units.Myr)
        if time_Myr > 1.:
            Lx_t = 10.**( 1.7*np.log10(star_mass) + 30. ) * (time_Myr)**(-2/5) | units.erg / units.s
        else:
            Lx_t = 10.**( 1.7*np.log10(star_mass) + 30. ) | units.erg / units.s
        return Lx_t
        
    @property
    def disk_gas_mass(self):
        '''
        Gas mass of disk (defined as total mass on VADER grid)
        '''
        return (self.code.grid.area*self.code.grid.column_density).sum()


    def evolve_model(self, end_time):
        """Evolve the disk including photoevaporative effects till end_time."""
        # update internal photo-evaporation automatically
        self.code.set_parameter(0, ( self.inner_photoevap_rate ).value_in(units.g/units.s))
        self.code.evolve_model(end_time)

        self.disk.position = self.code.grid.r
        self.disk.surface_gas = self.code.grid.column_density
        self.disk.surface_solid = self.code.grid_user[0].value | units.g / units.cm**2
        self.disk.temperature = self.code.grid_user[2].value | units.K
        self.disk.vd = self.code.grid_user[3].value | units.cm / units.s
        self.disk.st = self.code.grid_user[4].value

        omega = np.sqrt(constants.G * self.star.mass / self.disk.position**3)[:,0]
        self.disk.scale_height = np.sqrt(
            constants.kB * self.disk.temperature / (self.mu * 1.008 * constants.u)
        ) / omega
        self.model_time = self.code.model_time

        # Update temperature analytically, Initially from migration module. # warning: below costs more computational time. 
        # from migration_map_paadekooper import cal_temperature
        # dtgr = self.disk.surface_solid/self.disk.surface_gas
        # temp_d = np.array(cal_temperature(self.disk.position.value_in(units.cm),self.star.mass.value_in(units.g),self.star.radius.value_in(units.cm),
        #                                   self.star_teff.value_in(units.K), self.disk.alpha[0], self.disk.surface_gas.value_in(units.g/units.cm**2), dtgr)) |units.K
        # self.disk.temperature = temp_d
        # pressure = constants.kB*self.disk.temperature*self.disk.surface_gas / (self.mu*1.008*constants.u)

        # self.code.grid.pressure = pressure
        # self.code.grid_user[2].value = temp_d.value_in(units.K)


def _set_disk_params(viscous, alpha, alpha_acc, mu, M_dot_ph_ex, star_mass, v_frag, sigma, temp, fDG):
    """
    Initialise VADER code parameters for disk evolution.
    Args:
        viscous (code):                VADER code instance.
        alpha (float):                 Viscous alpha parameter for disk evolution.
        alpha_acc (float):             Alpha parameter for disk accretion.
        mu (float):                    Mean molecular weight of the disk gas.
        M_dot_ph_ex (units.G0):        External photoevaporation rate.
        star_mass (units.mass):        Mass of the central star.
        v_frag (units.velocity):       Dust fragmentation velocity.
        sigma (units.surface_density): Initial gas surface density profile.
        temp (units.temperature):      Initial temperature profile of the disk.
        fDG (float):                   Dust-to-gas mass ratio.
    Returns:
        Initialised VADER code instance.
    """
    viscous.parameters.alpha = alpha_acc # alpha_acc if disk accretion is mainly wind-driven.
    
    # some fixed parameters
    viscous.parameters.inner_boundary_function = True    
    viscous.parameters.inner_pressure_boundary_type = 1     # fixed mass flux
    viscous.parameters.maximum_tolerated_change = 1e99
    viscous.parameters.post_timestep_function = True
    viscous.parameters.number_of_user_outputs = 5
    viscous.parameters.number_of_user_parameters = 15

    # viscous.set_parameter(0, ( M_dot_ph_in ).value_in(units.g/units.s))
    viscous.set_parameter(1, ( M_dot_ph_ex ).value_in(units.g/units.s))  # DEBUG - Why is this fixed? Should update in model.
    viscous.set_parameter(2, SIGMA_FLOOR.value_in(units.g/units.cm**2))
    viscous.set_parameter(3, SUBL_TEMP.value_in(units.K))
    viscous.set_parameter(4, (mu*1.008*constants.u).value_in(units.g))   # Mean molecular mass in grams
    viscous.set_parameter(5, alpha)
    viscous.set_parameter(6, star_mass.value_in(units.MSun))
    viscous.set_parameter(7, v_frag)                                     # Dust fragmentation velocity [cm/s]
    viscous.set_parameter(8, RHO_DUST.value_in(units.g/units.cm**3))     # Dust internal density
    viscous.set_parameter(9, 1e-3)                                       # DEBUG - Different to Maite?
    viscous.set_parameter(10, CFL)
    viscous.set_parameter(14, alpha_acc)
    
    viscous.grid.column_density = sigma
    viscous.grid.pressure = constants.kB*temp*sigma / (mu*1.008*constants.u)
    viscous.grid_user[0].value = fDG * sigma.value_in(units.g/units.cm**2)  # dust density
    viscous.grid_user[1].value = A0.value_in(units.cm)  # DEBUG: CHANGED THIS - dust initial grain size
    viscous.grid_user[2].value = temp.value_in(units.K)  # temperature profile
    
    return viscous