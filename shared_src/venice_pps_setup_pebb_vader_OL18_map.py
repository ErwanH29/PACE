import numpy as np
from amuse.units import constants, units
from amuse.datamodel import new_regular_grid

import disk_in_clusters.FRIED_interp as FRIED
from shared_src.extra_funcs import sound_speed
from shared_src.module_pebbleaccretion_OL18 import PebbleGasAccretion
from shared_src.module_diskevolution import DiskGasDustEvolution, _set_disk_params
from shared_src.module_migration_map import nonisothermal_Migration
from shared_src.params import A0, SIGMA_FLOOR, DISK_MASS_TO_STAR
from shared_src.venice_src.venice import Venice


interpolator = FRIED.FRIED_interp(
    verbosity=False, 
    data_path="disk_data/friedgrid.dat"
)
    
def _disk_gas_mass(grid):
    '''
    Gas mass of disk (defined as total mass on VADER grid)
    '''
    return (grid.area*grid.column_density).sum()


def _fried_interp(star_mass, fuv_lum, gas_mass, disk_radius):
    """
    Interpolate the FRIED grid to get the photoevaporation mass loss rate.
    Args:
        star_mass (units.mass): Mass of the central star.
        fuv_lum (float): FUV luminosity of the star in units of Lsun.
        gas_mass (units.mass): Mass of the gas in the disk.
        disk_radius (units.length): Radius of the disk.
    Returns:
        Mass loss rate due to photoevaporation (units.mass/units.time).
    """
    return interpolator.interp_amuse(
        star_mass, fuv_lum, gas_mass, disk_radius
        )
    


def setup_single_pps(timestep, ndisk_cells, verbose=False):
    """
    Set up the Venice system. Codes:
    
        0: PebbleGasAccretion       - Handles planetary pebble and gas accretion.
        1: DiskGasDustEvolution     - Handles disk evolution (VADER with pedisk).
        2: nonisothermal_Migration  - Handles non-isothermal Type I migration.
    
    Args:
        timestep (units.time): Timestep for coupling the codes.
        ndisk_cells (int):     Number of cells for disk.
    """
    system = Venice()
    system.verbose = verbose

    # Initialize codes
    pebble_gas_accretion = PebbleGasAccretion(ndisk_cells)
    disk_gas_evolution = DiskGasDustEvolution(ndisk_cells)
    migration = nonisothermal_Migration(timestep, ndisk_cells)

    # Add codes
    system.add_code(pebble_gas_accretion)
    system.add_code(disk_gas_evolution)
    system.add_code(migration)

    # Set coupling timestep; matrix is symmetric, so no need to set [1,0]
    system.timestep_matrix[0,1] = timestep
    system.timestep_matrix[0,2] = timestep
    system.timestep_matrix[1,2] = timestep

    system.add_channel(
        0,2, 
        from_attributes = ['core_mass'], 
        to_attributes = ['core_mass'],
        from_set_name = 'planets', 
        to_set_name = 'planets'
    )

    system.add_channel(
        0,2,
        from_attributes = ['envelope_mass'],
        to_attributes = ['envelope_mass'],
        from_set_name = 'planets',
        to_set_name = 'planets'
    )
    
    system.add_channel(
        1,0,
        from_attributes = ['surface_solid'],
        to_attributes = ['surface_solid'],
        from_set_name = 'disk', 
        to_set_name = 'disk'
    )

    system.add_channel(
        1,0, 
        from_attributes = ['surface_gas'],
        to_attributes = ['surface_gas'],
        from_set_name = 'disk', 
        to_set_name = 'disk'
    )

    system.add_channel(
        1,2, 
        from_attributes = ['surface_solid'], 
        to_attributes = ['surface_solid'], 
        from_set_name = 'disk', 
        to_set_name = 'disk'
    )

    system.add_channel(
        1,2, 
        from_attributes = ['surface_gas'], 
        to_attributes = ['surface_gas'], 
        from_set_name = 'disk', 
        to_set_name = 'disk'
    )

    system.add_channel(
        1,2, 
        from_attributes = ['vd'], 
        to_attributes = ['vd'], 
        from_set_name = 'disk', 
        to_set_name = 'disk'
    )

    system.add_channel(
        1,2, 
        from_attributes = ['st'], 
        to_attributes = ['st'], 
        from_set_name = 'disk', 
        to_set_name = 'disk'
    )

    system.add_channel(
        2,0, 
        from_attributes=['semimajor_axis'], 
        to_attributes=['semimajor_axis'],
        from_set_name='planets', 
        to_set_name='planets'
    )

    return system, pebble_gas_accretion, disk_gas_evolution, migration

def run_single_pps(
    star_mass, Rdisk_in, Rdisk_out, fuv_lum,
    ndisk_cells, fDG, FeH, mu, v_frag, alpha, 
    alpha_acc, gamma, temp1, beta_T, stokes_number, 
    planets, t_birth, dt, times, N_snapshot, filename
    ):
    """
    Run planet population synthesis for a single disk.
    Args:
        star_mass (units.mass):   Mass of the central star.
        Rdisk_in (units.length):  Inner radius of the disk.
        Rdisk_out (units.length): Outer radius of the disk.
        fuv_lum (float):          FUV luminosity of the star in units of Lsun.
        ndisk_cells (int):        Number of cells in disk.
        fDG (float):              Dust-to-gas ratio in the disk.
        FeH (float):              Metallicity of the disk.
        mu (float):               Mean molecular weight of the disk gas.
        v_frag (units.velocity):  Fragmentation velocity of dust grains.
        alpha (float):            Viscosity parameter for disk evolution.
        alpha_acc (float):        Viscosity parameter for accretion.
        gamma (float):            Adiabatic index of the disk gas.
        temp1 (units.temp):       Temperature at 1 au in the disk.
        beta_T (float):           Power-law index for the temperature profile.
        stokes_number (float):    Stokes number of the dust grains.
        planets (Particles):      Initial planet population.
        t_birth (units.time):     Time of planet birth.
        dt (units.time):          Timestep for the simulation.
        times (array):            Array of times at which to record the results.
        N_snapshot (int):         Number of disk snapshots to save.
        filename (str):           Base filename for saving results.
    Returns:
        None. Saves results to files.
    """
    system,_,disk_gas_evolution,_ = setup_single_pps(dt, ndisk_cells)

    # Initialise disk
    code = system.codes[1].code
    code.initialize_keplerian_grid(
        ndisk_cells,      # grid cells
        False,            # True for linear, False for logarithmic
        Rdisk_in,         # inner disk edge
        1000 | units.au,  # outer disk edge - represents maximum possible disk size, but will be truncated to Rdisk_out
        star_mass[0]
    )

    disk_mass = DISK_MASS_TO_STAR * star_mass[0]
    Mdot_initial = _fried_interp(
        star_mass[0], fuv_lum[0], disk_mass, Rdisk_out[0]
    )
    temp = temp1 * (code.grid.r.value_in(units.au))**(-beta_T)
    sigma0 = disk_mass / (2. * np.pi * Rdisk_out[0]**2. * (1. - np.exp(-1.)))
    sigma = sigma0 * Rdisk_out[0] / code.grid.r * np.exp(-code.grid.r/Rdisk_out[0])
    sigma[code.grid.r > Rdisk_out[0]] = SIGMA_FLOOR # sharp edge
    
    code = _set_disk_params(
        code, 
        alpha, 
        alpha_acc, 
        mu, 
        Mdot_initial,   # DEBUG - MDOT_PH_EX SHOULD BE UPDATED
        star_mass[0], 
        v_frag,
        sigma,
        temp,
        fDG
        )

    disk = new_regular_grid(([ndisk_cells]), [1]|units.au)  # DEBUG, WHY 1 AU? SHOULD BE RDISK_OUT?
    disk.position = code.grid.r
    disk.surface_gas = sigma
    disk.temperature = temp
    disk.surface_solid = fDG * sigma * 10**FeH
    disk.scale_height = sound_speed(temp, mu) / np.sqrt(constants.G*star_mass[0]/code.grid.r**3)
    disk.alpha = alpha
    disk.alpha_acc = alpha_acc
    disk.gamma = gamma
    disk.vd = 0. | units.cm/units.s
    disk.st = stokes_number
    const = constants.kB / (mu * 1.008 * constants.u)
    
    system.codes[1].disk = disk
    system.codes[1].star.mass = star_mass[0]

    # initialize pebble accretion (code 0)
    system.codes[0].planets.add_particles(planets)
    system.codes[0].disk = disk
    system.codes[0].star.mass = star_mass[0]

    # initialize planet migration (code 2)
    system.codes[2].planets.add_particles(planets)
    system.codes[2].disk = disk
    system.codes[2].star.mass = star_mass[0]

    ### CURRENT PROGRESS
    
    N_plot_steps = len(times)
    time_arr = np.zeros((N_plot_steps, len(planets))) | units.Myr
    sma_arr = np.zeros((N_plot_steps, len(planets))) | units.au
    planet_mcore_arr = np.zeros((N_plot_steps, len(planets))) | units.MEarth
    planet_menvl_arr = np.zeros((N_plot_steps, len(planets))) | units.MEarth
    disk_gas_to_mstar_arr = np.zeros((N_plot_steps, len(planets)))
    mdot_ext_arr = np.zeros((N_plot_steps, len(planets)))
    
    rdisk_out_arr = []
    gas_arr = []
    solid_arr = []
    disk_time_arr = []
    for i, ti in enumerate(times):
        t_mstar = star_mass[i]
        t_fuv   = fuv_lum[i]
        t_rout  = Rdisk_out[i]
        
        # Synchronise star mass across codes
        system.codes[0].star.mass = t_mstar  # pebble/gas accretion
        system.codes[1].star.mass = t_mstar  # disk inner photoevaporation
        system.codes[2].star.mass = t_mstar  # migration
        system.codes[1].code.set_parameter(
            6, t_mstar.value_in(units.MSun)
        )

        if i == 0:
            prev_mstar = t_mstar
        else:
            rel_mass_change = abs((t_mstar - prev_mstar) / prev_mstar)

            if 1:#rel_mass_change > 1e-7:
                print(f"Updating Keplerian grid for new star mass: {t_mstar.value_in(units.MSun)} Msun")
                system.codes[1].code.update_keplerian_grid(t_mstar)
                prev_mstar = t_mstar

        # Apply truncation
        disk_evol = system.codes[1]
        disk_code = disk_evol.code
        mask = disk_code.grid.r > t_rout
        if np.any(mask):
            T = temp1 * disk_code.grid[mask].r.value_in(units.au)**(-beta_T)
            const = constants.kB / (mu * 1.008 * constants.u)

            # Gas
            disk_code.grid[mask].column_density = SIGMA_FLOOR
            disk_code.grid[mask].pressure = SIGMA_FLOOR * T * const

            # Dust
            disk_code.grid_user[0, mask].value = 1e-14
            disk_code.grid_user[1, mask].value = A0.value_in(units.cm)

            # Sync wrapper disk used by Venice channels
            disk_evol.disk.position = disk_code.grid.r
            disk_evol.disk.surface_gas = disk_code.grid.column_density
            disk_evol.disk.surface_solid = disk_code.grid_user[0].value | units.g / units.cm**2
            disk_evol.disk.temperature = disk_code.grid_user[2].value | units.K
            disk_evol.disk.vd = disk_code.grid_user[3].value | units.cm / units.s
            disk_evol.disk.st = disk_code.grid_user[4].value
            
            # DEBUG - WHAT TO DO WITH PLANET SMA > RDISK_OUT? CURRENTLY LEFT UNCHANGED, BUT MAY CAUSE ISSUES IN MIGRATION MODULE

        # Compute external photoevaporation rate from FRIED grid
        Mdot_ext = _fried_interp(
            t_mstar, 
            t_fuv, 
            _disk_gas_mass(disk_code.grid),
            t_rout
        )
        system.codes[1].code.set_parameter(
            1, (Mdot_ext).value_in(units.g/units.s)
        )
        
        # Evolve system
        if system.codes[1].disk_gas_mass / t_mstar < 5e-5:
            pass
        elif ti < t_birth:  # Planet not yet born
            disk_evol.evolve_model(ti)
            system._sync_code_to_codes(1, [0, 2])
            system.codes[0].model_time = disk_evol.model_time
            system.codes[2].model_time = disk_evol.model_time
            system.model_time = disk_evol.model_time
        elif i==0:
            disk_evol.evolve_model(t_birth)
            system._sync_code_to_codes(1, [0, 2])
            system.codes[0].model_time = disk_evol.model_time
            system.codes[2].model_time = disk_evol.model_time
            system.model_time = disk_evol.model_time
            system.evolve_model(ti)
        elif times[i-1] <= t_birth:
            disk_evol.evolve_model(t_birth)
            system._sync_code_to_codes(1, [0, 2])
            system.codes[0].model_time = disk_evol.model_time
            system.codes[2].model_time = disk_evol.model_time
            system.model_time = disk_evol.model_time
            system.evolve_model(ti)
        else:
            system.evolve_model(ti)

        time_arr[i] = ti
        sma_arr[i] = system.codes[0].planets.semimajor_axis
        planet_mcore_arr[i] = system.codes[0].planets.core_mass
        planet_menvl_arr[i] = system.codes[0].planets.envelope_mass
        disk_gas_to_mstar_arr[i] = system.codes[1].disk_gas_mass / t_mstar
        mdot_ext_arr[i] = Mdot_ext.value_in(units.MSun/units.yr)
        
        rout = system.codes[1].code.grid.r[system.codes[1].code.grid.column_density > SIGMA_FLOOR].max()
        rdisk_out_arr.append(rout.value_in(units.au))
        print(rout.value_in(units.au))

        # control the snapshots of disk
        if i%(int(N_plot_steps/N_snapshot)) == 0:
            disk_time_arr.append(system.codes[0].model_time.value_in(units.kyr))
            solid_arr.append(system.codes[0].disk.surface_solid.value_in(units.g/units.cm**2))  
            gas_arr.append(system.codes[0].disk.surface_gas.value_in(units.g/units.cm**2))
        
    np.savez(
        filename+'_disk.npz',
        time=disk_time_arr,
        position = system.codes[0].disk.position.value_in(units.AU),
        gas=gas_arr,
        solid=solid_arr,
        Rdisk_out=rdisk_out_arr[::int(N_plot_steps/N_snapshot)].value_in(units.au)
        )

    np.savez(
        filename+'_planet.npz', 
        time=time_arr.value_in(units.kyr), 
        Mc=planet_mcore_arr.value_in(units.MEarth), 
        Me=planet_menvl_arr.value_in(units.MEarth), 
        a=sma_arr.value_in(units.au), 
        qdisk=disk_gas_to_mstar_arr, 
        Mdot_ex=mdot_ext_arr, 
        fDG =fDG, 
        FeH=FeH, 
        mu=mu, 
        vfrag = v_frag, 
        alpha=alpha, 
        alpha_acc=alpha_acc, 
        gamma=gamma, 
        temp1 = temp1.value_in(units.K), 
        betaT = beta_T, 
        Rdisk_in = Rdisk_in.value_in(units.au), 
        Rdisk_out = rdisk_out_arr.value_in(units.au), 
        stokes_number = stokes_number, 
        star_mass = star_mass.value_in(units.MSun), 
        t_birth=t_birth.value_in(units.Myr)
        )

    system.stop()
    disk_gas_evolution.code.stop()
    return 0