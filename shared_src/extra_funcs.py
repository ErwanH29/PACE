import numpy as np
from amuse.units import units, constants
from amuse.io import read_set_from_file


pre_dt = 0.1 | units.kyr # timescale for integration
pre_ndisk = 500


def sma_to_period(a, Mstar):
    """
    Convert semi-major axis to orbital period using Kepler's third law.
    Args:
        a: Semi-major axis.
        Mstar: Mass of the central star.
    Returns:
        Orbital period.
    """
    return np.sqrt(a**3/(constants.G*Mstar)) * 2 * np.pi


def period_to_sma(P, Mstar):
    """
    Convert orbital period to semi-major axis using Kepler's third law.
    Args:
        P: Orbital period.
        Mstar: Mass of the central star.
    Returns:
        Semi-major axis.
    """
    period_in_yr = P.value_in(units.yr)
    mstar_in_msun = Mstar.value_in(units.MSun)
    a_in_au = (period_in_yr**2 * mstar_in_msun)**(1/3)
    return a_in_au | units.au


def get_rdisk_out(Mstar):
    """
    Get the outer radius of the disk based on the mass of the star.
    Based on empirical relations of:
        - 2010ApJ...723.1241A
        - 2020MNRAS.494.4130H
        - 2020ApJ...895..126H
        - arXiv:2302.03721
    Args:
        Mstar: Mass of the central star.
    Returns:
        Outer radius of the disk.
    """
    return 117 * (Mstar.value_in(units.MSun))**0.45 | units.au


def get_mdisk(Mstar):
    """
    Get the mass of the disk radius based on the mass of the star.
    Based on empirical relation of:
        - 2010ApJ...723.1241A
        - 2020MNRAS.494.4130H
        - 2020ApJ...895..126H
        - arXiv:2302.03721

    Args:
        Mstar (float):  Mass of star
    Returns:
        Disk mass.
    """
    return 0.24 * (Mstar.value_in(units.MSun))**(0.73) | units.MSun


def Rhills(Mp,Mstar,ap):
    """
    Calculate the Hill radius of a planet.
    Args:
        Mp: Mass of the planet.
        Mstar: Mass of the central star.
        ap: Semi-major axis of the planet.
    Returns:
        Hill radius of the planet.
    """
    return ap*(Mp/3/Mstar)**(1/3)


def Rdisk0(Rdisk_in, Rdisk_out, ndisk):
    """
    Generate initial disk position
    Args:
        Rdisk_in: Inner radius of the disk.
        Rdisk_out: Outer radius of the disk.
        ndisk: Number of disk points.
    Returns:
        Array of disk positions.
    """
    return 10**np.linspace(np.log10(Rdisk_in/(1|units.au)), np.log10(Rdisk_out/(1|units.au)), ndisk) | units.au


def temperature(Rdisk, pT, star_mass):
    """
    Calculate the disk temperature at a given radius.
    Args:
        Rdisk: Radius in the disk.
        pT: Power-law index for temperature profile.
        star_mass: Mass of the central star.
    Returns:
        Temperature at the given radius.
    """
    return (117 | units.K) * (Rdisk/(1|units.au))**pT *(star_mass/(1|units.MSun))


def sound_speed(temperature, mu):
    """
    Calculate the sound speed in the disk.
    Args:
        temperature: Temperature of the disk.
        mu: Mean molecular weight.
    Returns:
        Sound speed in the disk.
    """
    return np.sqrt(constants.kB*temperature/mu/constants.u)


def sigma_g0(fg, pg0, Rdisk, Rdisk_in, Rdisk_out):
    """
    Calculate the initial gas surface density.
    Args:
        fg: Scaling factor for gas surface density.
        pg0: Power-law index for gas surface density profile.
        Rdisk: Radius in the disk.
        Rdisk_in: Inner radius of the disk.
        Rdisk_out: Outer radius of the disk.
    Returns:
        Initial gas surface density at the given radius.
    """
    # typos from Mordasini??
    sigmag_0 = 2400 | units.g/units.cm**2
    sigmag = sigmag_0 * fg* (Rdisk/(1|units.au))**pg0 * np.exp(-(Rdisk/Rdisk_out)**(2-pg0))*(1-np.minimum(np.sqrt(Rdisk_in/Rdisk),1)) #* (np.array((Rdisk/Rdisk_out)**(2-pg0))<10)
    sigmag0 = np.maximum(sigmag.value_in(units.g/units.cm**2), 1e-300)|units.g/units.cm**2
    return sigmag0


def sigma_d0(sigma_g, fDG, FeH, temperature):
    """
    Calculate the initial dust surface density.
    Args:
        sigma_g: Gas surface density.
        fDG: Dust-to-gas ratio.
        FeH: Metallicity.
        temperature: Temperature of the disk.
    Returns:
        Initial dust surface density at the given radius.
    """
    judge = (temperature<(170|units.K))
    eta_ice = 1 # judge*0.75 + 0.25
    return  fDG * 10**FeH * eta_ice* sigma_g


def dynamical_mass(core_mass, envelope_mass):
    """
    Calculate the dynamical mass of a planet.
    Args:
        core_mass: Core mass of the planet.
        envelope_mass: Envelope mass of the planet.
    Returns:
        Dynamical mass of the planet.
    """
    return core_mass + envelope_mass


def get_sequential_indices (i0, i1, folder, dt=0.01|units.Myr):
    """Associate snapshot indices with snapshots."""
    indices = np.arange(i0, i1+1)
    N = len(indices)
    seq_indices_mask = np.ones(N, dtype=bool)
    time = -np.ones(N) | dt.unit
    particles = read_set_from_file(
        folder+'/viscous_particles_plt_i{a:05}.hdf5'.format(a=indices[0]),
        'hdf5')
    try:
        time[0] = particles.get_timestamp()
    except Exception as e:
        time[0] = 0 | dt.unit

    for i in range(N-1):
        particles = read_set_from_file(
            folder+'/viscous_particles_plt_i{a:05}.hdf5'.format(
                a=indices[i+1]), 'hdf5')
        try:
            time[i+1] = particles.get_timestamp()
        except Exception as e:
            time[i+1] = time[i] + dt

        if time[i+1] - time[i] < dt/2.:
            j = i
            while j >= 0 and time[i+1] - time[j] < dt/2.:
                seq_indices_mask[j] = False
                j -= 1
    return indices[seq_indices_mask], time[seq_indices_mask]