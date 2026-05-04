import numpy as np
from amuse.lab import units

# Power-law fit derived from UVBLUE spectra (z=0.0122, DOI:10.1086/429858)
DEFAULT_FIT_FILE = 'ML_fit.txt'


def load_fuv_fit(file):
    """
    Load the power law fit parameters for the FUV luminosity from the given file.
    
    Args:
        file (str): Path to the file containing the fit parameters.
    Returns:
        tuple: Tuple containing the A and B coefficients, as well as mass grid.
    """
    A, B, mass_grid = np.loadtxt(file+"/ML_fit.txt", unpack=True)
    return A, B, mass_grid


def fuv_luminosity_from_mass(
    M, 
    A=None, 
    B=None, 
    mass=None, 
    file=DEFAULT_FIT_FILE
    ):
    """
    Compute the FUV-luminosity from stellar mass. 
    
    Args:
        M (float):     Masses of Stars
        A (float):     A coefficient.
        B (float):     B coefficient.
        mass (float):  Mass grid in MSun.
    Returns:
        units.LSun: far-UV luminosity of star
    """
    if A is None or B is None or mass is None:
        A, B, mass = load_fuv_fit(file)

    m = M.value_in(units.MSun)
    if m < mass[0]:
        return 10.**(A[0] * np.log10(m) + B[0]) | units.LSun
    elif m > mass[-1]:
        return 10.**(A[-1] * np.log10(mass[-1]) + B[-1]) | units.LSun

    idx = np.argmax(mass > m)-1
    return 10.**(A[idx] * np.log10(m) + B[idx]) | units.LSun


def fuv_luminosity_from_masses(mass, file=DEFAULT_FIT_FILE):
    '''
    Compute the FUV-luminosity from stellar mass.
    
    Args:
        mass (float):  Stellar masses.
        file (str):    Path to the file containing the fit parameters.
    Returns:
        array: Far-UV luminosity of star
    '''
    A, B, mass_grid = load_fuv_fit(file)
    lfuv = [] | units.LSun
    for mi in mass:
        lfuv.append(fuv_luminosity_from_mass(mi, A=A, B=B, mass=mass_grid))

    return lfuv