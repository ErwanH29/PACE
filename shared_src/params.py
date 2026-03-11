"""Script containing physical and astronomical constants in cgs units."""

# -*- coding: utf-8 -*
import numpy as np

from amuse.units import constants, units

G         = constants.G.value_in(units.cm**3/units.g/units.s**2) # Newton’s gravitational constant
mSun      = (1 | units.MSun).value_in(units.g)
mEarth    = (1 | units.MEarth).value_in(units.g)
mElectron = constants.electron_mass.value_in(units.g)
mProton   = constants.proton_mass.value_in(units.g)
mNeutron  = constants.neutron_mass.value_in(units.g)
mAtom     = constants.unified_atomic_mass_unit.value_in(units.g)

rSun   = (1 | units.RSun).value_in(units.cm)
rEarth = (1 | units.REarth).value_in(units.cm)
rJup   = (1 | units.RJupiter).value_in(units.cm)
au     = (1 | units.au).value_in(units.cm)
yr     = (1 | units.yr).value_in(units.s)
day    = (1 | units.day).value_in(units.s)
GM     = G * (mSun + mEarth)

h_Plank  = 2*np.pi * (constants.Planck_constant_over_2_pi.value_in(units.g*units.cm**2/units.s))
k_SB     = constants.kB.value_in(units.erg/units.K)
sigma_SB = (constants.Stefan_hyphen_Boltzmann_constant.value_in(units.erg/(units.cm**2*units.s**1*units.K**4))) # boltzman constant
L_Sun    = (1 | units.LSun).value_in(units.erg/units.s)
lyr      = (1 | units.lightyear).value_in(units.cm)
pc       = (1 | units.parsec).value_in(units.cm)

H0 = 72 * 1e5/1e6/pc

label = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
mass = [0.09*mSun, 1.374*mEarth, 1.308*mEarth, 0.388*mEarth, 0.692*mEarth, 1.039*mEarth, 1.321*mEarth, 0.326*mEarth] # modeling planet masses. Credit: Agol+pre
radii = [0.,1.116,1.097,0.788,0.920,1.045,1.129,0.755] #observational planet radii. Credit: DucrotEtal2020.
