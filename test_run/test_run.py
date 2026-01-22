"""Small script used for testing purposes"""


import multiprocessing as mp
import numpy as np
import os
import scipy.stats as stats
import sys

from amuse.units import units
from amuse.datamodel import Particles

srcfile = '../shared_src/'
sys.path.append(srcfile)

from extra_funcs import dynamical_mass, period_to_sma, get_rdisk_out, Rhills
from venice_pps_setup_pebb_vader_OL18_map import run_single_pps


np.random.seed(12345678)
datafile = 'planet_evo/'
os.makedirs(datafile, exist_ok=True)

n_times = 100
disk_keys  = [1, 2, 3]
M_dot_exts = [1e-10*np.ones([n_times]) for _ in range(len(disk_keys))]     # | units.MSun/units.yr
star_ages  = [np.linspace(0,2000,n_times) for _ in range(len(disk_keys))]   # | units.kyr
star_mass  = [1 for _ in range(len(disk_keys))]                        # | units.MSun

ndisks = len(disk_keys)

random_IC_file = '../test_random_par.npz'
if os.path.exists(random_IC_file):
    print('Loading parameters...')
    with np.load(random_IC_file) as df:
        FeH_rand       = df['FeH_rand']
        alpha_rand     = df['alpha_rand']
        alpha_acc_rand = df['alpha_acc_rand']
        Rdisk_in_rand  = df['Rdisk_in_rand']
        Rdisk_out_rand = df['Rdisk_out_rand']
        tbirth_rand    = df['tbirth_rand']
        beta_L         = df['beta_L']
        pmass_rand     = df['pmass_rand']
        psma_rand      = df['psma_rand']
        fDG            = df['fDG']
        mu             = df['mu']
        beta_T         = df['beta_T']
        stokes_number  = df['stokes_number']
        v_frag         = df['v_frag']
        gamma          = df['gamma']
else:
    print('Generating random parameters...')
    FeH_rand = 0.02 * np.ones(ndisks)

    # Batygin et al. 2023
    lower, upper = np.log10(0.1), np.log10(100)
    mu, sigma = np.log10(4), 0.5
    logPdisk_in_rand_cal = stats.truncnorm(
        (lower - mu)/sigma, 
        (upper - mu)/sigma, 
        loc=mu, 
        scale=sigma
        )
    Pdisk_in_rand = 10**logPdisk_in_rand_cal.rvs(ndisks)   # | units.days

    Rdisk_in_rand = period_to_sma(
        Pdisk_in_rand | units.day, 
        np.array(star_mass) | units.MSun
        ).value_in(units.au)
    Rdisk_out_rand = get_rdisk_out(np.array(star_mass) | units.MSun).value_in(units.au)

    ## random conditions:
    alpha_rand = 1e-4 * np.ones(ndisks)
    alpha_acc_rand = 1e-3 * np.ones(ndisks)
    pmass_rand = 1e-2 * np.ones(ndisks)
    tbirth_rand = np.random.uniform(0,0.5,ndisks)
    beta_L = 2 # mass slope of mass-luminosity relation for proto stars.

    # psma_in = np.array(star_mass)**((2*beta_L-1)/3) # outside iceline
    psma_in = period_to_sma(
        100 | units.day, 
        np.array(star_mass) | units.MSun
    ).value_in(units.au)
    psma_out = Rdisk_out_rand
    
    log_sma_value = np.random.uniform(np.log10(psma_in), np.log10(psma_out))
    psma_rand = 10**log_sma_value

    # some fixed parameters
    fDG = 0.0149  # solar dust to gas ratio
    mu  = 2.3      # mean molecular weight
    # beta_T  = 3/7 # warning, does not have minors sign
    beta_T = 0.5
    stokes_number = 0.001
    v_frag = 1e3  # | units.cms
    gamma  = 7/5

    print(f'Outputting to {random_IC_file}...')
    np.savez(
        random_IC_file, 
        FeH_rand=FeH_rand, 
        alpha_rand=alpha_rand, 
        alpha_acc_rand=alpha_acc_rand, 
        Rdisk_in_rand=Rdisk_in_rand, 
        Rdisk_out_rand=Rdisk_out_rand, 
        tbirth_rand=tbirth_rand, 
        pmass_rand=pmass_rand, 
        psma_rand=psma_rand, 
        fDG=fDG, 
        mu=mu, 
        beta_T=beta_T, 
        beta_L=beta_L, 
        stokes_number=stokes_number,
        v_frag=v_frag, 
        gamma=gamma
    )

def main():
    USE_SLURM = 0
    HILL_SEP  = 20
    tbirth_rand = 0.*np.ones(ndisks) # unit: Myr
    n_jobs = 1
    print(f"Number of processes being used in parallel: {n_jobs}")

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=int(n_jobs )) # number of processes <---------------
    results=[]
    for i in range(ndisks):
        filename = datafile +'%06d'%i
        if ((os.path.exists(filename+'_planet.npz'))*(os.path.exists(filename+'_disk.npz'))==1):
            print(filename, 'exist, continue!')
            continue

        M_star    = star_mass[i] | units.MSun
        FeH       = FeH_rand[i]    # initial metalicity, default: 0
        alpha     = alpha_rand[i]  # alpha plarameter, default: 2e-3
        alpha_acc = alpha_acc_rand[i]
        Rdisk_in  = Rdisk_in_rand[i]  | units.AU
        Rdisk_out = Rdisk_out_rand[i] | units.AU
        t_birth   = tbirth_rand[i] | units.Myr

        ### Initialise planet embryos. Assume start at some Hill-radii separation
        Nplanets = np.random.randint(1, 12)  # number of planets
        planet_masses = [pmass_rand[i] for _ in range(Nplanets)] | units.MEarth
        embryo_separations = [ ]
        for planet in range(Nplanets):
            if planet == 0:
                sma = HILL_SEP * Rdisk_in
                embryo_separations.append(sma.value_in(units.AU))
            else:
                Rhill = Rhills(
                    Mp=planet_masses[planet],
                    Mstar=M_star,
                    ap=embryo_separations[-1] | units.AU
                )
                new_min = HILL_SEP * Rhill.value_in(units.au) + embryo_separations[-1]
                new_sma = np.random.uniform(new_min, Rdisk_out.value_in(units.au))  # | units.AU
                if new_sma >= Rdisk_out.value_in(units.au):
                    print(f"Reached disk outer edge at planet {planet}, stopping embryo placement.")
                    break
                embryo_separations.append(new_sma)

        Nplanets = len(embryo_separations)
        embryo_separations = np.array(embryo_separations) | units.AU
        planet_masses = planet_masses[:Nplanets]
        print(f"Disk {i}: Initialised {Nplanets} planets.")

        planets = Particles(Nplanets,
            core_mass=planet_masses,
            envelope_mass = 0|units.g,
            semimajor_axis = embryo_separations,
            isohist = False # it will become true when planet first reach pebble isolation mass
        )
        planets.add_calculated_attribute('dynamical_mass', dynamical_mass)

        
        # temp1 = 150*star_mass[i]**((2*beta_L-1)/7) | units.K
        temp1 = 150. * (M_star.value_in(units.MSun))**(1/4) | units.K
        M_dot_ph_ex = np.array(M_dot_exts[i]) | units.MSun/units.yr
        times = np.array(star_ages[i]) | units.kyr
        if len(M_dot_ph_ex) != len(times):
            raise ValueError('error: error with reading cluster data!')

        dt = 1 | units.kyr # timestep of the matrix (fixed)
        N_plot_disk = 20 # number of saved snapshots

        print(
            'fDG:', fDG, 
            'FeH:', FeH, 
            'alpha:', alpha, 
            'alpha_acc:', alpha_acc, 
            'gamma:', gamma, 
            'temp1:', temp1.value_in(units.K), 
            'betaT:', beta_T, 
            'R_in:', Rdisk_in_rand[i], 
            'R_out:', Rdisk_out_rand[i], 
            'St:', stokes_number, 
            'star_mass:', star_mass[i], 
            't_birth:', tbirth_rand[i], 
            'psma:', psma_rand[i]
            )
        argL = (
            fDG, FeH, mu, v_frag, alpha, 
            alpha_acc, gamma, temp1, beta_T, 
            Rdisk_in, Rdisk_out, stokes_number, 
            planets, M_star, M_dot_ph_ex, t_birth, 
            dt, times, N_plot_disk, filename
            )
        results.append (pool.apply_async(run_single_pps, argL))

    # clean up
    pool.close()
    pool.join()

    planet_results = np.array([i.get() for i in results])
    print(planet_results)
    
if __name__ == "__main__":
    mp.freeze_support()  # harmless on Linux; required on Windows/frozen apps
    main()