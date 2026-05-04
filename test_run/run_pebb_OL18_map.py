"""
Main script to run planet population synthesis using the OL18 pebble accretion model
with external photoevaporation rates from cluster simulations.
"""

import multiprocessing as mp
import numpy as np
import os
from pathlib import Path
import re
import scipy.stats as stats
import sys

from amuse.units import units
from amuse.datamodel import Particles
from amuse.io import read_set_from_file

srcfile = '../shared_src/'
sys.path.append(srcfile)

from extra_funcs import (
    dynamical_mass, get_sequential_indices, Rhills,
    period_to_sma, get_rdisk_out
)
from params import FDG, MU, STOKES_NUMBER, FRAGMENT_V, GAMMA
from venice_pps_setup_pebb_vader_OL18_map import run_single_pps



def main():
    data_output = 'planet_evo/'
    os.makedirs(data_output, exist_ok=True)

    M_dot_exts = [ ]   # | units.MSun/units.yr
    star_ages  = [ ]   # | units.kyr
    star_mass  = [ ]   # | units.MSun

    ### Gather data from cluster simulation outputs
    cluster_snapshots = srcfile+'cluster_data/'
    values = []
    for p in Path(cluster_snapshots).glob('*.hdf5'):
        file_number = re.search(r"(\d+)(?=\.hdf5$)", p.name)
        if file_number:
            values.append(int(file_number.group(1)))

    if len(values) == 0:
        raise ValueError(f"No cluster data found in {cluster_snapshots}.")

    i0 = min(values)
    i1 = max(values)

    bridge_dt = None
    try:
        with open(os.path.join(srcfile, "cluster_data", "bridge_step.txt"), "r") as f:
            bridge_dt = float(f.read().strip()) | units.yr
    except Exception as e:
        print(f"Warning: Could not read bridge_step.txt. Defaulting to 0.01 Myr. Error: {e}")
        bridge_dt = 0.01 | units.Myr
    
    disk_keys  = [ ]

    indexs, _ = get_sequential_indices(
        i0, i1, 
        cluster_snapshots, 
        dt=bridge_dt
        )
    for i in indexs:
        filename = os.path.join(srcfile, "cluster_data", f"viscous_particles_plt_i{i:05d}.hdf5")
        if not os.path.exists(filename):
            continue

        env_data = read_set_from_file(filename, "hdf5")
        all_disk_keys = env_data.disk_key
        for disk_key in all_disk_keys:
            if disk_key == -1:
                continue
                
            mask = all_disk_keys == disk_key
            mdot = env_data.outer_photoevap_rate[mask][0].value_in(units.MSun/units.yr)
            age  = env_data.age[mask][0].value_in(units.kyr)
            if disk_key in disk_keys:
                M_dot_exts[disk_keys.index(disk_key)].append(mdot)
                star_ages[disk_keys.index(disk_key)].append(age)
            else:
                disk_keys.append(disk_key)
                star_mass.append(env_data.mass[mask][0].value_in(units.MSun))
                M_dot_exts.append([mdot])
                star_ages.append([age])

    print(M_dot_exts[disk_keys.index(disk_key)])
    S
    ndisks = len(disk_keys)

    # Generate or load random disk parameters
    random_IC_file = '../random_IC_file.npz'
    if os.path.exists(random_IC_file):
        print('Loading parameters...')
        with np.load(random_IC_file) as df:
            FeH_rand       = df['FeH_rand']
            alpha_rand     = df['alpha_rand']
            alpha_acc_rand = df['alpha_acc_rand']
            rdisk_inner    = df['rdisk_inner']
            rdisk_outer    = df['rdisk_outer']
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

        # compute disk inner and outer radii
        rdisk_inner = period_to_sma(
            Pdisk_in_rand | units.day, 
            np.array(star_mass) | units.MSun
            ).value_in(units.au)
        rdisk_outer = get_rdisk_out(
            np.array(star_mass) | units.MSun
            ).value_in(units.au)

        ## random conditions:
        alpha_rand     = 1e-4 * np.ones(ndisks)
        alpha_acc_rand = 1e-3 * np.ones(ndisks)
        pmass_rand     = 1e-2 * np.ones(ndisks)
        tbirth_rand    = np.random.uniform(0, 0.5, ndisks)  # planet birth time
        beta_L         = 2  # Slope of mass-luminosity relation for protostars.

        psma_in = period_to_sma(
            100 | units.day, 
            np.array(star_mass) | units.MSun
        ).value_in(units.au)
        psma_out = rdisk_outer
        
        log_sma_value = np.random.uniform(np.log10(psma_in), np.log10(psma_out))
        psma_rand = 10**log_sma_value

        # some fixed parameters
        # beta_T  = 3/7 # warning, does not have minors sign
        fDG = FDG   # solar dust to gas ratio
        mu = MU      # mean molecular weight
        beta_T = 0.5
        stokes_number = STOKES_NUMBER
        v_frag = FRAGMENT_V
        gamma  = GAMMA

        print('Generating parameters...')
        np.savez(
            random_IC_file, 
            FeH_rand=FeH_rand, 
            alpha_rand=alpha_rand, 
            alpha_acc_rand=alpha_acc_rand, 
            rdisk_inner=rdisk_inner, 
            rdisk_outer=rdisk_outer, 
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

    tbirth_rand = 0. * np.ones(ndisks) # unit: Myr

    # Set up multiprocessing pool
    USE_SLURM = 0
    if USE_SLURM:
        n_jobs = os.environ['SLURM_JOB_CPUS_PER_NODE'] // 2
    else:
        n_jobs = os.cpu_count() // 2  # Leave some cores free. Each process requires 2 cores.
    print(f"Number of processes being used in parallel: {n_jobs}")

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=int(n_jobs))
    results=[]
    for i in range(ndisks):
        filename = data_output +'%06d'%i
        if ((os.path.exists(filename+'_planet.npz'))*(os.path.exists(filename+'_disk.npz'))==1):
            print(filename, 'exist, continue!')
            continue

        M_star    = star_mass[i] | units.MSun
        FeH       = FeH_rand[i]    # initial metalicity, default: 0
        alpha     = alpha_rand[i]  # alpha parameter, default: 2e-3
        alpha_acc = alpha_acc_rand[i]
        Rdisk_in  = rdisk_inner[i]  | units.AU
        Rdisk_out = rdisk_outer[i] | units.AU
        t_birth   = tbirth_rand[i] | units.Myr

        ### Initialise planet embryos. Assume start at some Hill-radii separation
        Nplanets = np.random.randint(1, 12)  # number of planets
        planet_masses = [pmass_rand[i] for _ in range(Nplanets)] | units.MEarth
        
        psma_in = period_to_sma(1 | units.yr, M_star)
        random  = np.random.uniform(1,10,1)
        psma_in *= random
        hill_sep = np.random.uniform(5, 10, 1)

        embryo_separations = [ ]
        for planet in range(Nplanets):
            if Nplanets == 1:
                sma = np.random.uniform(psma_in.value_in(units.au), Rdisk_out.value_in(units.au))
                embryo_separations.append(sma)
            elif planet == 0:
                sma = psma_in
                embryo_separations.append(sma.value_in(units.AU))
            else:
                Rhill = Rhills(
                    Mp=planet_masses[planet],
                    Mstar=M_star,
                    ap=embryo_separations[-1] | units.AU
                )
                new_min = hill_sep * Rhill.value_in(units.au) + embryo_separations[-1]
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

        dt = bridge_dt / 10.
        N_plot_disk = 20 # number of saved snapshots

        print(
            'fDG:', fDG, 
            'FeH:', FeH, 
            'alpha:', alpha, 
            'alpha_acc:', alpha_acc, 
            'gamma:', gamma, 
            'temp1:', temp1.value_in(units.K), 
            'betaT:', beta_T, 
            'R_in:', rdisk_inner[i], 
            'R_out:', rdisk_outer[i], 
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
    main()
