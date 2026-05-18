"""
Main script to run planet population synthesis using the OL18 pebble accretion model.
The script loads information on the star phase-space distribution, outer-disk edge
and radiative field. These are used as inputs to enable VADER evolves with the
background environment in consideration.

The planet embryos are assumed to start with 1% the Earth's mass. Additionally,
their multiplicity and initial placements are randomly sampled. For the latter,
a randomly chosen value between five and ten is taken and adjacent planets are
separated by that factor of their mutual Hill radius.
"""

import multiprocessing as mp
import numpy as np
import os
from pathlib import Path
import re
import traceback

from amuse.units import units
from amuse.datamodel import Particles
from amuse.io import read_set_from_file

from shared_src.extra_funcs import (
    dynamical_mass, get_sequential_indices, Rhills,
    period_to_sma, get_disk_inner_edge
)
from shared_src.params import (
    ALPHA, EMBRYO_MASS, FDG, FRAGMENT_V, 
    GAMMA, G0, MASS_MAX, MASS_MIN, METALLICITY, 
    MU, STOKES_NUMBER, PACE_DT, BRIDGE_DT
)
from shared_src.venice_pps_setup_pebb_vader_OL18_map import run_single_pps


def _generate_planet_sma(sma_inner, sma_outer, Mstar, Mplanets):
    """
    Generate planetary embryo initial semi-major axis.
    Args:
        sma_inner (units.length):  The inner disk edge
        sma_outer (units.length):  The outer disk edge
        Mstar (units.mass):        Host star's mass
        Mplanets (units.mass):     Planetary embryo mass array
    Returns:
        Array with initial planetary embryo semi-major axis.
    """
    embryo_sma = [ ]
    for i in range(len(Mplanets)):
        hill_sep = np.random.uniform(5, 10, 1)
        if len(Mplanets) == 1:
            sma = np.random.uniform(
                sma_inner.value_in(units.au), 
                0.9 * sma_outer
                )
            embryo_sma.append(sma)
        elif i == 0:
            sma = np.random.uniform(
                sma_inner.value_in(units.au), 
                0.9 * sma_outer
                )
            embryo_sma.append(sma)
        else:
            Rhill = Rhills(
                Mp=Mplanets[i],
                Mstar=Mstar,
                ap=embryo_sma[-1] | units.AU
            )
            sma_inner = embryo_sma[-1] + hill_sep * Rhill.value_in(units.au)
            new_sma = np.random.uniform(sma_inner, sma_outer)  # | units.AU
            if new_sma >= 0.9 * sma_outer:
                print(f"Reached disk outer edge at planet {i}, stopping embryo placement.")
                break
            embryo_sma.append(new_sma)
            
    return embryo_sma


def _generate_IC(star_mass, ndisks, Rdisk_out, random_IC_file, inner_edge=None):
    """
    Generate random parameters for planet population synthesis. These include:
    - Metallicity (FeH)
    - Viscosity parameter (alpha)
    - Accretion viscosity parameter (alpha_acc)
    - Inner disk edge (rdisk_inner)
    - Planet birth time (tbirth_rand)
    - Planet mass (planet_mass)
    - Initial planetary semi-major axis (planet_sma)
    - Temperature normalization (temp1)
    - Slope of temperature profile (beta_T)
    - Slope of mass-luminosity relation for protostars (beta_L)
    Args:
        star_mass (array):        Array of stellar masses.
        ndisks (int):             Number of disks to generate parameters for.
        Rdisk_out (array):        Array of outer disk edges.
        random_IC_file (str):     Path to save the generated parameters.
        inner_edge (units.time):  Inner edge of the disk. Optional.
    """
    print('Generating random parameters...')
    star_mass_arr = star_mass | units.MSun
    FeH_rand = METALLICITY * np.ones(ndisks)
    rdisk_inner = get_disk_inner_edge(
        star_mass_arr[:,0], ndisks=ndisks
        )

    # Random conditions:
    alpha_rand = 1e-4 * np.ones(ndisks)
    alpha_acc_rand = ALPHA * np.ones(ndisks)
    planet_mass = (EMBRYO_MASS.value_in(units.MEarth)) * np.ones(ndisks)
    tbirth_rand = np.random.uniform(0, 0.5, ndisks)  # planet birth time
    beta_T = 0.5
    beta_L = 2.0  # Slope of mass-luminosity relation for protostars.
    if inner_edge is not None:
        inner_edge = period_to_sma(
            inner_edge, star_mass_arr
            ).value_in(units.au)
    else:
        inner_edge = rdisk_inner

    log_sma_value = np.random.uniform(
        np.log10(inner_edge), 
        np.log10(Rdisk_out[:,0])
        )
    planet_sma = 10**log_sma_value

    print('Saving parameters...')
    np.savez(
        random_IC_file, 
        FeH_rand=FeH_rand, 
        alpha_rand=alpha_rand, 
        alpha_acc_rand=alpha_acc_rand, 
        rdisk_inner=rdisk_inner,
        tbirth_rand=tbirth_rand, 
        planet_mass=planet_mass, 
        planet_sma=planet_sma, 
        beta_T=beta_T, 
        beta_L=beta_L, 
        fDG=FDG, 
        mu=MU, 
        stokes_number=STOKES_NUMBER,
        v_frag=FRAGMENT_V, 
        gamma=GAMMA
    )


def _get_star_data(indexs):
    """
    Store data on cluster  configuraiton in arrays.
    Args:
        indexs (int):  Integer values of snapshot
    """
    disk_keys  = [ ]
    Rdisk_out  = [ ]   # | units.au
    fuv_lum    = [ ]   # | units.G0
    star_ages  = [ ]   # | units.kyr
    star_mass  = [ ]   # | units.MSun
    
    invalid_keys = set()
    snapshots_read = 0
    for i in indexs:
        filename = os.path.join(
            'shared_src', "cluster_data", f"viscous_particles_plt_i{i:05d}.hdf5"
            )
        if not os.path.exists(filename):
            continue

        snapshots_read += 1
        star_snapshot = read_set_from_file(filename, "hdf5")[:2]  # DEBUG
        if snapshots_read == 1:
            KE = star_snapshot.kinetic_energy()
            PE = star_snapshot.potential_energy()
            Qvir = abs(KE / PE)
            Rvir = star_snapshot.virial_radius().value_in(units.pc)
            print(
                f"Cluster initial conditions: Qvir={Qvir:.3f}, "
                f"Rvir={Rvir:.3f} pc, Nstars={len(star_snapshot)}"
                )
            
        for host in star_snapshot:
            if host.mass >= MASS_MAX:
                continue
            elif host.mass <= MASS_MIN:
                continue
            
            key = host.key
            if key in invalid_keys:
                continue

            rout = host.Rout.value_in(units.au)
            if rout <= 0.0:  # No disk at end, remove all references
                if key in disk_keys:
                    print(f"Disk around star {key} disappeared at snapshot {i}, removing from dataset.")
                    invalid_keys.add(key)
                continue

            fuv = host.fuv_ambient_flux.value_in(G0)
            age = host.age[0].value_in(units.kyr)
            mass = host.mass.value_in(units.MSun)
            if key in disk_keys:
                j = disk_keys.index(key)
                Rdisk_out[j].append(rout)
                fuv_lum[j].append(fuv)
                star_ages[j].append(age)
                star_mass[j].append(mass)
            else:
                disk_keys.append(key)
                Rdisk_out.append([rout])
                fuv_lum.append([fuv])
                star_ages.append([age])
                star_mass.append([mass])
    
    valid_keys = []
    for j, key in enumerate(disk_keys):
        n_entries = len(Rdisk_out[j])

        if key in invalid_keys:
            continue
        if n_entries != snapshots_read:
            print(f"Invalid Star: Star {key} has {n_entries} entries but expected {snapshots_read}.")
            continue
        
        valid_keys.append(j)
    
    disk_keys = [disk_keys[j] for j in valid_keys]
    Rdisk_out = np.array([Rdisk_out[j] for j in valid_keys])
    fuv_lum   = np.array([fuv_lum[j]   for j in valid_keys])
    star_ages = np.array([star_ages[j] for j in valid_keys])
    star_mass = np.array([star_mass[j] for j in valid_keys])

    return len(disk_keys), Rdisk_out, fuv_lum, star_ages, star_mass


def main(ndisk_cells=500):
    """Driver script to run planet population synthesis."""
    data_output = 'planet_evo/'
    os.makedirs(data_output, exist_ok=True)

    # Gather data from cluster simulation outputs
    cluster_snapshots = 'shared_src/cluster_data/'
    values = []
    for p in Path(cluster_snapshots).glob('*.hdf5'):
        file_number = re.search(r"(\d+)(?=\.hdf5$)", p.name)
        if file_number:
            values.append(int(file_number.group(1)))

    if len(values) == 0:
        raise ValueError(f"No cluster data found in {cluster_snapshots}.")

    i0 = min(values)
    i1 = max(values)
    
    indexs, _ = get_sequential_indices(
        i0, i1, 
        cluster_snapshots, 
        dt=BRIDGE_DT
        )
    ndisks, Rdisk_out, fuv_lum, star_ages, star_mass = _get_star_data(indexs)

    PACE_NSNAPS = 20

    # Generate random disk parameters or load them
    random_IC_file = '../random_IC_file.npz'
    if not os.path.exists(random_IC_file):
        _generate_IC(
            star_mass=star_mass,
            ndisks=ndisks,
            Rdisk_out=Rdisk_out,
            random_IC_file=random_IC_file
        )

    print('Loading parameters...')
    with np.load(random_IC_file) as df:
        FeH_rand       = df['FeH_rand']
        alpha_rand     = df['alpha_rand']
        alpha_acc_rand = df['alpha_acc_rand']
        rdisk_inner    = df['rdisk_inner']
        tbirth_rand    = df['tbirth_rand']
        beta_L         = df['beta_L']
        planet_mass    = df['planet_mass']
        planet_sma     = df['planet_sma']
        fDG            = df['fDG']
        mu             = df['mu']
        beta_T         = df['beta_T']
        stokes_number  = df['stokes_number']
        v_frag         = df['v_frag']
        gamma          = df['gamma']

    tbirth_rand = 0. * np.ones(ndisks)  # | units.Myr

    # Set up multiprocessing pool
    USE_SLURM = 0
    if USE_SLURM:
        n_jobs = os.environ['SLURM_JOB_CPUS_PER_NODE']
    else:
        n_jobs = min(os.cpu_count() // 2, ndisks // 2)  # Each process requires 2 cores.
    print(f"Number of processes being used in parallel: {n_jobs}")
    print(f"Number of disks to simulate: {ndisks}")

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(processes=int(n_jobs))
    results = [ ]
    for i in range(ndisks):
        filename = data_output +'%06d'%i
        if ((os.path.exists(filename+'_planet.npz'))*(os.path.exists(filename+'_disk.npz'))==1):
            print(filename, 'exist, continue!')
            continue
        
        try:
            # Load properties
            FeH = FeH_rand[i]
            alpha = alpha_rand[i]
            alpha_acc = alpha_acc_rand[i]
            Rdisk_in = rdisk_inner[i]  | units.AU
            t_birth = tbirth_rand[i] | units.Myr

            # Initialise planet embryos
            Nplanets = np.random.randint(1, 10)
            planet_masses = [planet_mass[i] for _ in range(Nplanets)] | units.MEarth
            
            random  = np.random.uniform(2, 10, 1)
            sma_inner = random * Rdisk_in
            sma_outer = Rdisk_out[i,0]
            embryo_sma = _generate_planet_sma(
                sma_inner, sma_outer,
                Mstar=star_mass[i][0] | units.MSun,
                Mplanets=planet_masses
                )

            Nplanets = len(embryo_sma)
            embryo_sma = np.array(embryo_sma) | units.AU
            planet_masses = planet_masses[:Nplanets]
            print(f"Disk {i}: Initialised {Nplanets} planets.")

            planets = Particles(
                Nplanets,
                core_mass=planet_masses,
                envelope_mass = 0|units.g,
                semimajor_axis = embryo_sma,
                isohist = False # it will become true when planet first reach pebble isolation mass
            )
            planets.add_calculated_attribute('dynamical_mass', dynamical_mass)

            # temp1 = 150*star_mass[i]**((2*beta_L-1)/7) | units.K
            temp1 = 150. * (star_mass[i][0])**(1/4) | units.K  # Taken from arXiv:2105.13101

            times = np.array(star_ages[i]) | units.kyr
            masses = np.array(star_mass[i]) | units.MSun
            ext_fuv = np.array(fuv_lum[i]) | G0
            disk_rin = np.array(rdisk_inner[i]) | units.au
            disk_rout = np.array(Rdisk_out[i]) | units.au

            print(
                f'fDG: {fDG:.3f}, ',
                f'\nFeH: {FeH:.3f}, ',
                f'\nalpha: {alpha:.3f}, ',
                f'\nalpha_acc: {alpha_acc:.3f}, ',
                f'\ngamma: {gamma:.3f}, ',
                f'\ntemp1: {temp1.value_in(units.K):.3f} K, ',
                f'\nbetaT: {beta_T:.3f}, ',
                f'\nR_in: {rdisk_inner[i]:.3f} au, ',
                f'\nR_out (initial): {Rdisk_out[i][0]:.3f} au, ',
                f'\nSt: {stokes_number}, ',
                f'\nstar_mass (initial): {star_mass[i][0]:.3f} MSun, ',
                f'\nt_birth: {tbirth_rand[i]:.3f} Myr, ',
                f'\npsma: {planet_sma[i]:.3f}\n',
                "=" * 50,
                )
            argL = (
                masses, disk_rin, disk_rout, ext_fuv, ndisk_cells,
                fDG, FeH, mu, v_frag, alpha, alpha_acc, gamma, temp1, 
                beta_T, stokes_number, planets, t_birth, PACE_DT, 
                times, PACE_NSNAPS, filename
                )
            results.append(pool.apply_async(run_single_pps, argL))
        except Exception:
            print(f"\nFAILED BEFORE SUBMITTING DISK {i}\n")
            traceback.print_exc()

    # clean up
    pool.close()
    for r in results:
        r.get()
    pool.join()

    

if __name__ == "__main__":
    main()
