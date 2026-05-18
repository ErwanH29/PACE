"""
Cluster evolution tracking the radiative background on stars and 
impact of dynamical truncations. 

A geometric approach is taken for computing radiative effects, which 
is shown to enhance effects (ignoring shielding via dust extinction, 
see arXiv:2302.03721).

The prescription for dynamical truncation is taken from arXiv:1403.8099
and only computes the strongest interaction if the distance between the 
host and the perturbing star is less than twice the crossing distance in
a bridge time-step.

Stars are assumed to be born all at the same epoch. For a molecular cloud
collapse, this script will need changes (inclusion of sink particle) and
changing of stellar attribute 'age'.
"""

import numpy as np
import os

from amuse.community.ph4.interface import Ph4
from amuse.community.seba.interface import SeBa
from amuse.ext.orbital_elements import orbital_elements
from amuse.lab import (
    constants, nbody_system, Particles, units,
    read_set_from_file, write_set_to_file,
)

from shared_src.extra_funcs import get_disk_outer_edge
from shared_src.params import MASS_MIN, MASS_MAX
from disk_in_clusters.fuv_luminosity import fuv_luminosity_from_masses


def ZAMS_radius(star_mass):
    """
    Get stellar radius assuming zero-age MS star.
    Args:
        star_mass (units.mass):  Mass of star.
    Returns:
        units.length:  The ZAMS radius of the star.
    """
    mass_in_sun = star_mass.value_in(units.MSun)
    mass_sq = (mass_in_sun)**2.

    numerator = mass_in_sun**1.25 * (0.1148 + 0.8604 * mass_sq)
    denominator = (0.04651 + mass_sq)
    r_zams = numerator / denominator

    return r_zams | units.RSun


def _apply_truncation(rperi, mass_a, mass_b):
    """Apply the truncation prescription from arXiv:1403.8099."""
    rtrunc_a = 0.28 * rperi * (mass_a / mass_b) ** 0.32
    rtrunc_b = 0.28 * rperi * (mass_b / mass_a) ** 0.32
    return rtrunc_a, rtrunc_b


def _get_new_radius(body, max_mass):
    """Get new radius for star after truncation."""
    return body.Rout / 0.28 * (max_mass / body.mass) ** 0.32


def _ensure_attributes(bodies):
    """
    Ensure bodies have necessary attributes.
    Args:
        bodies (Particles):  Stellar particles to check and add attributes to.
    """
    defaults = {
        "age": 0 | units.yr,
        "coll_events": 0,
        "fuv_luminosity": 0.0 | units.LSun,
        }
    for name, default in defaults.items():
        if not hasattr(bodies, name):
            setattr(bodies, name, default)


def _get_background_radiation(bodies):
    """Get the background radiation on star."""
    for i, star in enumerate(bodies):
        if star.Rout > (0. | units.au):
            externals = bodies - star
            dr2 = (star.position - externals.position).lengths_squared()
            star.fuv_ambient_flux = (externals.fuv_luminosity / (4.0 * np.pi * dr2)).sum()


def truncate_disks(bodies, dt_bridge, verbose):
    """
    Truncate disks using geometric approach.
    
    Args:
        bodies (Particles):    Stellar particles with disk attributes.
        dt_bridge (float):     Time step for bridge evolution.
        verbose (bool):        Whether to print truncation details.
    """
    if len(bodies) < 2:
        return
    
    pos = bodies.position.value_in(units.au)
    vel = bodies.velocity.value_in(units.kms)
    mass = bodies.mass.value_in(units.MSun)

    dr = pos[:, np.newaxis] - pos
    dr2 = np.sum(dr * dr, axis=2)
    np.fill_diagonal(dr2, np.inf)

    nn_index = np.argmin(dr2, axis=1)
    nn_dist = np.sqrt(dr2[np.arange(len(bodies)), nn_index])
    
    dv = vel[:, np.newaxis] - vel
    dv2 = np.sum(dv * dv, axis=2)
    nn_vel = np.sqrt(dv2[np.arange(len(bodies)), nn_index]) | units.kms
    
    # First-order estimate of how far the stars will move in the next time step
    dr_nn = (nn_vel * dt_bridge).value_in(units.au)

    pair_set = set()
    for i, j in enumerate(nn_index):
        if i == j: 
            continue
        if nn_dist[i] > dr_nn[i]:
            continue
        if bodies[i].Rout <= 0.0 | units.au and bodies[j].Rout <= 0.0 | units.au:
            continue

        pair_set.add(tuple(sorted((int(i), int(j)))))

    if not pair_set:
        return

    pairs = np.array(list(pair_set), dtype=int)
    i_all = pairs[:, 0]
    j_all = pairs[:, 1]

    r = pos[j_all] - pos[i_all]
    v = vel[j_all] - vel[i_all]

    rnorm = np.linalg.norm(r, axis=1)
    hvec = np.cross(r, v)
    h2 = np.sum(hvec * hvec, axis=1) | (units.au * units.kms) ** 2
    v2 = np.sum(v * v, axis=1)

    mu = constants.G * ((mass[i_all] + mass[j_all]) | units.MSun)
    eps = 0.5 * (v2 | units.kms**2) - mu / (rnorm | units.au)

    ecc2 = 1.0 + 2.0 * eps * h2 / (mu * mu)
    ecc = np.sqrt(ecc2)
    rperi = h2 / (mu * (1.0 + ecc))

    rtrunc_i, rtrunc_j = _apply_truncation(
        rperi, mass[i_all], mass[j_all]
    )
    
    # Strip units to vectorise
    rtrunc_i_au = rtrunc_i.value_in(units.au)
    rtrunc_j_au = rtrunc_j.value_in(units.au)
    
    new_rout = bodies.Rout.value_in(units.au)
    if verbose:
        trunc_i = new_rout[i_all] > rtrunc_i_au
        trunc_j = new_rout[j_all] > rtrunc_j_au
        if 1:
            for old, new in zip(new_rout[i_all][trunc_i], rtrunc_i_au[trunc_i]):
                print(f"    Disk truncations: {old:.2f} au --> {new:.2f} au")
            for old, new in zip(new_rout[j_all][trunc_j], rtrunc_j_au[trunc_j]):
                print(f"    Disk truncations: {old:.2f} au --> {new:.2f} au")
    np.minimum.at(new_rout, i_all, rtrunc_i_au)
    np.minimum.at(new_rout, j_all, rtrunc_j_au)
    bodies.Rout = new_rout | units.au


def merge_particles(bodies, colliders, model_time, output, trunc_mode=0):
    """
    Merge colliding particles via sticky-sphere approximation and log the encounter details.
    Args:
        bodies (Particles):    The full set of particles in the simulation.
        colliders (Particles): The subset of colliding particles.
        model_time (float):    The time at which the collision occurs.
        output (str):          Path to the output file for logging collision details.
        trunc_mode (int):      Mode for truncation (0 for classic, 1 for modified)
    """
    def _log_collision(output, model_time, colliders, sma, ecc, inc):
        with open(output, "w") as f:
            f.write(f"Tcoll: {model_time.in_(units.yr)}")
            f.write(f"\nKey1: {colliders[0].key}")
            f.write(f"\nKey2: {colliders[1].key}")
            f.write(f"\nM1: {colliders[0].mass.in_(units.MSun)}")
            f.write(f"\nM2: {colliders[1].mass.in_(units.MSun)}")
            f.write(f"\nSemi-major axis: {abs(sma).in_(units.au)}")
            f.write(f"\nEccentricity: {ecc}")
            f.write(f"\nInclination: {inc.in_(units.deg)}")
    
    def _create_remnant(colliders):
        new_particle = Particles(1)
        new_particle.mass = colliders.mass.sum()
        new_particle.position = colliders.center_of_mass()
        new_particle.velocity = colliders.center_of_mass_velocity()
        new_particle.coll_events = colliders.coll_events.sum() + 1
        new_particle.fuv_luminosity = 0.0 | units.LSun
        new_particle.Rout = 0.0 | units.au
        new_particle.unmodified_radius = 0.0 | units.au
        new_particle.radius = ZAMS_radius(new_particle.mass)
        return new_particle

    kepler_elements = orbital_elements(colliders, G=constants.G)
    sma = kepler_elements[2]
    ecc = kepler_elements[3]
    inc = kepler_elements[4]
    
    if trunc_mode:
        dr = (colliders[1].position - colliders[0].position).length()
        body_a_zams = ZAMS_radius(colliders[0].mass)
        body_b_zams = ZAMS_radius(colliders[1].mass)
        rcoll = body_a_zams + body_b_zams
        
        printed = False
        if dr > rcoll:
            q = sma * (1 - ecc)
            rtrunc_i, rtrunc_j = _apply_truncation(
                q, colliders[0].mass, colliders[1].mass
                )
            for collider, rtrunc in zip(colliders, [rtrunc_i, rtrunc_j]):
                if rtrunc >= collider.Rout:
                    continue
                if not printed:
                    print(f"   dr = {dr.value_in(units.au):.2f} au, q = {q.value_in(units.au):.2f} au")
                    printed = True
                print(f"    Disk truncation: {collider.Rout.value_in(units.au):.2f} au --> {rtrunc.value_in(units.au):.2f} au")
                collider.Rout = rtrunc
                collider.unmodified_radius = _get_new_radius(collider, bodies.mass.max())
                printed = True

            colliders.radius = colliders.radius / 3.  # Tuning parameter to reduce number of collisions and overhead
            
        else:
            _log_collision(output, model_time, colliders, sma, ecc, inc)
            new_particle = _create_remnant(colliders)

            bodies.remove_particles(colliders)
            bodies.add_particles(new_particle)
            
    else:
        _log_collision(output, model_time, colliders, sma, ecc, inc)
        new_particle = _create_remnant(colliders)

        bodies.remove_particles(colliders)
        bodies.add_particles(new_particle)


def run_code(
    input_file,
    dt_bridge=None,
    diag_time=0.01 | units.Myr,
    end_time=1 | units.Myr,
    verbose=False,
    number_of_workers=1,
    trunc_mode=0
):
    """
    Run the cluster + disk evolution simulation.
    Defaults are taken from arXiv:2302.03721.
    
    Args:
        input_file (str):          Path to the initial conditions file.
        dt_bridge (float):         Evolution time step.
        diag_time (float):         Diagnostic time step.
        end_time (float):          Time to end the simulation.
        verbose (bool):            Whether to print progress information.
        number_of_workers (int):   Number of workers to use for gravity and disk evolution
        trunc_mode (int):          Mode for truncation (0 for classic, 1 for modified)
    """
    if dt_bridge is not None and diag_time < dt_bridge:
        raise ValueError(
            "Diagnostic time step must be greater than evolution time step."
            )

    # Organise data outputs
    output_root="shared_src"
    snap_dir = os.path.join(output_root, "cluster_data")
    coll_dir = os.path.join(output_root, "collisions")
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(coll_dir, exist_ok=True)
    output_file = "viscous_particles_plt_i{:05d}.hdf5"
    data_file="disk_data/"
    
    # Read initial cluster file
    bodies = read_set_from_file(input_file)
    bodies = bodies[bodies.mass > MASS_MIN]
    bodies.Rout = get_disk_outer_edge(bodies.mass)
    if trunc_mode == 1:
        bodies.radius = bodies.Rout / 0.28 * (bodies.mass.max() / bodies.mass) ** 0.32
        bodies.unmodified_radius = bodies.radius.copy()

    diskless = bodies[
        (bodies.mass < MASS_MIN) | (bodies.mass > MASS_MAX)
        ]
    diskless.Rout = 0.0 | units.au
    _ensure_attributes(bodies)
    bodies.fuv_luminosity = fuv_luminosity_from_masses(
        mass=bodies.mass, file=data_file
        )

    # Setup integrators    
    converter = nbody_system.nbody_to_si(
        bodies.mass.sum(), bodies.virial_radius()
        )

    gravity = Ph4(converter, number_of_workers=number_of_workers)
    gravity.parameters.timestep_parameter = 0.03
    gravity.particles.add_particles(bodies)
    grav_coll = gravity.stopping_conditions.collision_detection
    grav_coll.enable()

    stellar = SeBa()
    stellar.particles.add_particles(bodies)

    chnl_grav_to_local = gravity.particles.new_channel_to(bodies)
    chnl_local_to_grav = bodies.new_channel_to(gravity.particles)
    if trunc_mode == 0:
        chnl_star_to_grav = stellar.particles.new_channel_to(gravity.particles)
    else:
        chnl_star_to_grav = stellar.particles.new_channel_to(
            gravity.particles, 
            attributes=["mass"], 
            target_names=["mass"]
            )
    
    if verbose:
        print(
            f"{len(bodies) - len(diskless)} stars in the FRIED grid range.",
            flush=True,
        )
        print(f"Gravity code has {len(gravity.particles)} particles", flush=True)
    
    with open("shared_src/cluster_data/bridge_step.txt", "w") as f:
        f.write(str(diag_time.value_in(units.yr)))
    if dt_bridge is None:
        dt_bridge = diag_time

    import matplotlib.pyplot as plt
    disks = np.sort(bodies.Rout.value_in(units.au))
    y = np.arange(len(disks)) / len(disks)
    plt.plot(disks, y)
    
    snap_no = 0
    coll_no = 0
    time = 0.0 | units.yr
    next_diag_time = diag_time
    while time < end_time:
        time += dt_bridge
        if verbose:
            print(f"\rtime={time.value_in(units.Myr):.3f} Myr", flush=True, end="")
            print()
            
        while gravity.model_time < time:
            gravity.evolve_model(time)
            if grav_coll.is_set():
                chnl_grav_to_local.copy()

                for ci in range(len(grav_coll.particles(0))):
                    coll_no += 1
                    encounter = Particles(
                        particles=[grav_coll.particles(0)[ci], grav_coll.particles(1)[ci]]
                    )
                    colliders = encounter.get_intersecting_subset_in(bodies)
                    outpath = os.path.join(coll_dir, f"collision_{coll_no}.txt")
                    merge_particles(
                        bodies,
                        colliders,
                        gravity.model_time,
                        outpath,
                        trunc_mode
                    )

                    bodies.synchronize_to(gravity.particles)
                    bodies.synchronize_to(stellar.particles)
                    chnl_local_to_grav.copy()

        if trunc_mode == 1:
            bodies.radius = bodies.unmodified_radius
            chnl_local_to_grav.copy()

        stellar.evolve_model(time)
        chnl_star_to_grav.copy()
        chnl_grav_to_local.copy()
        
        bodies.fuv_luminosity = fuv_luminosity_from_masses(
            mass=bodies.mass, file=data_file
            )
        _get_background_radiation(bodies)
        if trunc_mode == 0:
            truncate_disks(bodies, dt_bridge, verbose)

        if gravity.model_time >= next_diag_time:
            snap_no += 1
            next_diag_time += diag_time
            
            bodies.age = gravity.model_time
            
            filename = os.path.join(
                snap_dir, output_file.format(snap_no)
                )
            write_set_to_file(
                bodies,
                filename,
                "amuse",
                close_file=True,
                overwrite_file=True,
            )
    
    disks = np.sort(bodies.Rout.value_in(units.au))
    y = np.arange(len(disks)) / len(disks)
    plt.plot(disks, y)
    plt.xlabel("Disk outer radius (au)")
    plt.ylabel("Cumulative fraction")
    plt.xscale("log")
    plt.show()
    
    gravity.stop()
    stellar.stop()


if __name__ == "__main__":
    run_code(
        input_file="Run1_Nast500.hdf5", 
        dt_bridge=0.01 | units.Myr,
        verbose=True,
        trunc_mode=1
    )