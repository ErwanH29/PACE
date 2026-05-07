"""
Cluster + disk evolution driver using a pooled set of VADER workers.
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

from shared_src.extra_funcs import get_rdisk_out
from disk_in_clusters.FRIED_interp import FRIED_interp
from disk_in_clusters.fuv_luminosity import fuv_luminosity_from_masses


G0 = 1.6e-3 * units.erg / units.s / units.cm**2
MASS_MIN = 0.05 | units.MSun
MASS_MAX = 1.9 | units.MSun
DUMMY_MDOT = 1e-10 | units.MSun / units.yr


def _ensure_attributes(bodies):
    """
    Ensure bodies have necessary attributes.
    Args:
        bodies (Particles):  Stellar particles to check and add attributes to.
    """
    defaults = {
        "fuv_ambient_flux": 0.0 | G0,
        "coll_events": 0,
        "fuv_luminosity": 0.0 | units.LSun,
    }
    for name, default in defaults.items():
        if not hasattr(bodies, name):
            setattr(bodies, name, default)


def _compute_external_fields(bodies):
    """
    Compute the external FUV field at the location of each star.
    This uses the geometric approximation and ignores dust extinction,
    making it more violent than the more detailed approach (arXiv:2302.03721)
    
    Args:
        bodies (Particles):  Stellar particles.
    Returns:
        fields (Quantity array): FUV flux at each star's location.
    """
    if not hasattr(bodies, "fuv_luminosity"):
        raise AttributeError("bodies must already have fuv_luminosity")
    if len(bodies) < 2:
        return np.zeros(len(bodies)) | G0

    fields = np.zeros(len(bodies)) | G0
    for i, star in enumerate(bodies):
        externals = bodies - star

        dr2 = (star.position - externals.position).lengths_squared()
        fields[i] = (externals.fuv_luminosity / (4.0 * np.pi * dr2)).sum()
    return fields


def _get_background_radiation(bodies):
    """Get the background radiation on host star."""
    fields = _compute_external_fields(bodies)

    diskless_mask = bodies.Rout <= 0.0 | units.au
    diskless = bodies[diskless_mask]
    disk_host = bodies[~diskless_mask]
    
    diskless.fuv_ambient_flux = 0.0 | G0
    disk_host.fuv_ambient_flux = fields[~diskless_mask]


def truncate_disks(bodies, dt_bridge, verbose):
    """Truncate disks using prescription from arXiv:1403.8099"""
    if len(bodies) < 2:
        return
    if verbose:
        print(f"Calculating disk truncations", flush=True)

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
        if nn_dist[i] > 2. * dr_nn[i]:
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

    rtrunc_i = rperi / 3.0 * (mass[i_all] / mass[j_all]) ** 0.32
    rtrunc_j = rperi / 3.0 * (mass[j_all] / mass[i_all]) ** 0.32
    
    # Strip units to vectorise
    rtrunc_i_au = rtrunc_i.value_in(units.au)
    rtrunc_j_au = rtrunc_j.value_in(units.au)
    
    new_rout = bodies.Rout.value_in(units.au)
    np.minimum.at(new_rout, i_all, rtrunc_i_au)
    np.minimum.at(new_rout, j_all, rtrunc_j_au)
    bodies.Rout = new_rout | units.au


def merge_particles(bodies, colliders, model_time, output):
    """
    Merge colliding particles via sticky-sphere approximation and log the encounter details.
    Args:
        bodies (Particles):    The full set of particles in the simulation.
        colliders (Particles): The subset of colliding particles.
        model_time (float):    The time at which the collision occurs.
        output (str):          Path to the output file for logging collision details.
    """
    kepler_elements = orbital_elements(colliders, G=constants.G)
    sma = kepler_elements[2]
    ecc = kepler_elements[3]
    inc = kepler_elements[4]

    with open(output, "w") as f:
        f.write(f"Tcoll: {model_time.in_(units.yr)}")
        f.write(f"\nKey1: {colliders[0].key}")
        f.write(f"\nKey2: {colliders[1].key}")
        f.write(f"\nM1: {colliders[0].mass.in_(units.MSun)}")
        f.write(f"\nM2: {colliders[1].mass.in_(units.MSun)}")
        f.write(f"\nSemi-major axis: {abs(sma).in_(units.au)}")
        f.write(f"\nEccentricity: {ecc}")
        f.write(f"\nInclination: {inc.in_(units.deg)}")

    new_particle = Particles(1)
    new_particle.mass = colliders.mass.sum()
    new_particle.position = colliders.center_of_mass()
    new_particle.velocity = colliders.center_of_mass_velocity()
    new_particle.coll_events = colliders.coll_events.sum() + 1
    new_particle.fuv_luminosity = 0.0 | units.LSun
    new_particle.Rout = 0.0 | units.au

    bodies.remove_particles(colliders)
    bodies.add_particles(new_particle)


def run_code(
    input_file,
    dt_bridge=None,
    diag_time=0.01 | units.Myr,
    end_time=1 | units.Myr,
    output_file="viscous_particles_plt_i{:05d}.hdf5",  # DO NOT CHANGE
    verbose=False,
    number_of_workers=1,
    data_file="disk_data/",
    output_root="shared_src",
):
    """
    Run the cluster + disk evolution simulation.
    Defaults are taken from arXiv:2302.03721.
    
    Args:
        input_file (str):          Path to the initial conditions file.
        dt_bridge (float):         Evolution time step.
        diag_time (float):         Diagnostic time step.
        end_time (float):          Time to end the simulation.
        output_file (str):         Filename pattern for snapshots.
        verbose (bool):            Whether to print progress information.
        number_of_workers (int):   Number of workers to use for gravity and disk evolution
        data_file (str):           Path to the directory containing data files for disk evolution (e.g. FRIED grid).
        output_root (str):         Directory to save output snapshots and collision data.
    """
    if dt_bridge is not None and diag_time < dt_bridge:
        raise ValueError(
            "Diagnostic time step must be greater than evolution time step."
            )

    snap_dir = os.path.join(output_root, "cluster_data")
    coll_dir = os.path.join(output_root, "collisions")
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(coll_dir, exist_ok=True)

    bodies = read_set_from_file(input_file)
    bodies = bodies[bodies.mass > MASS_MIN][-15:]
    bodies.Rout = get_rdisk_out(bodies.mass)
    _ensure_attributes(bodies)

    pe_interp = FRIED_interp(verbosity=False, folder=data_file)
    bodies.fuv_luminosity = fuv_luminosity_from_masses(bodies, pe_interp)
    
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
    chnl_star_to_grav = stellar.particles.new_channel_to(gravity.particles)
    
    eligible_hosts = bodies[
        (bodies.mass >= MASS_MIN) & (bodies.mass <= MASS_MAX)
        ]
    if verbose:
        print(
            f"Identified {len(eligible_hosts)} stars with mass in the FRIED grid range "
            f"({MASS_MIN.in_(units.MSun)} - {MASS_MAX.in_(units.MSun)})",
            flush=True,
        )
        print(f"Gravity code has {len(gravity.particles)} particles", flush=True)
    
    with open("shared_src/cluster_data/bridge_step.txt", "w") as f:
        f.write(str(diag_time.value_in(units.yr)))
    if dt_bridge is None:
        dt_bridge = diag_time

    snap_no = 0
    coll_no = 0
    time = 0.0 | units.yr
    next_diag_time = diag_time
    while time < end_time:
        time += dt_bridge
        if verbose:
            print(f"time={time.in_(units.Myr)}", flush=True)
        
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
                    )

                    bodies.synchronize_to(gravity.particles)
                    bodies.synchronize_to(stellar.particles)
                    
                    assert len(bodies) == len(gravity.particles) == len(stellar.particles), "Particle counts must match after collision handling"
        
        stellar.evolve_model(time)
        chnl_grav_to_local.copy()
        chnl_star_to_grav.copy()
        
        bodies.fuv_luminosity = fuv_luminosity_from_masses(bodies, pe_interp)
        _get_background_radiation(bodies)
        truncate_disks(bodies, dt_bridge, verbose)
        
        ### JIJ BENT HIER

        if gravity.model_time >= next_diag_time:
            snap_no += 1
            next_diag_time += diag_time
            filename = os.path.join(snap_dir, output_file.format(snap_no))
            write_set_to_file(
                bodies,
                filename,
                "amuse",
                close_file=True,
                overwrite_file=True,
            )

    gravity.stop()
    stellar.stop()


if __name__ == "__main__":
    run_code(
        "Run1_Nast500.hdf5", 
        dt_bridge=0.01 | units.Myr,
        verbose=True,
    )