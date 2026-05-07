"""
Cluster + disk evolution driver using a pooled set of VADER workers.
"""

import numpy as np
import os
import signal
import psutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from amuse.community.ph4.interface import Ph4
from amuse.community.seba.interface import SeBa
from amuse.community.vader.interface import Vader
from amuse.datamodel import new_regular_grid
from amuse.ext.orbital_elements import orbital_elements
from amuse.lab import (
    constants, nbody_system, Particles, units,
    read_set_from_file, write_set_to_file,
)

from disk_in_clusters.FRIED_interp import FRIED_interp
from disk_in_clusters.disk_and_dust_class import Disk, DustParameters, SIGMA_FLOOR
from disk_in_clusters.fuv_luminosity import fuv_luminosity_from_masses
from shared_src.extra_funcs import get_rdisk_out, get_mdisk
from shared_src.params import FDG, MU, STOKES_NUMBER, GAMMA


G0 = 1.6e-3 * units.erg / units.s / units.cm**2
MASS_MIN = 0.05 | units.MSun
MASS_MAX = 1.9 | units.MSun
TRUNCATION_LIMIT = 1000 | units.au
DUMMY_MDOT = 1e-10 | units.MSun / units.yr

DISK_ATTRIBUTES = [
    "accreted_mass",
    "disk_gas_mass",
    "disk_dust_mass",
    "disk_mass",
    "truncation_mass_loss",
    "disk_radius",
    "outer_photoevap_rate",
    "t_viscous",
    "age",
    "disk_active",
    "disk_dispersed",
    "disk_convergence_failure",
    "ipe_mass_loss",
    "epe_mass_loss",
]
DISK_DEFAULTS = [
    0.0 | units.MSun,
    0.0 | units.MSun,
    0.0 | units.MSun,
    0.0 | units.MSun,
    0.0 | units.MSun,
    0.0 | units.au,
    0.0 | units.MSun / units.yr,
    0.0 | units.yr,
    0.0 | units.yr,
    False,
    False,
    False,
    0.0 | units.MSun,
    0.0 | units.MSun,
]


class SimulationParameters(object):
    """Container for simulation state and parameters."""
    def __init__(
        self,
        bodies,
        gravity,
        stellar,
        disk_map,
        pe_interp,
        workers_used,
        vader_mode,
        vader_setup={},
        disk_codes=[],
        mass_rtol=1e-7,
    ):
        """
        Initialise the simulation state.
        Args:
            bodies (Particles):        Stellar particles in the simulation.
            gravity (Ph4):             Gravity code instance.
            stellar (SeBa):            Stellar evolution code instance.
            disk_map (dict):           Dictionary mapping star keys to disks.
            pe_interp (FRIED_interp):  Interpolator for external photoevaporation rates.
            cpus_available (int):      Number of CPU cores available for VADER workers.
            vader_mode (str):          VADER mode used.
            vader_setup (dict):        Dictionary of VADER code internal parameters.
            disk_codes (list):         List of VADER worker instances.
            mass_rtol (float):         Threshold for relative mass change when updating VADER Keplerian grid.
        """
        if vader_mode not in ["pedisk_dusty", "pedisk", "pedisk_nataccr"]:
            raise ValueError(f"Invalid vader_mode: {vader_mode}")
        
        self.bodies  = bodies
        self.gravity = gravity
        self.stellar = stellar
        
        self.disk_codes = disk_codes
        self.disk_map = disk_map
        
        self.pe_interp = pe_interp
        self.vader_mode = vader_mode
        self.vader_setup = vader_setup
        
        self.disk_code_map  = {}
        self.disk_pid_map   = {}
        self.disk_chnl_map  = {}
        self.disk_last_mass = {}
        
        self.mass_rtol = mass_rtol
        self.disk_key = 1
        self.time = 0.0 | units.yr
        self.snap_no = 0
        self.coll_no = 0
        self.coll_hist = []
        self.trnc_hist = []
        
        self.cpus_available = os.cpu_count() - workers_used


def ensure_output_attributes(bodies):
    """
    Ensure bodies have necessary attributes.
    Args:
        bodies (Particles):  Stellar particles to check and add attributes to.
    """
    defaults = {
        "Mdisk": 0.0 | units.MSun,
        "Rdisk": 0.0 | units.au,
        "Mdot_ext": DUMMY_MDOT,
        "fuv_ambient_flux": 0.0 | G0,
        "truncation_mass_loss": 0.0 | units.MSun,
        "epe_mass_loss": 0.0 | units.MSun,
        "coll_events": 0,
    }
    for name, default in defaults.items():
        if not hasattr(bodies, name):
            setattr(bodies, name, default)

    for name, default in zip(DISK_ATTRIBUTES, DISK_DEFAULTS):
        if not hasattr(bodies, name):
            setattr(bodies, name, default)


def setup_vader(
    n_codes,
    n_cells=1000,
    r_min=0.01 | units.au,
    r_max=3000.0 | units.au,
    alpha=1e-3,
    mu=MU,
    vader_mode="pedisk_dusty",
):
    """
    Initialise a pool of VADER workers with the same grid and parameters.
    Code parameters are based on arXiv:2302.03721.
    Args:
        n_codes (int):     Number of VADER workers to create.
        n_cells (int):     Number of radial grid cells in each VADER worker.
        r_min (float):     Inner radius of the VADER grid.
        r_max (float):     Outer radius of the VADER grid.
        alpha (float):     Viscosity parameter alpha to set in each VADER worker.
        mu (float):        Mean molecular weight to set in each VADER worker.
        vader_mode (str):  Which VADER mode to use ("pedisk_dusty", "pedisk", or "pedisk_nataccr").
        verbosity (bool):  Whether to print setup information.
    """
    mean_particle_mass = (mu * 1.008 * constants.u).value_in(units.g)
    codes = [Vader(mode=vader_mode, redirection="none") for _ in range(n_codes)]

    for code in codes:
        code.initialize_code()
        code.initialize_keplerian_grid(
            n_cells, False, r_min, r_max, 1.0 | units.MSun
        )

        code.parameters.alpha = alpha
        code.parameters.post_timestep_function = True
        code.parameters.maximum_tolerated_change = 1e99
        code.parameters.inner_pressure_boundary_type = 1
        code.parameters.inner_boundary_function = True
        code.parameters.initial_timestep = 1.0 | units.yr

        if vader_mode == "pedisk_dusty":
            code.parameters.number_of_user_parameters = 15
            code.parameters.number_of_user_outputs = 5
        else:
            code.parameters.number_of_user_parameters = 7
            code.parameters.number_of_user_outputs = 0

        code.recommit_parameters()

        code.set_parameter(2, SIGMA_FLOOR.value_in(units.g / units.cm**2))
        code.set_parameter(4, mean_particle_mass)

        if vader_mode == "pedisk_dusty":
            code.set_parameter(3, DustParameters["Tsubl"].value_in(units.K))
            code.set_parameter(5, alpha)
            code.set_parameter(7, DustParameters["uf"].value_in(units.cm / units.s))
            code.set_parameter(8, DustParameters["rho_s"].value_in(units.g / units.cm**3))
            code.set_parameter(9, DustParameters["a0"].value_in(units.cm))
            code.set_parameter(10, DustParameters["CFL"])
            code.set_parameter(11, 0.0)
            code.set_parameter(12, 0.0)
            code.set_parameter(13, 0.0)
            code.set_parameter(14, alpha)
        elif vader_mode == "pedisk_nataccr":
            code.set_parameter(5, alpha)

    return codes


def _vader_worker_pids():
    """Return a list of PIDs for all VADER worker processes."""
    pids = set()
    
    children = psutil.Process().children(recursive=True)
    for child in children:
        info = child.as_dict(attrs=["pid", "name"])
        name = info.get("name") or ""
        cmdline = " ".join(info.get("cmdline") or []).lower()
        if "vader" in name or "vader" in cmdline or "amuse_vader" in cmdline:
            pids.add(info["pid"])
    return pids


def _hibernate_workers(pid_list):
    """
    Hibernate VADER worker.
    Args:
        pid_list (list): List of VADER worker PIDs.
    """
    for pid in pid_list:
        try:
            os.kill(pid, signal.SIGSTOP)
        except ProcessLookupError:
            pass  # Process may have already exited


def _resume_pids(pid_list):
    """
    Wake VADER worker.
    Args:
        pid_list (list): List of VADER worker PIDs.
    """
    for pid in pid_list:
        try:
            os.kill(pid, signal.SIGCONT)
        except ProcessLookupError:
            pass  # Process may have already exited


def _new_vader_worker(state):
    """Create a VADER worker and return the code and PID"""
    pids_before = _vader_worker_pids()
    config = state.vader_setup
    code = setup_vader(
        n_codes=1,
        n_cells=config["n_cells"],
        r_min=config["r_min"],
        r_max=config["r_max"],
        alpha=config["alpha"],
        mu=config["mu"],
        vader_mode=state.vader_mode
    )[0]
    pids_after = _vader_worker_pids()
    return code, list(pids_after - pids_before)


def _setup_vader_disk_params(disk, code, vader_mode):
    """Set VADER code parameters based on the state of a Disk object."""
    code.set_parameter(
        0, disk.ipe_flag * disk.inner_photoevap_rate.value_in(units.g / units.s)
    )
    code.set_parameter(
        1, disk.epe_flag * disk.outer_photoevap_rate.value_in(units.g / units.s)
    )
    code.set_parameter(6, disk.central_mass.value_in(units.MSun))
    
    if vader_mode != "pedisk_dusty":
        code.set_parameter(3, disk.Tm.value_in(units.K))
    if vader_mode == "pedisk":
        code.set_parameter(5, disk.accretion_rate.value_in(units.g / units.s))


def _attach_disk_to_star(state, star_key, verbose):
    """Assign VADER code to a star based on its key."""
    if star_key in state.disk_code_map:
        return

    disk = state.disk_map[star_key]
    code, pid_list = _new_vader_worker(state)

    state.disk_code_map[star_key] = code
    state.disk_pid_map[star_key] = pid_list
    body = state.bodies[state.bodies.key == star_key][0]
    body.disk_key = state.disk_key
    state.disk_key += 1

    channels_to_code = [disk.grid.new_channel_to(code.grid)]
    channels_from_code = [code.grid.new_channel_to(disk.grid)]

    if code.parameters.number_of_user_parameters:
        channels_to_code.append(disk.grid_user.new_channel_to(code.grid_user))
        channels_from_code.append(code.grid_user.new_channel_to(disk.grid_user))

    state.disk_chnl_map[star_key] = {
        "to_code": channels_to_code,
        "from_code": channels_from_code,
    }

    if verbose:
        print(
            f"[DISK] Attached persistent VADER worker to star {star_key}; "
            f"pid(s)={pid_list if pid_list else 'unknown'}",
            flush=True,
        )

    # Initial load. The worker is still running here.
    code.update_keplerian_grid(disk.central_mass)
    state.disk_last_mass[star_key] = disk.central_mass
    _setup_vader_disk_params(disk, code, state.vader_mode)
    for channel in channels_to_code:
        channel.copy()

    _hibernate_workers(state.disk_pid_map.get(star_key, []))


def _update_keplerian_grid(state, star_key):
    """Recompute the Keplerian grid in the VADER worker if mass change beyond some threshold."""
    disk = state.disk_map[star_key]
    code = state.disk_code_map[star_key]

    old_mass = state.disk_last_mass.get(star_key)
    if old_mass is None:
        code.update_keplerian_grid(disk.central_mass)
        state.disk_last_mass[star_key] = disk.central_mass
        return

    rel_change = abs(((disk.central_mass - old_mass) / old_mass))
    if rel_change > state.mass_rtol:
        code.update_keplerian_grid(disk.central_mass)
        state.disk_last_mass[star_key] = disk.central_mass


def _stop_disk_worker(state, star_key):
    """Stop and clean up a VADER worker."""
    code = state.disk_code_map.pop(star_key, None)
    _resume_pids(state.disk_pid_map.get(star_key, []))
    code.cleanup_code()
    code.stop()

    state.disk_pid_map.pop(star_key, None)
    state.disk_chnl_map.pop(star_key, None)
    state.disk_last_mass.pop(star_key, None)
    

def cleanup_orphaned_workers(state):
    """Cleanup VADER workers whose disks disappeared after collisions/removal."""
    live_keys = set(int(k) for k in state.bodies.key)
    owned_keys = list(state.disk_code_map.keys())
    for key in owned_keys:
        if key not in state.disk_map or key not in live_keys:
            _stop_disk_worker(state, key)


def build_disk_map(
    bodies,
    template_code,
    alpha=1e-2,
    mu=MU,
    Tm=None,
    critical_radii=None,
    vader_mode="pedisk_dusty",
    data_file="disk_data/",
):
    """
    Create a mapping from star keys to Disk objects.
    
    Args:
        bodies (Particles):    Stellar particles to create disks for.
        template_code (Vader): A VADER code instance to use as a template for grid and parameters.
        alpha (float):         Viscosity parameter alpha to set in each Disk.
        mu (float):            Mean molecular weight to set in each Disk.
        Tm (list):             List of midplane temperatures to set in each Disk
        critical_radii (list): List of critical radii to set in each Disk.
        vader_mode (str):      Which VADER mode to use ("pedisk_dusty", "pedisk", or "pedisk_nataccr").
        data_file (str):       Path to the directory containing data files for disk evolution (e.g. FRIED grid).
    Returns:
        disk_map (dict):      Mapping from star keys to Disk objects.
    """
    eligible_hosts = bodies[
        (MASS_MIN <= bodies.mass)
        & (bodies.mass <= MASS_MAX)
        ]
    number_of_disks = len(eligible_hosts)

    if critical_radii is None:
        critical_radii = [None] * number_of_disks
    if Tm is None:
        Tm = [None] * number_of_disks

    disk_map = {}
    code_idx = 0
    for star in bodies:
        if not (MASS_MIN <= star.mass <= MASS_MAX):
            continue

        mdisk = get_mdisk(star.mass)
        rdisk = get_rdisk_out(star.mass)

        new_disk = Disk(
            disk_radius=rdisk,
            disk_gas_mass=mdisk,
            central_mass=star.mass,
            grid=template_code.grid,
            alpha=alpha,
            mu=mu,
            Tm=Tm[code_idx],
            critical_radius=critical_radii[code_idx],
            data_file=data_file,
        )

        if vader_mode == "pedisk_dusty":
            new_disk.grid_user = template_code.grid_user.copy()
            new_disk.grid_user[0].value = (
                0.01 * new_disk.grid.column_density.value_in(units.g / units.cm**2)
            )
            new_disk.grid_user[1].value = DustParameters["a0"].value_in(units.cm)
            new_disk.grid_user[2].value = (
                new_disk.Tm.value_in(units.K)
                * new_disk.grid.r.value_in(units.au) ** -0.5
            )
            if (
                (new_disk.grid_user[2, 0].value | units.K) >= DustParameters["Tsubl"]
                and (new_disk.grid_user[2, 1].value | units.K) < DustParameters["Tsubl"]
            ):
                new_disk.grid_user[2, 0].value = new_disk.grid_user[2, 1].value

        new_disk.dm_trunc = 0.0 | units.MSun
        new_disk.host_star_key = star.key
        disk_map[star.key] = new_disk
        code_idx += 1

    return disk_map


def sync_disk_scalars_to_bodies(bodies, disk_map):
    """
    Synchronise disk properties to their corresponding host.
    
    Args:
        bodies (Particles): Stellar particles to synchronise disk properties to.
        disk_map (dict):    Mapping from star keys to Disk objects.
    """
    ensure_output_attributes(bodies)

    for i, star in enumerate(bodies):
        disk = disk_map.get(star.key)
        if disk is None:
            bodies[i].Mdisk = 0.0 | units.MSun
            bodies[i].Rdisk = 0.0 | units.au
            bodies[i].Mdot_ext = DUMMY_MDOT
            bodies[i].fuv_ambient_flux = 0.0 | G0
            for name, default in zip(DISK_ATTRIBUTES, DISK_DEFAULTS):
                setattr(bodies[i], name, default)
            continue

        disk.central_mass = star.mass
        bodies[i].Mdisk = disk.disk_gas_mass
        bodies[i].Rdisk = disk.disk_radius
        bodies[i].Mdot_ext = disk.outer_photoevap_rate
        bodies[i].fuv_ambient_flux = disk.fuv_ambient_flux
        for name in DISK_ATTRIBUTES:
            setattr(bodies[i], name, getattr(disk, name))


def compute_external_fields(bodies):
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


def update_mdot_ext(state):
    """Update the mass-loss rate due to external photoevaporation"""
    fields = compute_external_fields(state.bodies)

    epe_keys = []
    epe_indices = []
    masses = []
    fluxes = []
    disk_masses = []
    disk_radii = []
    for i, star in enumerate(state.bodies):
        disk = state.disk_map.get(star.key)
        if disk is None:
            state.bodies[i].Mdot_ext = DUMMY_MDOT
            state.bodies[i].fuv_ambient_flux = 0.0 | G0
            continue

        disk.central_mass = star.mass
        disk.fuv_ambient_flux = fields[i]
        state.bodies[i].fuv_ambient_flux = fields[i]
        if fields[i] > (0 | G0):
            epe_keys.append(star.key)
            epe_indices.append(i)
            masses.append(star.mass.value_in(units.MSun))
            fluxes.append(fields[i].value_in(G0))
            disk_masses.append(disk.disk_gas_mass.value_in(units.MSun))
            disk_radii.append(disk.disk_radius.value_in(units.au))
        else:
            disk.outer_photoevap_rate = DUMMY_MDOT
            state.bodies[i].Mdot_ext = DUMMY_MDOT

    if epe_keys:
        epe_rates = state.pe_interp.interp_amuse(
            np.array(masses) | units.MSun,
            np.array(fluxes) | G0,
            np.array(disk_masses) | units.MSun,
            np.array(disk_radii) | units.au,
        )

        for key, i, rate in zip(epe_keys, epe_indices, epe_rates):
            disk = state.disk_map[key]
            disk.outer_photoevap_rate = rate
            state.bodies[i].Mdot_ext = rate


def evolve_haworth2018_dust(disk, dt):
    """
    Evolve the dust mass in the disk according to arXiv:1808.07484.
    
    Args:
        disk (Disk):    Disk object to evolve.
        dt (float):     Time step to evolve the dust mass (in years).
    """
    v_th = disk._get_thermal_speed(disk.Tm, disk.disk_radius)
    Hd = disk._get_disk_scale_height(
        disk.Tm, disk.disk_radius, disk.central_mass
        )
    fill_factor = Hd / (Hd**2 + disk.disk_radius**2) ** 0.5  # Disk filling factor of sphere at disk edge

    disk.dust_photoevap_rate = (
        disk.epe_flag
        * disk.delta
        * disk.outer_photoevap_rate ** (3.0 / 2.0)
        * (v_th / (4.0 * np.pi * fill_factor * constants.G * disk.central_mass * disk.rho_g * disk.a_min))**0.5
        * np.exp(-disk.delta * (constants.G * disk.central_mass) ** 0.5 * disk.model_time / (2.0 * disk.disk_radius ** (3.0 / 2.0)))
    )

    # Can't entrain more dust than is available
    if disk.dust_photoevap_rate > disk.delta * disk.outer_photoevap_rate:
        disk.dust_photoevap_rate = disk.delta * disk.outer_photoevap_rate

    dM_dust = disk.dust_photoevap_rate * dt
    if disk.disk_dispersed:  # If disk is dispersed, do only half a step
        dM_dust /= 2.0

    disk._disk_dust_mass -= dM_dust
    if disk._disk_dust_mass < (0.0 | units.MSun):  # Can't have negative mass
        disk._disk_dust_mass = 0.0 | units.MSun


def evolve_disks(state, dt, verbose=False):
    """
    Evolve active disks.
    
    Args:
        state (dataclass):  Simulation state containing bodies, disk map, and VADER workers.
        dt (float):         Time step to evolve the disks (in years).
        verbose (bool):     Whether to print evolution information.
    """
    def _evolve_disks(key):
        disk = state.disk_map[key]
        if not disk.disk_active:
            return

        host = state.bodies[state.bodies.key == key][0]
        disk.central_mass = host.mass

        code = state.disk_code_map[key]
        channels = state.disk_chnl_map[key]
        _resume_pids(state.disk_pid_map.get(key, []))
        if verbose:
            print(
                f"Processing disk star_key={key}. "
                f"M={disk.central_mass.value_in(units.MSun):.12g} MSun, "
                f"code_time={code.model_time.value_in(units.yr):.12g} yr, "
                f"disk_time={disk.model_time.value_in(units.yr):.12g} yr",
                flush=True,
            )

        try:
            _update_keplerian_grid(state, key)
            _setup_vader_disk_params(disk, code, state.vader_mode)

            for channel in channels["to_code"]:
                channel.copy()

            accreted_before = -code.inner_boundary_mass_out
            ipe_before = code.get_parameter(12)
            epe_before = code.get_parameter(13)

            if state.vader_mode != "pedisk_dusty":
                disk.ipe_mass_loss += disk.ipe_flag * disk.inner_photoevap_rate * dt
                disk.epe_mass_loss += disk.epe_flag * disk.outer_photoevap_rate * dt
                
            code.evolve_model(code.model_time + dt)
            if state.vader_mode == "pedisk_dusty" and code.get_parameter(11) != 0.0:
                print(
                    f"[DISK] Dust convergence failure for star {disk.host_star_key} "
                    f"at {disk.model_time.value_in(units.Myr):.6f} Myr",
                    flush=True,
                )
                disk.disk_convergence_failure = True
                disk.disk_active = False

            disk.model_time += dt
            disk.accreted_mass += -code.inner_boundary_mass_out - accreted_before

            if state.vader_mode == "pedisk_dusty":
                disk.ipe_mass_loss += (code.get_parameter(12) - ipe_before) | units.g
                disk.epe_mass_loss += (code.get_parameter(13) - epe_before) | units.g
            else:
                disk.ipe_mass_loss += disk.ipe_flag * disk.inner_photoevap_rate * dt
                disk.epe_mass_loss += disk.epe_flag * disk.outer_photoevap_rate * dt

            for channel in channels["from_code"]:
                channel.copy()

            if disk.disk_gas_mass < (8e-6 | units.MSun):
                disk.disk_dispersed = True
                disk.disk_active = False
                print(
                    f"[DISK] Disk dispersal for star {disk.host_star_key} at "
                    f"{disk.model_time.value_in(units.Myr):.6f} Myr",
                    flush=True,
                )
                
        except Exception as exc:
            print(
                f"[DISK] VADER failure for star {disk.host_star_key} "
                f"at {disk.model_time.value_in(units.Myr):.6f} Myr: {exc}",
                flush=True,
            )
            disk.disk_convergence_failure = True
            disk.disk_active = False
            
        finally:
            _hibernate_workers(state.disk_pid_map.get(key, []))
        
        
    active_stars = [
        star for star in state.bodies
        if star.key in state.disk_map and state.disk_map[star.key].disk_active
    ]
    if not active_stars:
        return

    active_keys = [star.key for star in active_stars]
    if verbose:
        print(
            f"Evolving {len(active_keys)} active disk(s)",
            flush=True,
        )
    
    for i, key in enumerate(active_keys):
        _attach_disk_to_star(state, key, verbose=verbose)
        
    if state.vader_mode != "pedisk_dusty":
        for key in active_keys:
            evolve_haworth2018_dust(state.disk_map[key], dt / 2.0)

    
    with ThreadPoolExecutor(max_workers=state.cpus_available) as executor:
        futures = {
            executor.submit(_evolve_disks, key): key
            for key in active_keys
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                future.result()
            except Exception as exc:
                print(
                    f"[DISK] Exception in disk evolution for star {key}: {exc}",
                    flush=True,
                )

    if state.vader_mode != "pedisk_dusty":
        for key in active_keys:
            disk = state.disk_map[key]
            if disk.disk_active:
                evolve_haworth2018_dust(disk, dt / 2.0)


def truncate_disks_from_nearest_neighbor(state):
    """Truncate disks using prescription from arXiv:1403.8099"""
    bodies = state.bodies
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

    pair_set = set()
    for i, j in enumerate(nn_index):
        if i == j or nn_dist[i] > TRUNCATION_LIMIT.value_in(units.au):
            continue
        if (bodies[i].key not in state.disk_map) and (bodies[j].key not in state.disk_map):
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
    for k, i in enumerate(i_all):
        disk = state.disk_map.get(bodies[i].key)
        if disk is not None:
            dm = disk.truncate_disk(rtrunc_i[k], state.vader_mode)
            if dm > (0.0 | units.MSun):
                state.trnc_hist.append(
                    {
                        "time": state.time,
                        "star_key": int(bodies[i].key),
                        "partner_key": int(bodies[j_all[k]].key),
                        "removed_mass": dm,
                        "new_radius": disk.disk_radius,
                    }
                )

    for k, j in enumerate(j_all):
        disk = state.disk_map.get(bodies[j].key)
        if disk is not None:
            dm = disk.truncate_disk(rtrunc_j[k], state.vader_mode)
            if dm > (0.0 | units.MSun):
                state.trnc_hist.append(
                    {
                        "time": state.time,
                        "star_key": int(bodies[j].key),
                        "partner_key": int(bodies[i_all[k]].key),
                        "removed_mass": dm,
                        "new_radius": disk.disk_radius,
                    }
                )


def merge_particles(bodies, enc_set, model_time, output, disk_map):
    kepler_elements = orbital_elements(enc_set, G=constants.G)
    sma = kepler_elements[2]
    ecc = kepler_elements[3]
    inc = kepler_elements[4]

    with open(output, "w") as f:
        f.write(f"Tcoll: {model_time.in_(units.yr)}")
        f.write(f"\nKey1: {enc_set[0].key}")
        f.write(f"\nKey2: {enc_set[1].key}")
        f.write(f"\nM1: {enc_set[0].mass.in_(units.MSun)}")
        f.write(f"\nM2: {enc_set[1].mass.in_(units.MSun)}")
        f.write(f"\nSemi-major axis: {abs(sma).in_(units.au)}")
        f.write(f"\nEccentricity: {ecc}")
        f.write(f"\nInclination: {inc.in_(units.deg)}")

    for p in enc_set:
        disk_map.pop(p.key, None)

    new_particle = Particles(1)
    new_particle.mass = enc_set.mass.sum()
    new_particle.position = enc_set.center_of_mass()
    new_particle.velocity = enc_set.center_of_mass_velocity()
    new_particle.coll_events = enc_set.coll_events.sum() + 1
    new_particle.fuv_luminosity = 0.0 | units.LSun

    bodies.remove_particles(enc_set)
    bodies.add_particles(new_particle)


def evolve_gravity(state, dt, coll_dir, chnl_grav_to_local, chnl_star_to_grav):
    target_time = state.time + dt
    chnl_star_to_grav.copy()
    grav_coll = state.gravity.stopping_conditions.collision_detection

    while state.gravity.model_time < target_time:
        state.gravity.evolve_model(target_time)
        if grav_coll.is_set():
            chnl_grav_to_local.copy()

            for ci in range(len(grav_coll.particles(0))):
                state.coll_no += 1
                encounter = Particles(
                    particles=[grav_coll.particles(0)[ci], grav_coll.particles(1)[ci]]
                )
                enc = encounter.get_intersecting_subset_in(state.bodies)
                outpath = os.path.join(coll_dir, f"collision_{state.coll_no}.txt")
                merge_particles(
                    state.bodies,
                    enc,
                    state.gravity.model_time,
                    outpath,
                    state.disk_map,
                )
                state.coll_hist.append(
                    {
                        "time": state.gravity.model_time,
                        "particles": [int(p.key) for p in enc],
                        "output": outpath,
                    }
                )

                state.gravity.particles.synchronize_to(state.bodies)
                state.stellar.particles.synchronize_to(state.bodies)

    state.stellar.evolve_model(target_time)
    chnl_grav_to_local.copy()
    state.time = target_time


def create_state(
    input_file,
    number_of_workers,
    alpha,
    mu,
    n_cells,
    r_min,
    r_max,
    vader_mode,
    data_file,
    verbosity=False,
):
    bodies = read_set_from_file(input_file)
    bodies = bodies[bodies.mass > MASS_MIN][-15:]
    ensure_output_attributes(bodies)

    if not hasattr(bodies, "fuv_luminosity"):
        bodies.fuv_luminosity = fuv_luminosity_from_masses(bodies.mass, file=data_file)

    converter = nbody_system.nbody_to_si(bodies.mass.sum(), bodies.virial_radius())

    gravity = Ph4(converter, number_of_workers=number_of_workers)
    gravity.parameters.timestep_parameter = 0.03
    gravity.particles.add_particles(bodies)
    gravity.stopping_conditions.collision_detection.enable()

    stellar = SeBa()
    stellar.particles.add_particles(bodies)

    eligible_hosts = bodies[(bodies.mass >= MASS_MIN) & (bodies.mass <= MASS_MAX)]
    n_disk_workers = max(1, min(number_of_workers, len(eligible_hosts))) if len(eligible_hosts) else 0

    if verbosity:
        print(
            f"Identified {len(eligible_hosts)} stars with mass in the FRIED grid range "
            f"({MASS_MIN.in_(units.MSun)} - {MASS_MAX.in_(units.MSun)})",
            flush=True,
        )
        print(f"Gravity code has {len(gravity.particles)} particles", flush=True)

    template_codes = setup_vader(
        1,
        n_cells=n_cells,
        r_min=r_min,
        r_max=r_max,
        alpha=alpha,
        mu=mu,
        vader_mode=vader_mode
    )

    disk_map = build_disk_map(
        bodies=bodies,
        template_code=template_codes[0],
        alpha=alpha,
        mu=mu,
        vader_mode=vader_mode,
        data_file=data_file,
    )

    pe_interp = FRIED_interp(verbosity=False, folder=data_file)

    state = SimulationParameters(
        bodies=bodies,
        gravity=gravity,
        stellar=stellar,
        disk_codes=[],
        disk_map=disk_map,
        pe_interp=pe_interp,
        workers_used=number_of_workers+1,
        vader_mode=vader_mode,
        vader_setup={
            "n_cells": n_cells,
            "r_min": r_min,
            "r_max": r_max,
            "alpha": alpha,
            "mu": mu,
        },
    )
    sync_disk_scalars_to_bodies(state.bodies, state.disk_map)
    return state


def _disk_to_grid(disk, star, state, alpha, alpha_acc):
    """Extract the disk properties onto a grid-like particle set"""
    grid.disk_key = star.disk_key
    grid.host_star_key = star.key
    grid.star_mass = star.mass
    grid.position = disk.grid.r
    
    grid.surface_gas = disk.grid.column_density
    grid.surface_solid = disk.grid_user[0].value | units.g / units.cm**2
    grid.grain_size = disk.grid_user[1].value | units.cm
    grid.temperature = disk.grid_user[2].value | units.K
    grid.vd = disk.grid_user[3].value | units.cm / units.s
    grid.st = disk.grid_user[4].value

    grid.scale_height = disk._get_disk_scale_height(
        disk.Tm, disk.disk_radius, disk.central_mass
    )
    
    grid.gamma = GAMMA
    grid.alpha = alpha
    grid.alpha_acc = alpha if alpha_acc is None else alpha_acc

    grid.disk_gas_mass = disk.disk_gas_mass
    grid.disk_dust_mass = disk.disk_dust_mass
    grid.disk_radius = disk.disk_radius
    grid.outer_photoevap_rate = disk.outer_photoevap_rate
    grid.truncation_mass_loss = disk.truncation_mass_loss
    
    return grid

def _get_disk_snapshot(state, alpha, alpha_acc):
    """Save disk properties into particle set"""
    disk_hosts = [
        (star, state.disk_map[star.key])
        for star in state.bodies
        if state.disk_map.get(star.key) is not None
    ]

    if len(disk_hosts) == 0:
        return None

    n_disks = len(disk_hosts)
    n_cells = len(disk_hosts[0][1].grid)

    disk_data = new_regular_grid((n_disks, n_cells), [1, 1])
    for i, (star, disk) in enumerate(disk_hosts):
        disk_data[i, :] = _disk_to_grid(
            disk_data[i, :],
            star,
            state,
            alpha,
            alpha_acc
        )
    
    return disk_data


def run_code(
    input_file,
    dt=None,
    diag_time=0.01 | units.Myr,
    end_time=1 | units.Myr,
    verbose=False,
    number_of_workers=1,
    alpha=1e-3,
    mu=MU,
    n_cells=330,
    r_min=0.01 | units.au,
    r_max=3000 | units.au,
    vader_mode="pedisk_dusty",
    data_file="disk_data/",
    output_root="shared_src",
):
    """
    Run the cluster + disk evolution simulation.
    Defaults are taken from arXiv:2302.03721.
    
    Args:
        input_file (str):          Path to the initial conditions file.
        dt (float):                Evolution time step.
        diag_time (float):         Diagnostic time step.
        end_time (float):          Time to end the simulation.
        output_file (str):         Filename pattern for snapshots.
        verbose (bool):            Whether to print progress information.
        number_of_workers (int):   Number of workers to use for gravity and disk evolution
        alpha (float):             Viscosity parameter alpha for the disks.
        mu (float):                Mean molecular weight for the disks.
        n_cells (int):             Number of radial grid cells in each VADER worker.
        r_min (float):             Inner radius of the VADER grid.
        r_max (float):             Outer radius of the VADER grid.
        vader_mode (str):          Which VADER mode to use ("pedisk_dusty", "pedisk", or "pedisk_nataccr").
        data_file (str):           Path to the directory containing data files for disk evolution (e.g. FRIED grid).
        output_root (str):         Directory to save output snapshots and collision data.
    """
    if diag_time < dt:
        raise ValueError("Diagnostic time step must be greater than or equal to evolution time step.")

    snap_dir = os.path.join(output_root, "cluster_data")
    coll_dir = os.path.join(output_root, "collisions")
    os.makedirs(snap_dir, exist_ok=True)
    os.makedirs(coll_dir, exist_ok=True)

    state = create_state(
        input_file=input_file,
        number_of_workers=number_of_workers,
        alpha=alpha,
        mu=mu,
        n_cells=n_cells,
        r_min=r_min,
        r_max=r_max,
        vader_mode=vader_mode,
        data_file=data_file,
        verbosity=verbose,
    )
    
    with open("shared_src/cluster_data/bridge_step.txt", "w") as f:
        f.write(str(diag_time.value_in(units.yr)))

    chnl_grav_to_local = state.gravity.particles.new_channel_to(state.bodies)
    chnl_star_to_grav = state.stellar.particles.new_channel_to(state.gravity.particles)

    if dt is None:
        dt = diag_time

    next_diag_time = diag_time
    while state.time < end_time:
        if verbose:
            print(f"time={state.time.in_(units.Myr)}", flush=True)
            print(f"Evolving gravity for dt={dt.in_(units.yr)}", flush=True)

        evolve_gravity(state, dt, coll_dir, chnl_grav_to_local, chnl_star_to_grav)
        cleanup_orphaned_workers(state)
        update_mdot_ext(state)
        
        if verbose:
            print(f"Evolving disks for dt={dt.in_(units.yr)}", flush=True)
        evolve_disks(state, dt, verbose=verbose)
        
        if verbose:
            print(f"Calculating disk truncations", flush=True)
        truncate_disks_from_nearest_neighbor(state)
        sync_disk_scalars_to_bodies(state.bodies, state.disk_map)

        if state.gravity.model_time >= next_diag_time:
            state.snap_no += 1
            next_diag_time += diag_time
            
            star_filename = os.path.join(
                snap_dir, 
                f"cluster_snap{state.snap_no:05d}.hdf5"
                )
            write_set_to_file(
                state.bodies,
                star_filename,
                "amuse",
                close_file=True,
                overwrite_file=True,
            )
            
            disk_filename = os.path.join(
                snap_dir, 
                f"disk_snap{state.snap_no:05d}.hdf5"
                )
            disk_profiles = _get_disk_snapshot(
                state,
                alpha=alpha,
                alpha_acc=alpha
            )
            write_set_to_file(
                disk_profiles,
                disk_filename,
                "amuse",
                close_file=True,
                overwrite_file=True,
            )
            
            

    for key in list(state.disk_code_map.keys()):
        _stop_disk_worker(state, key)
    for code in state.disk_codes:
        code.stop()
        
    state.gravity.stop()
    state.stellar.stop()
    
    return state


if __name__ == "__main__":
    run_code(
        "Run1_Nast500.hdf5", 
        dt=0.01 | units.Myr,
        verbose=True,
    )