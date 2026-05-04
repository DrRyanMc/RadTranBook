#!/usr/bin/env python3
"""
Crooked Pipe Test Problem - 2D Cylindrical (r-z) Geometry
MULTIGROUP IMC VERSION

Multigroup (power-law) opacity matched to the diffusion reference:
    sigma(T, E) = 10.0 * rho(r,z) * T^{-1/2} * E^{-3}  cm^{-1}

Material geometry and density follow the same thick/thin crooked-pipe map as
nonEquilibriumDiffusion/problems/crooked_pipe_multigroup_noneq.py so that
IMC and diffusion results can be directly compared.

Boundary condition: time-dependent blackbody source at z=0, r < 0.5 cm,
ramping from T_start to T_end over bc_ramp_time ns.

Restart/checkpoint: pickle-based (consistent with existing MG_IMC conventions).

Domain: r in [0.0, 2.0] cm, z in [0.0, 7.0] cm
Initial condition: T = Tr = T_INIT = 0.01 keV everywhere
Fiducial monitor points:
    (r=0.0, z=0.25), (r=0.0, z=2.75), (r=1.25, z=3.5),
    (r=0.0, z=4.25), (r=0.0, z=6.75)
"""

import argparse
import os
import pickle
import random
import sys
import time as _time

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from MG_IMC import A_RAD, C_LIGHT, SimulationState2DMG, init_simulation, step

_utils = os.path.join(_ROOT, 'utils')
if _utils not in sys.path:
    sys.path.insert(0, _utils)
from plotfuncs import show

# ── Physical constants ───────────────────────────────────────────────────────
# A_RAD and C_LIGHT are imported from MG_IMC.

# ── Problem constants ────────────────────────────────────────────────────────
T_INIT    = 0.01   # keV — cold initial condition
RHO_THICK = 2.0    # g/cm^3 — optically thick regions
RHO_THIN  = 0.01   # g/cm^3 — optically thin regions
CV_MASS   = 0.05   # GJ/(g*keV) — mass-specific heat capacity

CHECKPOINT_VERSION = 1


# ═══════════════════════════════════════════════════════════════════════════════
# MATERIAL GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

def is_optically_thick(r, z):
    """Return True where (r,z) is in an optically thick region.

    Crooked-pipe thick/thin map (identical to the diffusion reference):
      r in [0.0, 0.5]:
        z in [0.0, 2.5)  — thin
        z in [2.5, 7.0]  — thick   (with thin patches near r<1.5)
      r in [0.5, 2.0]:
        z in [0.0, 3.0)  — thick
        z in [3.0, 4.5)  — thin
        z in [4.5, 7.0]  — thick

    Vectorised over array inputs.
    """
    scalar_input = np.isscalar(r) and np.isscalar(z)
    r = np.atleast_1d(np.asarray(r, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))

    result = np.ones_like(r, dtype=bool)  # default: thick

    lower_thin          = (r < 0.5)  & ((z < 3.0) | (z > 4.0))
    ascending_left_thin = (r < 1.5)  & (z > 2.5) & (z < 3.0)
    ascending_right_thin= (r < 1.5)  & (z > 4.0) & (z < 4.5)
    top_thin            = (r >= 1.0) & (r < 1.5) & (z > 2.5) & (z < 4.5)

    result[lower_thin]           = False
    result[ascending_left_thin]  = False
    result[ascending_right_thin] = False
    result[top_thin]             = False

    if scalar_input:
        return bool(result[0])
    return result


def material_density(r, z):
    """Region-dependent density (g/cm^3): RHO_THICK or RHO_THIN."""
    scalar_input = np.isscalar(r) and np.isscalar(z)
    thick = is_optically_thick(r, z)
    result = np.where(thick, RHO_THICK, RHO_THIN)
    if scalar_input:
        return float(result)
    return result


def specific_heat(T, r, z):
    """Volumetric heat capacity rho * c_v  (GJ / cm^3 / keV)."""
    scalar_input = np.isscalar(T) and np.isscalar(r) and np.isscalar(z)
    rho = np.atleast_1d(material_density(r, z))
    result = rho * CV_MASS
    if scalar_input:
        return float(result.flat[0])
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# OPACITY MODEL — POWER-LAW, MG-IMC COMPATIBLE
# ═══════════════════════════════════════════════════════════════════════════════

def powerlaw_opacity_at_energy(T, E, rho=1.0):
    """sigma(T,E) = 10 * rho * T^{-1/2} * E^{-3}  (clipped at 1e14)."""
    T_safe = np.maximum(T, 1e-2)
    return np.minimum(10.0 * rho * (T_safe ** -0.5) * (E ** -3.0), 1e14)


def make_mg_opacity_func(E_low, E_high, rho_grid):
    """Build a sigma_a_g(T) callable for use in MG_IMC step().

    Parameters
    ----------
    E_low, E_high : float
        Group boundary energies (keV).
    rho_grid : ndarray, shape (nr, nz)
        Precomputed material density over the mesh.

    Returns
    -------
    callable (T_2d) -> sigma_2d
        T_2d has shape (nr, nz); returns opacity array of the same shape.
    """
    # Precomputed — rho_grid does not change during the simulation.
    def opacity_func(T):
        sigma_low  = powerlaw_opacity_at_energy(T, E_low,  rho_grid)
        sigma_high = powerlaw_opacity_at_energy(T, E_high, rho_grid)
        return np.sqrt(sigma_low * sigma_high)
    return opacity_func


# ═══════════════════════════════════════════════════════════════════════════════
# MESH GENERATION (WITH OPTIONAL INTERFACE REFINEMENT)
# ═══════════════════════════════════════════════════════════════════════════════

def _create_log_spacing_one_sided(z_start, z_end, n_cells, from_left=True):
    """Log-spaced sub-cells within [z_start, z_end]."""
    if n_cells <= 1:
        return [z_end]
    width = z_end - z_start
    max_ratio = 5.0
    r = max_ratio ** (1.0 / (n_cells - 1))
    if abs(r - 1.0) < 1e-10:
        cell_widths = [width / n_cells] * n_cells
    else:
        w0 = width * (r - 1.0) / (r**n_cells - 1.0)
        cell_widths = [w0 * r**i for i in range(n_cells)]
    if not from_left:
        cell_widths = cell_widths[::-1]
    faces = []
    pos = z_start
    for w in cell_widths:
        pos += w
        faces.append(pos)
    return faces


def _create_log_spacing_around_interface(z_left, z_right, z_int, n_cells):
    """Log-spaced sub-cells inside [z_left, z_right] with smallest near z_int."""
    if z_int <= z_left:
        return _create_log_spacing_one_sided(z_left, z_right, n_cells, from_left=True)
    if z_int >= z_right:
        return _create_log_spacing_one_sided(z_left, z_right, n_cells, from_left=False)
    n_left  = max(1, int(n_cells * (z_int - z_left) / (z_right - z_left)))
    n_right = n_cells - n_left
    faces_l = _create_log_spacing_one_sided(z_left,  z_int,  n_left,  from_left=False)
    faces_r = _create_log_spacing_one_sided(z_int,   z_right, n_right, from_left=True)
    return faces_l + faces_r


def generate_refined_faces(coord_min, coord_max, interface_locations,
                           n_refine, n_coarse, refine_width=0.05):
    """Coarse grid with log-refinement zones around each interface."""
    coarse_faces = np.linspace(coord_min, coord_max, n_coarse + 1)
    refine_info = {}
    for z_int in interface_locations:
        for i in range(n_coarse):
            cl = coarse_faces[i]
            cr = coarse_faces[i + 1]
            if cl <= z_int + refine_width and cr >= z_int - refine_width:
                mid = 0.5 * (cl + cr)
                if i not in refine_info or abs(z_int - mid) < abs(refine_info[i] - mid):
                    refine_info[i] = z_int
    faces_list = [coord_min]
    for i in range(n_coarse):
        cl = coarse_faces[i]
        cr = coarse_faces[i + 1]
        if i in refine_info:
            refined = _create_log_spacing_around_interface(cl, cr, refine_info[i], n_refine)
            faces_list.extend(refined)
        else:
            faces_list.append(cr)
    return np.array(faces_list)


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_solution(T_mat, T_rad, r_centers, z_centers, time_value, save_prefix,
                  r_edges=None, z_edges=None, T_bc=None):
    """Separate colourmap figures for material T and radiation T.

    Uses pcolormesh with actual mesh edges so non-uniform (refined) grids
    render at the correct physical coordinates.  One PNG per field, matching
    the style of the diffusion solver version.
    """
    # Build edge arrays if not supplied (uniform-mesh fall-back).
    if r_edges is None:
        dr = r_centers[1] - r_centers[0] if len(r_centers) > 1 else 1.0
        r_edges = np.concatenate([[r_centers[0] - 0.5 * dr],
                                   0.5 * (r_centers[:-1] + r_centers[1:]),
                                   [r_centers[-1] + 0.5 * dr]])
    if z_edges is None:
        dz = z_centers[1] - z_centers[0] if len(z_centers) > 1 else 1.0
        z_edges = np.concatenate([[z_centers[0] - 0.5 * dz],
                                   0.5 * (z_centers[:-1] + z_centers[1:]),
                                   [z_centers[-1] + 0.5 * dz]])

    ZE, RE = np.meshgrid(z_edges, r_edges)  # both (nr+1, nz+1)

    vmin = T_INIT

    for T_field, tag in [(T_mat, 'material'), (T_rad, 'radiation')]:
        vmax = np.ceil(T_field.max() * 100) / 100.0
        if tag == 'radiation' and T_bc is not None:
            vmax = min(vmax, 1.1 * T_bc)

        fig, ax = plt.subplots(1, 1, figsize=(8, 3 * 1.275))
        im = ax.pcolormesh(
            ZE, RE, T_field,
            vmin=vmin, vmax=vmax,
            cmap='plasma',
            shading='flat',
        )
        plt.colorbar(im, ax=ax)
        ax.set_xlabel('z (cm)')
        ax.set_ylabel('r (cm)')
        ax.set_aspect('equal')
        plt.tight_layout()
        fname = f'{save_prefix}_{tag}_t_{time_value:.5f}ns.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'    Saved {fname}')


def plot_fiducial_history(times, fiducial_data, fiducial_data_rad, mesh_tag):
    """Temperature vs time at fiducial monitor points (mat and rad)."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fiducial_data)))

    for ax, data_dict, ylabel in zip(
        axes,
        [fiducial_data, fiducial_data_rad],
        ['Material Temperature (keV)', 'Radiation Temperature (keV)'],
    ):
        for (label, vals), c in zip(data_dict.items(), colors):
            ax.plot(times, vals, label=label, color=c)
        ax.set_xlabel('Time (ns)')
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'Crooked Pipe MG-IMC — {mesh_tag}', fontsize=11)
    fig.tight_layout()
    fname = f'crooked_pipe_mg_imc_fiducial_{mesh_tag}.png'
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {fname}')


def plot_material_layout(r_centers, z_centers, run_tag, r_edges=None, z_edges=None):
    """Quick sanity-check plot of the thick/thin material map."""
    R, Z = np.meshgrid(r_centers, z_centers, indexing='ij')
    rho = material_density(R, Z)

    if r_edges is None:
        dr = r_centers[1] - r_centers[0] if len(r_centers) > 1 else 1.0
        r_edges = np.concatenate([[r_centers[0] - 0.5 * dr],
                                   0.5 * (r_centers[:-1] + r_centers[1:]),
                                   [r_centers[-1] + 0.5 * dr]])
    if z_edges is None:
        dz = z_centers[1] - z_centers[0] if len(z_centers) > 1 else 1.0
        z_edges = np.concatenate([[z_centers[0] - 0.5 * dz],
                                   0.5 * (z_centers[:-1] + z_centers[1:]),
                                   [z_centers[-1] + 0.5 * dz]])

    ZE, RE = np.meshgrid(z_edges, r_edges)
    fig, ax = plt.subplots(figsize=(4, 7))
    im = ax.pcolormesh(
        ZE, RE, rho,
        cmap='RdBu_r',
        shading='flat',
    )
    plt.colorbar(im, ax=ax, label='rho (g/cm^3)')
    ax.set_xlabel('z (cm)')
    ax.set_ylabel('r (cm)')
    ax.set_title('Material density')
    fig.tight_layout()
    fname = f'crooked_pipe_mg_imc_material_{run_tag}.png'
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)


# ═══════════════════════════════════════════════════════════════════════════════
# CHECKPOINT  (pickle-based, matching existing MG_IMC conventions)
# ═══════════════════════════════════════════════════════════════════════════════

def _serialize_state(state):
    payload = {
        'weights':    state.weights,
        'dir1':       state.dir1,
        'dir2':       state.dir2,
        'times':      state.times,
        'pos1':       state.pos1,
        'pos2':       state.pos2,
        'cell_i':     state.cell_i,
        'cell_j':     state.cell_j,
        'groups':     state.groups,
        'internal_energy':          state.internal_energy,
        'temperature':              state.temperature,
        'radiation_temperature':    state.radiation_temperature,
        'radiation_energy_by_group': state.radiation_energy_by_group,
        'time':                     float(state.time),
        'previous_total_energy':    float(state.previous_total_energy),
        'count':                    int(state.count),
        'radiation_energy_by_group_postcomb': state.radiation_energy_by_group_postcomb,
    }
    return payload


def _deserialize_state(data):
    return SimulationState2DMG(
        weights=data['weights'],
        dir1=data['dir1'],
        dir2=data['dir2'],
        times=data['times'],
        pos1=data['pos1'],
        pos2=data['pos2'],
        cell_i=data['cell_i'],
        cell_j=data['cell_j'],
        groups=data['groups'],
        internal_energy=data['internal_energy'],
        temperature=data['temperature'],
        radiation_temperature=data['radiation_temperature'],
        radiation_energy_by_group=data['radiation_energy_by_group'],
        time=float(data['time']),
        previous_total_energy=float(data['previous_total_energy']),
        count=int(data['count']),
        radiation_energy_by_group_postcomb=data.get('radiation_energy_by_group_postcomb'),
    )


def save_checkpoint(path, state, step_count, current_dt, times, fiducial_data,
                    fiducial_data_rad, output_times_saved, metadata):
    """Persist full simulation state so the run can be resumed after a crash."""
    payload = {
        'checkpoint_version':    CHECKPOINT_VERSION,
        'state':                 _serialize_state(state),
        'step_count':            int(step_count),
        'current_dt':            float(current_dt),
        'times':                 list(times),
        'fiducial_data':         {k: list(v) for k, v in fiducial_data.items()},
        'fiducial_data_rad':     {k: list(v) for k, v in fiducial_data_rad.items()},
        'output_times_saved':    list(output_times_saved),
        'metadata':              metadata,
        'np_random_state':       np.random.get_state(),
        'py_random_state':       random.getstate(),
    }
    # Write to a temporary file first, then rename for atomic replacement.
    tmp = path + '.tmp'
    with open(tmp, 'wb') as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(tmp, path)


def load_checkpoint(path):
    """Load and validate a checkpoint file.  Returns the full payload dict."""
    with open(path, 'rb') as f:
        payload = pickle.load(f)
    version = payload.get('checkpoint_version')
    if version != CHECKPOINT_VERSION:
        raise ValueError(
            f'Unsupported checkpoint version {version!r} '
            f'(expected {CHECKPOINT_VERSION})'
        )
    return payload


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN SIMULATION DRIVER
# ═══════════════════════════════════════════════════════════════════════════════

def main(
    n_groups=10,
    output_times=None,
    nr=60,
    nz=210,
    dt_initial=1e-4,
    dt_max=0.01,
    dt_increase_factor=1.1,
    Ntarget=20000,
    Nboundary=20000,
    Ntotal=0,
    Ntotal_T_floor=0.0,
    Nmax=100000,
    bc_t_start=0.05,
    bc_t_end=0.5,
    bc_ramp_time=20.0,
    use_refined_mesh=False,
    n_refine=10,
    n_coarse_r=None,
    n_coarse_z=None,
    refine_width=0.05,
    max_events_per_particle=10**6,
    restart_file=None,
    checkpoint_file=None,
    checkpoint_every=0,
):
    """Run the multigroup IMC crooked pipe problem.

    Parameters
    ----------
    n_groups : int
        Number of energy groups.
    output_times : list of float
        Times (ns) at which to save solution plots and checkpoints.
    nr, nz : int
        Coarse-mesh cell counts in r and z directions.
    dt_initial : float
        Initial time step (ns).
    dt_max : float
        Maximum time step (ns).
    dt_increase_factor : float
        Factor by which dt grows each step (no output hit).
    Ntarget : int
        Target emission particles per step (used when Ntotal=0).
    Nboundary : int
        Boundary source particles per step (used when Ntotal=0).
    Ntotal : int
        If > 0, override Ntarget/Nboundary and split this total budget
        between boundary and material emission proportional to the energy
        emitted from each source every step.
    Ntotal_T_floor : float
        When using Ntotal mode, material cells at or below this temperature
        (keV) are excluded from the material emission estimate so that a cold
        bulk domain does not dilute the boundary share.  Set to
        ``1.1 * T_INIT`` (or similar) to ignore unheated material.
        Clamp is always applied: f_bc is restricted to [0.1, 0.9].
    Nmax : int
        Maximum census particles after combing.
    bc_t_start : float
        Boundary temperature at t=0 (keV).
    bc_t_end : float
        Boundary temperature after ramp (keV).
    bc_ramp_time : float
        Time (ns) over which the boundary temperature ramps.
    use_refined_mesh : bool
        If True, refine mesh near material interfaces.
    n_refine : int
        Sub-divisions per coarse cell in refined zones.
    n_coarse_r, n_coarse_z : int or None
        Override coarse cell counts for refined mesh (defaults to nr, nz).
    refine_width : float
        Half-width (cm) of refinement zones around each interface.
    max_events_per_particle : int
        Maximum collision events per particle per step.
    restart_file : str or None
        Path to a checkpoint (.pkl) to resume from.
    checkpoint_file : str or None
        Output checkpoint path (auto-generated if None).
    checkpoint_every : int
        Periodic checkpoint cadence in steps (0 = output-time only).
    """
    if output_times is None:
        output_times = [1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

    t_final = max(output_times)

    # ── Energy groups ────────────────────────────────────────────────────────
    # Range [~1 keV, 10 keV] on a log scale — matches the diffusion reference.
    energy_edges = np.logspace(1e-4, np.log10(10.0), n_groups + 1)

    # ── Mesh construction ────────────────────────────────────────────────────
    mesh_tag = 'refined' if use_refined_mesh else 'uniform'
    n_coarse_r = n_coarse_r if n_coarse_r is not None else nr
    n_coarse_z = n_coarse_z if n_coarse_z is not None else nz

    if use_refined_mesh:
        print('Generating refined mesh …')
        r_edges = generate_refined_faces(
            0.0, 2.0,
            interface_locations=[0.5, 1.0, 1.5],
            n_refine=n_refine,
            n_coarse=n_coarse_r,
            refine_width=refine_width,
        )
        z_edges = generate_refined_faces(
            0.0, 7.0,
            interface_locations=[2.5, 3.0, 4.0, 4.5],
            n_refine=n_refine,
            n_coarse=n_coarse_z,
            refine_width=refine_width,
        )
        print(f'  Refined mesh: {len(r_edges)-1} x {len(z_edges)-1} cells')
    else:
        r_edges = np.linspace(0.0, 2.0, nr + 1)
        z_edges = np.linspace(0.0, 7.0, nz + 1)

    r_centers = 0.5 * (r_edges[:-1] + r_edges[1:])
    z_centers = 0.5 * (z_edges[:-1] + z_edges[1:])
    nr_actual = len(r_centers)
    nz_actual = len(z_centers)

    # Precomputed spatial grids used throughout the simulation.
    R_grid, Z_grid = np.meshgrid(r_centers, z_centers, indexing='ij')
    rho_grid = material_density(R_grid, Z_grid)   # (nr_actual, nz_actual)

    # ── Material closures ────────────────────────────────────────────────────
    def eos_func(T):
        """Energy density e = rho*c_v * T."""
        return rho_grid * CV_MASS * T

    def inv_eos_func(e):
        """T = e / (rho*c_v).  c_v is temperature-independent."""
        return e / (rho_grid * CV_MASS)

    def cv_func(T):
        """Volumetric heat capacity rho*c_v (spatially varying)."""
        return rho_grid * CV_MASS

    # ── Per-group opacity closures ────────────────────────────────────────────
    sigma_a_funcs = [
        make_mg_opacity_func(energy_edges[g], energy_edges[g + 1], rho_grid)
        for g in range(n_groups)
    ]

    # ── Run tag and checkpoint filename ──────────────────────────────────────
    run_tag = f'{n_groups}g_{mesh_tag}_{nr_actual}x{nz_actual}'
    if checkpoint_file is None:
        checkpoint_file = f'crooked_pipe_mg_imc_checkpoint_{run_tag}.pkl'

    # ── Print run summary ─────────────────────────────────────────────────────
    print('=' * 72)
    print('CROOKED PIPE — MULTIGROUP IMC  2D Cylindrical (r-z)')
    print('=' * 72)
    print(f'  n_groups  = {n_groups}')
    print(f'  mesh      = {nr_actual} x {nz_actual}  ({mesh_tag})')
    if Ntotal > 0:
        floor_str = f', T_floor={Ntotal_T_floor:.4g} keV' if Ntotal_T_floor > 0 else ''
        print(f'  Ntotal    = {Ntotal}  (energy-proportional split{floor_str}),  Nmax = {Nmax}')
    else:
        print(f'  Ntarget   = {Ntarget},  Nboundary = {Nboundary},  Nmax = {Nmax}')
    print(f'  dt_init   = {dt_initial} ns,  dt_max = {dt_max} ns,  growth = {dt_increase_factor}')
    print(f'  T_bc ramp : {bc_t_start} → {bc_t_end} keV over {bc_ramp_time} ns')
    print(f'  output_times: {output_times}  (t_final={t_final} ns)')
    print(f'  checkpoint: {checkpoint_file}')
    print()
    print('  Boundary condition type: SURFACE blackbody source (NOT volumetric)')
    print(f'    Face: z=0 (zmin), r < 0.5 cm')
    print(f'    Source area: pi*0.5^2 = {np.pi*0.5**2:.4f} cm^2')
    print(f'    Emission weight per particle = (a*c/4) * T_bc^4 * A_cell * dt  (Planck half-flux)')
    print(f'    All other boundary faces: vacuum (open)')
    print(f'    Left (rmin): reflecting (cylindrical axis symmetry)')
    T_bc_t0 = bc_t_start
    E_bc_per_ns = A_RAD * C_LIGHT / 4.0 * T_bc_t0**4 * np.pi * 0.5**2
    print(f'    Expected boundary power at t=0: {E_bc_per_ns:.4e} GJ/ns')
    print('=' * 72)

    # ── Material layout plot ──────────────────────────────────────────────────
    plot_material_layout(r_centers, z_centers, run_tag, r_edges=r_edges, z_edges=z_edges)

    # ── Boundary temperature ramp ────────────────────────────────────────────
    def boundary_temperature(t):
        """Linearly ramp T_bc from bc_t_start to bc_t_end over bc_ramp_time."""
        if bc_ramp_time <= 0.0:
            return bc_t_end
        frac = float(np.clip(t / bc_ramp_time, 0.0, 1.0))
        return bc_t_start + (bc_t_end - bc_t_start) * frac

    # Mutable reference so the boundary_source_func can see the current time
    # without being re-created every step.
    _t_now = [0.0]

    def boundary_source_func(r, z, side):
        """Position-dependent SURFACE blackbody source.

        Called by _sample_boundary_rz for each r-cell on the zmin face.
        Returns the blackbody emission temperature (keV); 0.0 means no emission.
        Emission weight = a*c/4 * T^4 * annular_area * dt  (surface flux, NOT volumetric).
        Source region: z=0 face, r < 0.5 cm only.
        """
        if side == 'zmin' and r < 0.5:
            return boundary_temperature(_t_now[0])
        return 0.0

    # Precompute expected boundary emission per ns for the source region.
    # E_bc = (a*c/4) * T_bc^4 * pi * r_source^2 * dt
    _r_source = 0.5  # cm

    # ── Fiducial monitor points ───────────────────────────────────────────────
    fiducial_points = {
        'r=0.00 z=0.25': (0.00, 0.25),
        'r=0.00 z=2.75': (0.00, 2.75),
        'r=1.25 z=3.50': (1.25, 3.50),
        'r=0.00 z=4.25': (0.00, 4.25),
        'r=0.00 z=6.75': (0.00, 6.75),
    }
    fiducial_indices = {}
    for label, (rv, zv) in fiducial_points.items():
        ii = int(np.argmin(np.abs(r_centers - rv)))
        jj = int(np.argmin(np.abs(z_centers - zv)))
        fiducial_indices[label] = (ii, jj)

    # ── Initialise state ──────────────────────────────────────────────────────
    T_init_2d  = np.full((nr_actual, nz_actual), T_INIT)
    Tr_init_2d = np.full((nr_actual, nz_actual), T_INIT)
    source_2d  = np.zeros((nr_actual, nz_actual))  # no volumetric source

    state = init_simulation(
        Ntarget,
        T_init_2d,
        Tr_init_2d,
        r_edges,
        z_edges,
        energy_edges,
        eos_func,
        inv_eos_func,
        geometry='rz',
    )

    # T_boundary all zero — boundary emission is handled by boundary_source_func.
    T_boundary = (0.0, 0.0, 0.0, 0.0)
    # Reflect at r=0 (symmetry axis); all other faces are vacuum.
    reflect = (True, False, False, False)

    # ── Restart or fresh start ────────────────────────────────────────────────
    if restart_file is not None:
        print(f'\nLoading restart from: {restart_file}')
        payload = load_checkpoint(restart_file)
        # Validate geometry compatibility
        saved_meta = payload['metadata']
        if saved_meta.get('n_groups') != n_groups:
            raise ValueError(
                f"Checkpoint n_groups={saved_meta['n_groups']} != {n_groups}"
            )
        # Restore state
        state = _deserialize_state(payload['state'])
        step_count       = int(payload['step_count'])
        current_dt       = float(payload['current_dt'])
        times            = list(payload['times'])
        fiducial_data    = {k: list(v) for k, v in payload['fiducial_data'].items()}
        fiducial_data_rad= {k: list(v) for k, v in payload['fiducial_data_rad'].items()}
        output_times_saved = set(payload['output_times_saved'])
        np.random.set_state(payload['np_random_state'])
        random.setstate(payload['py_random_state'])
        print(f'  Resumed at t = {state.time:.6e} ns, step {step_count}')
        print(f'  Census particles: {len(state.weights)}')
        print(f'  Current dt: {current_dt:.6e} ns')
    else:
        step_count        = 0
        current_dt        = dt_initial
        times             = [0.0]
        fiducial_data     = {label: [T_INIT] for label in fiducial_points}
        fiducial_data_rad = {label: [T_INIT] for label in fiducial_points}
        output_times_saved = set()
        print('\nStarting from initial conditions.')
        # Print fiducial cell mapping for reference
        for label, (ii, jj) in fiducial_indices.items():
            print(f'  Fiducial {label} -> cell ({ii},{jj})'
                  f' at (r={r_centers[ii]:.3f}, z={z_centers[jj]:.3f})')

    print(f'\nCheckpoint file: {checkpoint_file}')
    if checkpoint_every > 0:
        print(f'Periodic checkpoints every {checkpoint_every} steps')

    metadata = {
        'n_groups':        n_groups,
        'nr':              nr_actual,
        'nz':              nz_actual,
        'mesh_tag':        mesh_tag,
        'Ntarget':         Ntarget,
        'Nboundary':       Nboundary,
        'Ntotal':          Ntotal,
        'Ntotal_T_floor':  Ntotal_T_floor,
        'Nmax':            Nmax,
        'bc_t_start':      bc_t_start,
        'bc_t_end':        bc_t_end,
        'bc_ramp_time':    bc_ramp_time,
        'dt_initial':      dt_initial,
        'dt_max':          dt_max,
        'dt_increase_factor': dt_increase_factor,
    }

    # ── Time evolution ────────────────────────────────────────────────────────
    print('\nTime stepping …')
    print(f"{'Time':>12}  {'N_par':>8}  {'E_mat':>14}  {'E_rad':>14}")
    print('-' * 56)

    wall_start = _time.perf_counter()

    while state.time < t_final - 1e-12:
        step_dt = min(current_dt, t_final - state.time)

        # Snap to output time if necessary.
        hit_output_time = False
        for tout in output_times:
            if tout not in output_times_saved and state.time < tout <= state.time + step_dt:
                step_dt = tout - state.time
                hit_output_time = True
                break

        # Update time reference used inside boundary_source_func.
        _t_now[0] = float(state.time)

        # --- advance one step ---
        state, info = step(
            state=state,
            Ntarget=Ntarget,
            Nboundary=Nboundary,
            Ntotal=Ntotal,
            Ntotal_T_floor=Ntotal_T_floor,
            Nsource=0,
            Nmax=Nmax,
            T_boundary=T_boundary,
            dt=step_dt,
            edges1=r_edges,
            edges2=z_edges,
            energy_edges=energy_edges,
            sigma_a_funcs=sigma_a_funcs,
            inv_eos=inv_eos_func,
            cv=cv_func,
            source=source_2d,
            reflect=reflect,
            geometry='rz',
            max_events_per_particle=max_events_per_particle,
            boundary_source_func=boundary_source_func,
        )

        step_count += 1

        # Record history
        times.append(float(state.time))
        for label, (ii, jj) in fiducial_indices.items():
            fiducial_data[label].append(float(state.temperature[ii, jj]))
            fiducial_data_rad[label].append(float(state.radiation_temperature[ii, jj]))

        # ── Per-step diagnostic ────────────────────────────────────────────
        e_mat = info.get('total_internal_energy', float('nan'))
        e_rad = info.get('total_radiation_energy', float('nan'))
        n_par = info.get('N_particles', len(state.weights))
        actual_be   = info.get('boundary_emission', float('nan'))
        T_bc_now    = boundary_temperature(_t_now[0])
        expected_be = A_RAD * C_LIGHT / 4.0 * T_bc_now**4 * (np.pi * _r_source**2) * step_dt
        ratio_be    = actual_be / expected_be if expected_be > 0 else float('nan')
        split_str = ''
        if Ntotal > 0:
            nb_act = info.get('N_boundary', 0)
            nt_act = info.get('N_target', 0)
            ebc = info.get('E_boundary_est', float('nan'))
            emt = info.get('E_material_est', float('nan'))
            frac = ebc / (ebc + emt) if (ebc + emt) > 0 else float('nan')
            split_str = f'  [split] Nbc={nb_act} Nmat={nt_act} f_bc={frac:.3f}'
        print(f'{state.time:10.6f} ns  N={n_par:8d}  '
              f'E_mat={e_mat:.4e}  E_rad={e_rad:.4e}  '
              f'[BC] T_bc={T_bc_now:.4f}keV  '
              f'expected={expected_be:.4e}  actual={actual_be:.4e}  '
              f'ratio={ratio_be:.3f}{split_str}')

        # Output at requested times
        for tout in output_times:
            if tout not in output_times_saved and abs(state.time - tout) < 1e-9:
                print(f'  >> Snapshot at t = {state.time:.4f} ns')
                plot_solution(
                    state.temperature.copy(),
                    state.radiation_temperature.copy(),
                    r_centers, z_centers,
                    state.time,
                    save_prefix=f'crooked_pipe_mg_imc_{run_tag}',
                    r_edges=r_edges,
                    z_edges=z_edges,
                    T_bc=T_bc_now,
                )
                output_times_saved.add(tout)
                # Always checkpoint at output times
                print(f'  >> Checkpoint -> {checkpoint_file}')
                save_checkpoint(
                    checkpoint_file, state, step_count, current_dt,
                    times, fiducial_data, fiducial_data_rad,
                    output_times_saved, metadata,
                )
                break

        # Periodic checkpoint (if not already written above)
        if (checkpoint_every > 0
                and step_count % checkpoint_every == 0
                and not any(abs(state.time - tout) < 1e-9
                            for tout in output_times
                            if tout not in output_times_saved)):
            print(f'  >> Periodic checkpoint (step {step_count}) -> {checkpoint_file}')
            save_checkpoint(
                checkpoint_file, state, step_count, current_dt,
                times, fiducial_data, fiducial_data_rad,
                output_times_saved, metadata,
            )

        if not hit_output_time:
            current_dt = min(current_dt * dt_increase_factor, dt_max)

    # ── Post-processing ───────────────────────────────────────────────────────
    times_arr = np.array(times)
    fiducial_data    = {k: np.array(v) for k, v in fiducial_data.items()}
    fiducial_data_rad= {k: np.array(v) for k, v in fiducial_data_rad.items()}

    wall_elapsed = _time.perf_counter() - wall_start
    print(f'\nFinished. t_final = {state.time:.4f} ns, '
          f'{step_count} steps, wall = {wall_elapsed:.1f} s')

    # Save final solution
    npz_file = f'crooked_pipe_mg_imc_solution_{run_tag}.npz'
    np.savez(
        npz_file,
        r_centers=r_centers,
        z_centers=z_centers,
        energy_edges=energy_edges,
        T_final=state.temperature,
        Tr_final=state.radiation_temperature,
        times=times_arr,
        wall_elapsed_s=np.float64(wall_elapsed),
        **{f'fid_{k}': v for k, v in fiducial_data.items()},
        **{f'fid_rad_{k}': v for k, v in fiducial_data_rad.items()},
    )
    print(f'Saved solution: {npz_file}')

    # Save final radiation energy by group if available
    if hasattr(state, 'radiation_energy_by_group') and state.radiation_energy_by_group is not None:
        np.save(
            f'crooked_pipe_mg_imc_Erad_bygroup_{run_tag}.npy',
            state.radiation_energy_by_group,
        )

    print('\nPlotting fiducial history …')
    plot_fiducial_history(times_arr, fiducial_data, fiducial_data_rad, run_tag)

    print('\nFinal fiducial temperatures:')
    for label, (ii, jj) in fiducial_indices.items():
        print(f'  {label}: T_mat={state.temperature[ii,jj]:.4f}'
              f'  T_rad={state.radiation_temperature[ii,jj]:.4f}  keV')

    return state


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND-LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run the multigroup IMC crooked-pipe test problem.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--n-groups',   type=int,   default=10,
                        help='Number of energy groups')
    parser.add_argument('--nr',         type=int,   default=60,
                        help='Coarse r-direction cells')
    parser.add_argument('--nz',         type=int,   default=210,
                        help='Coarse z-direction cells')
    parser.add_argument('--dt-initial', type=float, default=1e-4,
                        help='Initial time step (ns)')
    parser.add_argument('--dt-max',     type=float, default=0.01,
                        help='Maximum time step (ns)')
    parser.add_argument('--dt-growth',  type=float, default=1.1,
                        help='Time-step growth factor per step')
    parser.add_argument('--Ntarget',    type=int,   default=200000,
                        help='Target emission particles per step (ignored when --Ntotal > 0)')
    parser.add_argument('--Nboundary',  type=int,   default=200000,
                        help='Boundary source particles per step (ignored when --Ntotal > 0)')
    parser.add_argument('--Ntotal',     type=int,   default=0,
                        help='If > 0, total new particles per step split proportionally '
                             'between boundary and material emission by energy. '
                             'Overrides --Ntarget and --Nboundary.')
    parser.add_argument('--Ntotal-T-floor', type=float, default=0.0,
                        help='When using --Ntotal, exclude material cells at or below '
                             'this temperature (keV) from the emission estimate. '
                             'E.g. set to 1.1*T_init=0.011 to ignore cold material. '
                             'Split is always clamped to [0.1, 0.9].')
    parser.add_argument('--Nmax',       type=int,   default=1000000,
                        help='Maximum census particles after combing')
    parser.add_argument('--bc-t-start', type=float, default=0.05,
                        help='Boundary source T at t=0 (keV)')
    parser.add_argument('--bc-t-end',   type=float, default=0.5,
                        help='Boundary source T after ramp (keV)')
    parser.add_argument('--bc-ramp-time', type=float, default=20.0,
                        help='Duration of boundary temperature ramp (ns)')
    parser.add_argument('--output-times', type=str, default=None,
                        help='Comma-separated output times in ns '
                             '(e.g. "1,5,10,50,100")')
    parser.add_argument('--use-refined-mesh', action='store_true',
                        help='Apply log-refinement near material interfaces')
    parser.add_argument('--n-refine',   type=int,   default=10,
                        help='Sub-cells per coarse cell in refinement zones')
    parser.add_argument('--n-coarse-r', type=int,   default=None,
                        help='Coarse r cells before refinement (default: --nr)')
    parser.add_argument('--n-coarse-z', type=int,   default=None,
                        help='Coarse z cells before refinement (default: --nz)')
    parser.add_argument('--refine-width', type=float, default=0.05,
                        help='Half-width of refinement zone around interfaces (cm)')
    parser.add_argument('--max-events', type=int,   default=10**9,
                        help='Maximum collision events per particle per step')
    parser.add_argument('--restart-file', type=str, default=None,
                        help='Checkpoint (.pkl) to restart from')
    parser.add_argument('--checkpoint-file', type=str, default=None,
                        help='Checkpoint output path (auto if omitted)')
    parser.add_argument('--checkpoint-every', type=int, default=0,
                        help='Periodic checkpoint cadence in steps (0=off)')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_arguments()

    if args.output_times:
        output_times = [float(t) for t in args.output_times.split(',')]
    else:
        output_times = [0.001, 0.01, 0.1, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0, 200.0, 500.0, 1000.0]

    main(
        n_groups=args.n_groups,
        output_times=output_times,
        nr=args.nr,
        nz=args.nz,
        dt_initial=args.dt_initial,
        dt_max=args.dt_max,
        dt_increase_factor=args.dt_growth,
        Ntarget=args.Ntarget,
        Nboundary=args.Nboundary,
        Ntotal=args.Ntotal,
        Ntotal_T_floor=args.Ntotal_T_floor,
        Nmax=args.Nmax,
        bc_t_start=args.bc_t_start,
        bc_t_end=args.bc_t_end,
        bc_ramp_time=args.bc_ramp_time,
        use_refined_mesh=args.use_refined_mesh,
        n_refine=args.n_refine,
        n_coarse_r=args.n_coarse_r,
        n_coarse_z=args.n_coarse_z,
        refine_width=args.refine_width,
        max_events_per_particle=args.max_events,
        restart_file=args.restart_file,
        checkpoint_file=args.checkpoint_file,
        checkpoint_every=args.checkpoint_every,
    )
