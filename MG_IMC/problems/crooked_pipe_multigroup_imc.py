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

Geometry — U-shaped (reversed) crooked pipe:
  Inner leg  (r < 0.5):          thin channel up from z=0 (inlet)
  Bend       (r < 1.5, z > 2.5): connects inner to outer leg at the top
  Outer leg  (1.0 <= r < 1.5):   return channel back down to z=0
  Thick wall (0.5 <= r < 1.0, z <= 2.5): separates the two legs
  Outer wall (r >= 1.5):         always thick

    
Initial condition: T = Tr = T_INIT = 0.01 keV everywhere
Fiducial monitor points:
    (r=0.0,  z=0.25),  (r=0.0,  z=2.25), (r=1.25, z=3.5),
    (r=0.0, z=4.25),  (r=0.0, z=6.75)
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

try:
    from planck_integrals import Bg_multigroup
    _PLANCK_AVAILABLE = True
except ImportError:
    _PLANCK_AVAILABLE = False

# ── Path setup ──────────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # problems -> MG_IMC -> RadTranBook
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
T_INIT    = 0.05   # keV — cold initial condition
RHO_THICK = 2.0    # g/cm^3 — optically thick regions
RHO_THIN  = 0.1 #0.01   # g/cm^3 — optically thin regions
CV_MASS   = 0.05   # GJ/(g*keV) — mass-specific heat capacity

CHECKPOINT_VERSION = 1


# ═══════════════════════════════════════════════════════════════════════════════
# COLOUR-TEMPERATURE HELPERS (MATCH DILUTE-SPHERE FIT METHOD)
# ═══════════════════════════════════════════════════════════════════════════════

def _planck_group_integral_fallback(E_low, E_high, T):
    """Planck group integral using fixed quadrature when planck_integrals is unavailable."""
    if T <= 0:
        return 0.0
    E = np.linspace(E_low, E_high, 60)
    B = (2.0 * E**3 / C_LIGHT**2) / (np.exp(np.clip(E / T, 0, 500)) - 1 + 1e-300)
    if hasattr(np, 'trapezoid'):
        return float(np.trapezoid(B, E))
    return float(np.trapz(B, E))


def _Bg_all(energy_edges, T):
    """Return array of Planck group integrals at temperature T."""
    if _PLANCK_AVAILABLE:
        return Bg_multigroup(energy_edges, T)
    n = len(energy_edges) - 1
    return np.array([
        _planck_group_integral_fallback(energy_edges[g], energy_edges[g + 1], T)
        for g in range(n)
    ])


def _peak_nu_from_spec(spec, nu_c):
    """Return parabolic-interpolation peak frequency of a discrete spectrum."""
    if np.all(spec <= 0):
        return np.nan
    i = int(np.argmax(spec))
    nu_peak = nu_c[i]
    if 0 < i < len(spec) - 1 and spec[i - 1] > 0 and spec[i + 1] > 0:
        xl, yl = np.log(nu_c[i - 1]), np.log(spec[i - 1])
        xm, ym = np.log(nu_c[i]),     np.log(spec[i])
        xr, yr = np.log(nu_c[i + 1]), np.log(spec[i + 1])
        denom = (xl - xm) * (xl - xr) * (xm - xr)
        a = (xr * (ym - yl) + xm * (yl - yr) + xl * (yr - ym)) / denom
        b = (xr**2 * (yl - ym) + xm**2 * (yr - yl) + xl**2 * (ym - yr)) / denom
        if a < 0:
            x_peak = -b / (2.0 * a)
            if xl <= x_peak <= xr:
                nu_peak = np.exp(x_peak)
    return float(nu_peak)


_PLANCK_PEAK_CACHE = {}


def _get_planck_peak_lookup(energy_edges, T_min=0.05, T_max=15.0, n_pts=300):
    """Build (or fetch) lookup table mapping T -> nu_peak(B_g/DeltaE_g)."""
    key = tuple(np.asarray(energy_edges, dtype=float))
    if key not in _PLANCK_PEAK_CACHE:
        dE = energy_edges[1:] - energy_edges[:-1]
        nu_c = np.sqrt(energy_edges[:-1] * energy_edges[1:])
        T_arr = np.logspace(np.log10(T_min), np.log10(T_max), n_pts)
        nu_peaks = np.array([
            _peak_nu_from_spec(_Bg_all(energy_edges, T) / np.where(dE > 0, dE, 1e-300), nu_c)
            for T in T_arr
        ])
        _PLANCK_PEAK_CACHE[key] = (T_arr, nu_peaks)
    return _PLANCK_PEAK_CACHE[key]


def fit_color_temperature(E_g, energy_edges, T_min=0.05, T_max=15.0):
    """Estimate colour temperature from group energies using discrete-peak matching."""
    if len(energy_edges) == 2:
        E_tot = float(E_g[0])
        if E_tot <= 0:
            return np.nan
        return float(np.clip((E_tot / A_RAD) ** 0.25, T_min, T_max))

    dE = energy_edges[1:] - energy_edges[:-1]
    spec = np.asarray(E_g, dtype=float) / np.where(dE > 0, dE, 1e-300)
    if np.all(spec <= 0):
        return np.nan

    nu_c = np.sqrt(energy_edges[:-1] * energy_edges[1:])
    nu_peak_sim = _peak_nu_from_spec(spec, nu_c)
    if not np.isfinite(nu_peak_sim):
        return np.nan

    T_arr, nu_peak_B = _get_planck_peak_lookup(energy_edges, T_min, T_max)
    if nu_peak_sim <= nu_peak_B[0]:
        return float(T_min)
    if nu_peak_sim >= nu_peak_B[-1]:
        return float(T_max)
    T_c = float(np.interp(nu_peak_sim, nu_peak_B, T_arr))
    return float(np.clip(T_c, T_min, T_max))


# ═══════════════════════════════════════════════════════════════════════════════
# MATERIAL GEOMETRY
# ═══════════════════════════════════════════════════════════════════════════════

def is_optically_thick(r, z):
    """Return True where (r,z) is in an optically thick region.

     Crooked-pipe thick/thin map matching the diffusion implementation.
     Thin regions are defined as:
        1) lower_thin:
            r < 0.5 and (z < 3.0 or z > 4.0)
        2) ascending_left_thin:
            r < 1.5 and 2.5 < z < 3.0
        3) ascending_right_thin:
            r < 1.5 and 4.0 < z < 4.5
        4) top_thin:
            1.0 <= r < 1.5 and 2.5 < z < 4.5

    Everything else is thick.

    Vectorised over array inputs.
    """
    scalar_input = np.isscalar(r) and np.isscalar(z)
    r = np.atleast_1d(np.asarray(r, dtype=float))
    z = np.atleast_1d(np.asarray(z, dtype=float))

    result = np.ones_like(r, dtype=bool)  # default: thick

    lower_thin = (r < 0.5) & ((z < 3.0) | (z > 4.0))
    ascending_left_thin = (r < 1.5) & (z > 2.5) & (z < 3.0)
    ascending_right_thin = (r < 1.5) & (z > 4.0) & (z < 4.5)
    top_thin = (r >= 1.0) & (r < 1.5) & (z > 2.5) & (z < 4.5)

    result[lower_thin] = False
    result[ascending_left_thin] = False
    result[ascending_right_thin] = False
    result[top_thin] = False

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
    #if rho < 1, make result = 1.0 to avoid zero-heat-capacity issues in the thin regions.
    rho = np.where(rho < 1.0e-2, 1.0/CV_MASS, rho)
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

def plot_solution(T_mat, T_rad, T_col, r_centers, z_centers, time_value, save_prefix,
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

    cbar_labels = {
        'material': 'T (keV)',
        'radiation': r'$T_\mathrm{r}$ (keV)',
        'color': r'$T_\mathrm{c}$ (keV)',
    }

    for T_field, tag in [(T_mat, 'material'), (T_rad, 'radiation'), (T_col, 'color')]:
        vmax = np.ceil(T_field.max() * 100) / 100.0
        if T_bc is not None:
            vmax = min(vmax, 1.1 * T_bc)
        vmax = max(vmax, vmin + 0.01)  # ensure vmax > vmin

        fig, ax = plt.subplots(1, 1, figsize=(5.5, 3 * 1.275))
        im = ax.pcolormesh(
            ZE, RE, T_field,
            vmin=vmin, vmax=vmax,
            cmap='plasma',
            shading='flat',
        )
        ax.set_xlabel('z (cm)')
        ax.set_ylabel('r (cm)')
        ax.set_aspect('equal')
        from mpl_toolkits.axes_grid1 import make_axes_locatable
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('top', size='5%', pad=0.3)
        cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
        cax.xaxis.set_ticks_position('top')
        cax.xaxis.set_label_position('top')
        cax.set_xlabel(cbar_labels[tag])
        plt.tight_layout()
        fname = f'{save_prefix}_{tag}_t_{time_value:.5f}ns.png'
        fig.savefig(fname, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'    Saved {fname}')


def plot_fiducial_history(times, fiducial_data, fiducial_data_rad, fiducial_data_col,
                          mesh_tag, out_dir='.'):
    """Temperature vs time at fiducial monitor points (mat, rad, colour)."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=False)
    colors = plt.cm.tab10(np.linspace(0, 1, len(fiducial_data)))

    for ax, data_dict, ylabel in zip(
        axes,
        [fiducial_data, fiducial_data_rad, fiducial_data_col],
        ['Material Temperature (keV)', 'Radiation Temperature (keV)', 'Color Temperature (keV)'],
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
    fname = os.path.join(out_dir, f'crooked_pipe_mg_imc_fiducial_{mesh_tag}.png')
    fig.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'    Saved {fname}')


def _plot_intermediate_spectra(spectra_snapshots, spectrum_times, energy_edges,
                                labels, out_prefix):
    """Plot radiation spectra at all fiducial points for every snapshot so far.
    Produces one PDF per fiducial point (overwritten at each output time).
    Curves are coloured early→late using the plasma colormap.
    """
    if not spectra_snapshots:
        return
    n_snaps = len(spectra_snapshots)
    energy_centers = 0.5 * (energy_edges[:-1] + energy_edges[1:])
    cmap = plt.cm.plasma
    snap_colors = [cmap(i / max(n_snaps - 1, 1)) for i in range(n_snaps)]
    for pt_idx, pt_label in enumerate(labels):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for si, (col, t_s) in enumerate(zip(snap_colors, spectrum_times)):
            spec = spectra_snapshots[si][pt_idx, :]
            ax.stairs(spec, energy_edges, baseline=None,
                      color=col, linewidth=1.8, label=f't = {t_s:.3g} ns')
            mask = spec > 0
            if np.any(mask):
                ax.scatter(energy_centers[mask], spec[mask],
                           color=col, s=18, zorder=5)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Photon energy $E$ (keV)', fontsize=12)
        ax.set_ylabel(r'$dE_r/dE$ (GJ cm$^{-3}$ keV$^{-1}$)', fontsize=11)
        raw = str(pt_label).strip()
        try:
            parts = raw.split()
            r_part = next(p for p in parts if p.startswith('r='))
            z_part = next(p for p in parts if p.startswith('z='))
            rv = r_part.split('=')[1].strip(',')
            zv = z_part.split('=')[1].strip(',')
            title = fr'$(r,z)=({rv},{zv})$ cm'
        except Exception:
            title = raw
        ax.set_title(title, fontsize=12)
        all_vals = np.concatenate([spectra_snapshots[si][pt_idx, :]
                                   for si in range(n_snaps)])
        pos_vals = all_vals[all_vals > 0]
        if pos_vals.size > 0:
            ymax = float(np.max(pos_vals)) * 3.0
            ymin = max(float(np.min(pos_vals)) / 100.0, ymax * 1e-12)
            ax.set_ylim(ymin, ymax)
        ax.set_xlim(energy_edges[0] * 0.9, energy_edges[-1] * 1.1)
        ax.grid(True, which='both', alpha=0.25, linestyle='--')
        ax.grid(True, which='minor', alpha=0.12, linestyle=':')
        if n_snaps <= 12:
            ax.legend(fontsize=8, loc='best')
        else:
            sm = plt.cm.ScalarMappable(
                cmap=cmap,
                norm=plt.Normalize(vmin=spectrum_times[0], vmax=spectrum_times[-1]))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label='time (ns)')
        plt.tight_layout()
        outname = f'{out_prefix}_spectra_intermediate_pt{pt_idx + 1}.pdf'
        show(outname, close_after=True)
        print(f'    Saved {outname}')


def plot_material_layout(r_centers, z_centers, run_tag, r_edges=None, z_edges=None, out_dir='.'):
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
    fname = os.path.join(out_dir, f'crooked_pipe_mg_imc_material_{run_tag}.png')
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)


def plot_mesh(r_edges, z_edges, run_tag, out_dir='.'):
    """Plot the computational mesh for quick geometry sanity checks."""
    fig, ax = plt.subplots(figsize=(10, 6))
    for r in r_edges:
        ax.plot([z_edges[0], z_edges[-1]], [r, r], color='black', linewidth=0.3, alpha=0.5)
    for z in z_edges:
        ax.plot([z, z], [r_edges[0], r_edges[-1]], color='black', linewidth=0.3, alpha=0.5)

    ax.set_xlabel('z (cm)')
    ax.set_ylabel('r (cm)')
    ax.set_title(f'Computational mesh ({len(r_edges)-1} x {len(z_edges)-1} cells)')
    ax.set_aspect('equal')
    plt.tight_layout()
    fname = os.path.join(out_dir, f'crooked_pipe_mg_imc_mesh_{run_tag}.png')
    fig.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)


def save_snapshot(state, t, save_prefix, energy_edges, r_edges, z_edges, T_col_2d=None, T_bc=None):
    """Save 2D field arrays at an output time to NPZ — same format as diffusion snapshots.

    Writes {save_prefix}_snapshot_t_{t:.5f}ns.npz with keys matching the diffusion
    version so plot_crooked_pipe_noneq_solutions.py can read both.
    """
    E_r_groups_3d = state.radiation_energy_by_group  # (n_groups, nr, nz) GJ/cm^3
    if E_r_groups_3d is None:
        n_g = len(energy_edges) - 1
        E_r_groups_3d = np.zeros((n_g, *state.temperature.shape))
    Er_2d  = np.sum(E_r_groups_3d, axis=0)           # total E_r (GJ/cm^3)
    phi_2d = Er_2d * C_LIGHT                          # scalar flux = c * E_r

    snap_file = f'{save_prefix}_snapshot_t_{t:.5f}ns.npz'
    np.savez_compressed(
        snap_file,
        T_2d=state.temperature,
        T_rad_2d=state.radiation_temperature,
        T_col_2d=(T_col_2d if T_col_2d is not None else np.full_like(state.temperature, np.nan)),
        phi_2d=phi_2d,
        Er_2d=Er_2d,
        E_r_groups_3d=E_r_groups_3d,
        r_centers=0.5 * (r_edges[:-1] + r_edges[1:]),
        z_centers=0.5 * (z_edges[:-1] + z_edges[1:]),
        r_faces=r_edges,
        z_faces=z_edges,
        time=np.float64(t),
        T_bc=np.float64(T_bc if T_bc is not None else np.nan),
        energy_edges=energy_edges,
    )
    print(f'    Saved snapshot: {snap_file}')


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
                    fiducial_data_rad, fiducial_data_col, output_times_saved, metadata):
    """Persist full simulation state so the run can be resumed after a crash."""
    payload = {
        'checkpoint_version':    CHECKPOINT_VERSION,
        'state':                 _serialize_state(state),
        'step_count':            int(step_count),
        'current_dt':            float(current_dt),
        'times':                 list(times),
        'fiducial_data':         {k: list(v) for k, v in fiducial_data.items()},
        'fiducial_data_rad':     {k: list(v) for k, v in fiducial_data_rad.items()},
        'fiducial_data_col':     {k: list(v) for k, v in fiducial_data_col.items()},
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
    mode='standard',
    n_groups=10,
    output_times=None,
    nr=40,
    nz=140,
    dt_initial=1e-4,
    dt_max=0.01,
    dt_increase_factor=1.1,
    Ntarget=None,
    Nboundary=None,
    Ntotal=0,
    Ntotal_T_floor=0.0,
    particle_budget_fmin=None,
    T_emit_floor=0.0,
    Nmax=None,
    Nmax_growth=None,
    Nmax_final=None,
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
    mode : {'quick', 'standard', 'publication'}
        Particle-count preset. Explicit CLI overrides still take precedence.
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
    Ntarget : int or None
        Target emission particles per step (used when Ntotal=0).
        If None, taken from *mode* preset.
    Nboundary : int or None
        Boundary source particles per step (used when Ntotal=0).
        If None, taken from *mode* preset.
    Ntotal : int
        If > 0, override Ntarget/Nboundary and split this total budget
        between boundary and material emission proportional to the energy
        emitted from each source every step.
    Ntotal_T_floor : float
        When using Ntotal mode, material cells at or below this temperature
        (keV) are excluded from the material emission estimate so that a cold
        bulk domain does not dilute the boundary share.  Set to
        ``1.1 * T_INIT`` (or similar) to ignore unheated material.
    particle_budget_fmin : float or None
        Minimum fraction per source channel in Ntotal mode. For a two-channel
        split (boundary/material), this is clamped to [0, 0.5]. If None,
        taken from *mode* preset.
    Nmax : int or None
        Initial census cap after combing. If None, taken from *mode* preset.
    Nmax_growth : int or None
        Census-cap growth per step. If None, taken from *mode* preset.
    Nmax_final : int or None
        Maximum census cap reached by growth. If None, taken from *mode* preset.
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

    # Mode-based particle presets (same pattern used by dilute-sphere scripts).
    mode_params = {
        'quick': {
            'Ntarget': 20_000,
            'Nboundary': 20_000,
            'Nmax_init': 100_000,
            'Nmax_growth': 10_000,
            'Nmax_final': 500_000,
        },
        'standard': {
            'Ntarget': 200_000,
            'Nboundary': 200_000,
            'Nmax_init': 1_000_000,
            'Nmax_growth': 50_000,
            'Nmax_final': 5_000_000,
        },
        'publication': {
            'Ntarget': 1_000_000,
            'Nboundary': 1_000_000,
            'Nmax_init': 5_000_000,
            'Nmax_growth': 200_000,
            'Nmax_final': 30_000_000,
        },
    }
    if mode not in mode_params:
        raise ValueError(f"Unknown mode {mode!r}; choose from {list(mode_params)}")
    p = mode_params[mode]
    if Ntarget is None:
        Ntarget = p['Ntarget']
    if Nboundary is None:
        Nboundary = p['Nboundary']
    if Nmax is None:
        Nmax = p['Nmax_init']
    if Nmax_growth is None:
        Nmax_growth = p['Nmax_growth']
    if Nmax_final is None:
        Nmax_final = p['Nmax_final']
    if particle_budget_fmin is None:
        particle_budget_fmin = 0.1

    Ntarget = int(Ntarget)
    Nboundary = int(Nboundary)
    Nmax = int(Nmax)
    Nmax_growth = int(Nmax_growth)
    Nmax_final = int(Nmax_final)
    particle_budget_fmin = float(np.clip(particle_budget_fmin, 0.0, 0.5))
    if Nmax_final < Nmax:
        Nmax_final = Nmax

    # ── Energy groups ────────────────────────────────────────────────────────
    # Range [5e-3 keV, 10 keV] on a log scale 
    energy_edges = np.logspace(np.log10(1e-2), np.log10(10), n_groups + 1)

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

    # ── Output directory, run tag, and checkpoint filename ───────────────────
    run_tag = f'{n_groups}g_{mesh_tag}_{nr_actual}x{nz_actual}'
    out_tag = f'imc_{n_groups}g_{mode}_{mesh_tag}_{nr_actual}x{nz_actual}'
    out_dir = os.path.join('results', 'crooked_pipe_multigroup_imc', out_tag)
    os.makedirs(out_dir, exist_ok=True)
    if checkpoint_file is None:
        checkpoint_file = os.path.join(out_dir, 'checkpoint.pkl')

    # ── Print run summary ─────────────────────────────────────────────────────
    print('=' * 72)
    print('CROOKED PIPE — MULTIGROUP IMC  2D Cylindrical (r-z)')
    print('=' * 72)
    print(f'  mode      = {mode}')
    print(f'  n_groups  = {n_groups}')
    print(f'  mesh      = {nr_actual} x {nz_actual}  ({mesh_tag})')
    if Ntotal > 0:
        floor_str = f', T_floor={Ntotal_T_floor:.4g} keV' if Ntotal_T_floor > 0 else ''
        print(f'  Ntotal    = {Ntotal}  (energy-proportional split{floor_str}), '
              f'Nmax = {Nmax} -> {Nmax_final} (growth {Nmax_growth}/step)')
        print(f'  split fmin= {particle_budget_fmin:.3f}  '
              f'(each channel >= {100.0 * particle_budget_fmin:.1f}% of Ntotal)')
    else:
        print(f'  Ntarget   = {Ntarget},  Nboundary = {Nboundary}, '
              f'Nmax = {Nmax} -> {Nmax_final} (growth {Nmax_growth}/step)')
    print(f'  dt_init   = {dt_initial} ns,  dt_max = {dt_max} ns,  growth = {dt_increase_factor}')
    print(f'  T_bc ramp : {bc_t_start} → {bc_t_end} keV over {bc_ramp_time} ns')
    print(f'  output_times: {output_times}  (t_final={t_final} ns)')
    print(f'  output dir: {out_dir}')
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
    plot_material_layout(r_centers, z_centers, run_tag,
                         r_edges=r_edges, z_edges=z_edges, out_dir=out_dir)
    plot_mesh(r_edges, z_edges, run_tag, out_dir=out_dir)

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
        'Point 1: r=0.0, z=0.25': (0.0, 0.25),
        'Point 2: r=0.0, z=2.75': (0.0, 2.75),
        'Point 3: r=1.25, z=3.5': (1.25, 3.5),
        'Point 4: r=0.0, z=4.25': (0.0, 4.25),
        'Point 5: r=0.0, z=6.75': (0.0, 6.75)
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
        T_emit_floor=T_emit_floor,
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
        fiducial_data_col= payload.get('fiducial_data_col')
        if fiducial_data_col is None:
            fiducial_data_col = {label: [T_INIT] * len(times) for label in fiducial_points}
        else:
            fiducial_data_col = {k: list(v) for k, v in fiducial_data_col.items()}
        output_times_saved = set(payload['output_times_saved'])
        np.random.set_state(payload['np_random_state'])
        random.setstate(payload['py_random_state'])
        Nmax_current = int(saved_meta.get('Nmax_current', Nmax))
        print(f'  Resumed at t = {state.time:.6e} ns, step {step_count}')
        print(f'  Census particles: {len(state.weights)}')
        print(f'  Current dt: {current_dt:.6e} ns')
        print(f'  Current Nmax: {Nmax_current}')
    else:
        step_count        = 0
        current_dt        = dt_initial
        times             = [0.0]
        fiducial_data     = {label: [T_INIT] for label in fiducial_points}
        fiducial_data_rad = {label: [T_INIT] for label in fiducial_points}
        fiducial_data_col = {label: [T_INIT] for label in fiducial_points}
        output_times_saved = set()
        Nmax_current      = int(Nmax)
        print('\nStarting from initial conditions.')
        # Print fiducial cell mapping for reference
        for label, (ii, jj) in fiducial_indices.items():
            print(f'  Fiducial {label} -> cell ({ii},{jj})'
                  f' at (r={r_centers[ii]:.3f}, z={z_centers[jj]:.3f})')

    print(f'\nCheckpoint file: {checkpoint_file}')
    if checkpoint_every > 0:
        print(f'Periodic checkpoints every {checkpoint_every} steps')

    metadata = {
        'mode':            mode,
        'n_groups':        n_groups,
        'nr':              nr_actual,
        'nz':              nz_actual,
        'mesh_tag':        mesh_tag,
        'Ntarget':         Ntarget,
        'Nboundary':       Nboundary,
        'Ntotal':          Ntotal,
        'Ntotal_T_floor':  Ntotal_T_floor,
        'particle_budget_fmin': particle_budget_fmin,
        'T_emit_floor':    T_emit_floor,
        'Nmax':            Nmax,
        'Nmax_growth':     Nmax_growth,
        'Nmax_final':      Nmax_final,
        'Nmax_current':    int(Nmax_current),
        'bc_t_start':      bc_t_start,
        'bc_t_end':        bc_t_end,
        'bc_ramp_time':    bc_ramp_time,
        'dt_initial':      dt_initial,
        'dt_max':          dt_max,
        'dt_increase_factor': dt_increase_factor,
    }

    # ── Time evolution ────────────────────────────────────────────────────────
    print('\nTime stepping …')
    print(f"{'Step':>6}  {'t (ns)':>9}  {'N_part':>8}  {'N_bc':>7}  "
          f"{'E_bc':>12}  {'E_tot':>12}  {'E_int':>12}  {'E_rad':>12}  "
          f"{'Resid':>10}")
    print('-' * 105)

    wall_start = _time.perf_counter()
    dE = energy_edges[1:] - energy_edges[:-1]          # group widths (keV)
    # Reload previously accumulated spectra if the file already exists
    # (handles restarts where the in-memory lists are lost between runs).
    _spectra_file = os.path.join(out_dir, f'crooked_pipe_mg_imc_{run_tag}_spectra.npz')
    fiducial_spectra_snapshots: list = []
    spectrum_output_times: list = []
    Tc_col_2d = np.full_like(state.temperature, np.nan)
    if os.path.exists(_spectra_file):
        try:
            _prev = np.load(_spectra_file, allow_pickle=True)
            for _pt, _sp in zip(list(_prev['output_times']), list(_prev['spectra'])):
                if _pt < state.time - 1e-12:
                    spectrum_output_times.append(float(_pt))
                    fiducial_spectra_snapshots.append(np.array(_sp))
            if spectrum_output_times:
                print(f'  Restored {len(spectrum_output_times)} spectra snapshot(s) '
                      f'from {_spectra_file} (up to t={spectrum_output_times[-1]:.5f} ns)')
        except Exception as _e:
            print(f'  Warning: could not reload {_spectra_file}: {_e}')

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
        t0_step = _time.perf_counter()
        state, info = step(
            state=state,
            Ntarget=Ntarget,
            Nboundary=Nboundary,
            Ntotal=Ntotal,
            Ntotal_T_floor=Ntotal_T_floor,
            particle_budget_fmin=particle_budget_fmin,
            T_emit_floor=T_emit_floor,
            Nsource=0,
            Nmax=Nmax_current,
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
            Nmax_growth=Nmax_growth,
            Nmax_final=Nmax_final,
        )
        wall_step = _time.perf_counter() - t0_step

        if 'Nmax_next' in info:
            Nmax_current = int(info['Nmax_next'])
        else:
            Nmax_current = min(Nmax_current + Nmax_growth, Nmax_final)
        metadata['Nmax_current'] = int(Nmax_current)

        step_count += 1

        # Record history
        times.append(float(state.time))
        for label, (ii, jj) in fiducial_indices.items():
            fiducial_data[label].append(float(state.temperature[ii, jj]))
            fiducial_data_rad[label].append(float(state.radiation_temperature[ii, jj]))
            if state.radiation_energy_by_group is not None:
                tc = fit_color_temperature(state.radiation_energy_by_group[:, ii, jj], energy_edges)
            else:
                tc = np.nan
            fiducial_data_col[label].append(float(tc) if np.isfinite(tc) else np.nan)

        # ── Per-step diagnostic ────────────────────────────────────────────
        e_mat = info.get('total_internal_energy', float('nan'))
        e_rad = info.get('total_radiation_energy', float('nan'))
        e_tot = info.get('total_energy', float('nan'))
        e_res = info.get('energy_residual', float('nan'))
        n_par = info.get('N_particles', len(state.weights))
        actual_be   = info.get('boundary_emission', float('nan'))
        T_bc_now    = boundary_temperature(_t_now[0])
        n_bc = info.get('N_boundary', 0)
        split_str = ''
        if Ntotal > 0:
            nb_act = info.get('N_boundary', 0)
            nt_act = info.get('N_target', 0)
            ebc = info.get('E_boundary_est', float('nan'))
            emt = info.get('E_material_est', float('nan'))
            frac = ebc / (ebc + emt) if (ebc + emt) > 0 else float('nan')
            split_str = f'  [split] Nbc={nb_act} Nmat={nt_act} f_bc={frac:.3f}'
        print(f"{step_count:>6d}  {state.time:>9.5f}  "
              f"{n_par:>8d}  {n_bc:>7d}  "
              f"{actual_be:>12.5e}  {e_tot:>12.5e}  {e_mat:>12.5e}  {e_rad:>12.5e}  "
              f"{e_res:>10.3e}  [{wall_step:.1f}s]{split_str}")

        # Output at requested times
        for tout in output_times:
            if tout not in output_times_saved and abs(state.time - tout) < 1e-9:
                print(f'  >> Snapshot at t = {state.time:.4f} ns')
                if state.radiation_energy_by_group is not None:
                    n_g, nr_loc, nz_loc = state.radiation_energy_by_group.shape
                    Tc_col_2d = np.full((nr_loc, nz_loc), np.nan)
                    for ii in range(nr_loc):
                        for jj in range(nz_loc):
                            Tc_col_2d[ii, jj] = fit_color_temperature(
                                state.radiation_energy_by_group[:, ii, jj], energy_edges
                            )
                else:
                    Tc_col_2d = np.full_like(state.temperature, np.nan)
                plot_solution(
                    state.temperature.copy(),
                    state.radiation_temperature.copy(),
                    Tc_col_2d.copy(),
                    r_centers, z_centers,
                    state.time,
                    save_prefix=os.path.join(out_dir, f'crooked_pipe_mg_imc_{run_tag}'),
                    r_edges=r_edges,
                    z_edges=z_edges,
                    T_bc=T_bc_now,
                )
                save_snapshot(
                    state, state.time,
                    save_prefix=os.path.join(out_dir, f'crooked_pipe_mg_imc_{run_tag}'),
                    energy_edges=energy_edges,
                    r_edges=r_edges,
                    z_edges=z_edges,
                    T_col_2d=Tc_col_2d,
                    T_bc=T_bc_now,
                )
                output_times_saved.add(tout)
                # Collect radiation spectrum at fiducial points (GJ/cm^3/keV).
                if state.radiation_energy_by_group is not None:
                    snap = np.array([
                        state.radiation_energy_by_group[:, ii, jj] / dE
                        for (ii, jj) in fiducial_indices.values()
                    ])  # shape (n_points, n_groups)
                    fiducial_spectra_snapshots.append(snap)
                    spectrum_output_times.append(float(state.time))
                    # Intermediate spectrum plots (overwritten at each output time)
                    _plot_intermediate_spectra(
                        fiducial_spectra_snapshots, spectrum_output_times,
                        energy_edges, list(fiducial_indices.keys()),
                        os.path.join(out_dir, f'crooked_pipe_mg_imc_{run_tag}'),
                    )
                    # Save accumulated spectra NPZ (same format as diffusion version)
                    _sf = os.path.join(out_dir, f'crooked_pipe_mg_imc_{run_tag}_spectra.npz')
                    np.savez_compressed(
                        _sf,
                        output_times=np.array(spectrum_output_times),
                        labels=np.array(list(fiducial_indices.keys())),
                        spectra=np.array(fiducial_spectra_snapshots),
                        energy_edges=energy_edges,
                        energy_centers=0.5 * (energy_edges[:-1] + energy_edges[1:]),
                    )
                    print(f'    Saved spectra:  {_sf}')
                # Intermediate fiducial history plot
                plot_fiducial_history(times, fiducial_data, fiducial_data_rad,
                                     fiducial_data_col, run_tag, out_dir=out_dir)
                # Save accumulated fiducial history NPZ (same format as diffusion version)
                _fid_labels = list(fiducial_data.keys())
                _fhf = os.path.join(out_dir, f'crooked_pipe_mg_imc_{run_tag}_fiducial_history.npz')
                np.savez_compressed(
                    _fhf,
                    times=np.array(times),
                    labels=np.array(_fid_labels),
                    T_mat=np.array([fiducial_data[l] for l in _fid_labels]),
                    T_rad=np.array([fiducial_data_rad[l] for l in _fid_labels]),
                    T_col=np.array([fiducial_data_col[l] for l in _fid_labels]),
                )
                print(f'    Saved fiducial: {_fhf}')
                # Always checkpoint at output times
                print(f'  >> Checkpoint -> {checkpoint_file}')
                save_checkpoint(
                    checkpoint_file, state, step_count, current_dt,
                    times, fiducial_data, fiducial_data_rad, fiducial_data_col,
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
                times, fiducial_data, fiducial_data_rad, fiducial_data_col,
                output_times_saved, metadata,
            )

        if not hit_output_time:
            current_dt = min(current_dt * dt_increase_factor, dt_max)

    # ── Post-processing ───────────────────────────────────────────────────────
    times_arr = np.array(times)
    fiducial_data    = {k: np.array(v) for k, v in fiducial_data.items()}
    fiducial_data_rad= {k: np.array(v) for k, v in fiducial_data_rad.items()}
    fiducial_data_col= {k: np.array(v) for k, v in fiducial_data_col.items()}

    wall_elapsed = _time.perf_counter() - wall_start
    print(f'\nFinished. t_final = {state.time:.4f} ns, '
          f'{step_count} steps, wall = {wall_elapsed:.1f} s')

    # Save final solution
    npz_file = os.path.join(out_dir, f'crooked_pipe_mg_imc_solution_{run_tag}.npz')
    np.savez(
        npz_file,
        r_centers=r_centers,
        z_centers=z_centers,
        energy_edges=energy_edges,
        T_final=state.temperature,
        Tr_final=state.radiation_temperature,
        Tc_final=Tc_col_2d,
        times=times_arr,
        wall_elapsed_s=np.float64(wall_elapsed),
        **{f'fid_{k}': v for k, v in fiducial_data.items()},
        **{f'fid_rad_{k}': v for k, v in fiducial_data_rad.items()},
        **{f'fid_col_{k}': v for k, v in fiducial_data_col.items()},
    )
    print(f'Saved solution: {npz_file}')

    # Save fiducial time-series in a dedicated file for post-run plotting.
    # Arrays:
    #   times        — 1-D, shape (n_steps,)
    #   labels       — 1-D string array, shape (n_points,)
    #   T_mat        — 2-D, shape (n_points, n_steps)
    #   T_rad        — 2-D, shape (n_points, n_steps)
    fiducial_labels_list = list(fiducial_data.keys())
    fid_T_mat = np.array([fiducial_data[lbl]     for lbl in fiducial_labels_list])
    fid_T_rad = np.array([fiducial_data_rad[lbl] for lbl in fiducial_labels_list])
    fid_T_col = np.array([fiducial_data_col[lbl] for lbl in fiducial_labels_list])
    fid_npz = os.path.join(out_dir, f'crooked_pipe_mg_imc_fiducial_{run_tag}.npz')
    np.savez(
        fid_npz,
        times=times_arr,
        labels=np.array(fiducial_labels_list),
        T_mat=fid_T_mat,
        T_rad=fid_T_rad,
        T_col=fid_T_col,
    )
    print(f'Saved fiducial history: {fid_npz}')

    # Save radiation spectrum at fiducial points for each output snapshot.
    # Array layout:  spectra[snapshot, point, group]  units: GJ/cm^3/keV
    if fiducial_spectra_snapshots:
        spectra_npz = os.path.join(out_dir, f'crooked_pipe_mg_imc_spectra_{run_tag}.npz')
        np.savez(
            spectra_npz,
            output_times=np.array(spectrum_output_times),
            labels=np.array(list(fiducial_indices.keys())),
            spectra=np.array(fiducial_spectra_snapshots),
            energy_edges=energy_edges,
            energy_centers=0.5 * (energy_edges[:-1] + energy_edges[1:]),
        )
        print(f'Saved fiducial spectra: {spectra_npz}')

    # Save final radiation energy by group if available
    if hasattr(state, 'radiation_energy_by_group') and state.radiation_energy_by_group is not None:
        np.save(
            os.path.join(out_dir, f'crooked_pipe_mg_imc_Erad_bygroup_{run_tag}.npy'),
            state.radiation_energy_by_group,
        )

    print('\nPlotting fiducial history …')
    plot_fiducial_history(times_arr, fiducial_data, fiducial_data_rad,
                         fiducial_data_col, run_tag, out_dir=out_dir)

    print('\nFinal fiducial temperatures:')
    for label, (ii, jj) in fiducial_indices.items():
        if state.radiation_energy_by_group is not None:
            tc_fin = fit_color_temperature(state.radiation_energy_by_group[:, ii, jj], energy_edges)
        else:
            tc_fin = np.nan
        print(f'  {label}: T_mat={state.temperature[ii,jj]:.4f}'
              f'  T_rad={state.radiation_temperature[ii,jj]:.4f}'
              f'  T_col={tc_fin:.4f}  keV')

    return state


# ═══════════════════════════════════════════════════════════════════════════════
# COMMAND-LINE INTERFACE
# ═══════════════════════════════════════════════════════════════════════════════

def parse_arguments():
    parser = argparse.ArgumentParser(
        description='Run the multigroup IMC crooked-pipe test problem.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--mode', choices=['quick', 'standard', 'publication'],
                        default='standard',
                        help='Particle-count preset (Ntarget/Nboundary/Nmax growth)')
    parser.add_argument('--n-groups',   type=int,   default=10,
                        help='Number of energy groups')
    parser.add_argument('--nr',         type=int,   default=40,
                        help='Coarse r-direction cells')
    parser.add_argument('--nz',         type=int,   default=140,
                        help='Coarse z-direction cells')
    parser.add_argument('--dt-initial', type=float, default=1e-4,
                        help='Initial time step (ns)')
    parser.add_argument('--dt-max',     type=float, default=0.01,
                        help='Maximum time step (ns)')
    parser.add_argument('--dt-growth',  type=float, default=1.1,
                        help='Time-step growth factor per step')
    parser.add_argument('--Ntarget',    type=int,   default=None,
                        help='Override mode target emission particles per step '
                            '(ignored when --Ntotal > 0)')
    parser.add_argument('--Nboundary',  type=int,   default=None,
                        help='Override mode boundary source particles per step '
                            '(ignored when --Ntotal > 0)')
    parser.add_argument('--Ntotal',     type=int,   default=0,
                        help='If > 0, total new particles per step split proportionally '
                             'between boundary and material emission by energy. '
                             'Overrides --Ntarget and --Nboundary.')
    parser.add_argument('--Ntotal-T-floor', type=float, default=0.0,
                        help='When using --Ntotal, exclude material cells at or below '
                             'this temperature (keV) from the emission estimate. '
                             'E.g. set to 1.1*T_init=0.011 to ignore cold material. '
                             'Split floor is controlled by --particle-budget-fmin.')
    parser.add_argument('--particle-budget-fmin', type=float, default=None,
                        help='Minimum split fraction for each source channel in '
                             'Ntotal mode (default 0.1; clamped to [0, 0.5]).')
    parser.add_argument('--T-emit-floor', type=float, default=0.075,
                        help='Emission temperature floor (keV). Cells with T below this '
                             'value contribute zero emission energy and no particles are '
                             'sourced from them. Default 0 (disabled).')
    parser.add_argument('--Nmax',       type=int,   default=None,
                        help='Override mode initial census cap after combing')
    parser.add_argument('--Nmax-growth', type=int,  default=None,
                        help='Override mode census-cap growth per step')
    parser.add_argument('--Nmax-final', type=int,   default=None,
                        help='Override mode maximum census cap')
    parser.add_argument('--bc-t-start', type=float, default=0.5,
                        help='Boundary source T at t=0 (keV)')
    parser.add_argument('--bc-t-end',   type=float, default=0.5,
                        help='Boundary source T after ramp (keV)')
    parser.add_argument('--bc-ramp-time', type=float, default=1.0,
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
    parser.add_argument('--refine-width', type=float, default=0.01,
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
        mode=args.mode,
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
        particle_budget_fmin=args.particle_budget_fmin,
        T_emit_floor=args.T_emit_floor,
        Nmax=args.Nmax,
        Nmax_growth=args.Nmax_growth,
        Nmax_final=args.Nmax_final,
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
