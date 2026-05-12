#!/usr/bin/env python3
"""
plot_crooked_pipe_noneq_solutions.py — Regenerate all figures from saved NPZ data.

Reads the NPZ files produced by the multigroup non-equilibrium crooked pipe solver
and reproduces:
  - Material/radiation temperature colormaps at every saved output time
  - Fiducial-point temperature history (material and radiation)
  - Radiation spectra at every fiducial point for every output snapshot

Usage
-----
Run from the directory containing the NPZ files:

    python3 plot_crooked_pipe_noneq_solutions.py \\
        --prefix crooked_pipe_10g_noneq_refined_114x282

The script auto-discovers files matching the pattern
``{prefix}_snapshot_t_*.npz``, ``{prefix}_spectra.npz``, and
``{prefix}_solution_*.npz`` in the current working directory.

Options
-------
--prefix        File-name prefix (required).
--no-snapshots  Skip per-time colormaps.
--no-history    Skip fiducial temperature history.
--no-spectra    Skip radiation spectra plots.
--show-mesh     Overlay mesh lines on colormaps.
"""
import argparse
import glob
import os
import sys

import numpy as np

# ---------------------------------------------------------------------------
# Make utils/plotfuncs importable regardless of cwd
# ---------------------------------------------------------------------------
_script_dir = os.path.dirname(os.path.abspath(__file__))
_utils_dir = os.path.join(_script_dir, 'utils')
if _utils_dir not in sys.path:
    sys.path.insert(0, _utils_dir)

import matplotlib
matplotlib.use('Agg')   # headless; avoids display requirement
import matplotlib.pyplot as plt
from plotfuncs import show


# ---------------------------------------------------------------------------
# Colourmap snapshots
# ---------------------------------------------------------------------------

def plot_snapshot(snap_file, save_prefix, show_mesh=False):
    """Produce material-temperature and radiation-temperature PNG colormaps."""
    d = np.load(snap_file, allow_pickle=False)
    T_2d      = d['T_2d']
    T_rad_2d  = d['T_rad_2d']
    r_centers = d['r_centers']
    z_centers = d['z_centers']
    T_bc_raw  = float(d['T_bc'])
    T_bc      = None if np.isnan(T_bc_raw) else T_bc_raw
    t         = float(d['time'])

    # Use the data-minimum as vmin (cells at T_init are the coldest)
    T_init_approx = float(T_2d.min())

    _R, Z = np.meshgrid(r_centers, z_centers, indexing='ij')   # noqa: F841

    # ---- Plot 1: Material temperature ----
    max_T  = float(T_2d.max())
    vmax_T = np.ceil(max_T * 100) / 100.0
    vmax_T = max(vmax_T, T_init_approx + 0.01)

    fig1, ax1 = plt.subplots(1, 1, figsize=(8, 3 * 1.275))
    im1 = ax1.pcolormesh(Z, _R, T_2d, shading='auto', cmap='plasma',
                         vmin=T_init_approx, vmax=vmax_T)
    ax1.set_xlabel('z (cm)', fontsize=12)
    ax1.set_ylabel('r (cm)', fontsize=12)
    cbar1 = plt.colorbar(im1, ax=ax1, orientation='horizontal',
                          location='top', pad=0.15, label='T (keV)')
    ax1.set_aspect('equal')

    if show_mesh and 'r_faces' in d and 'z_faces' in d:
        r_faces = d['r_faces']
        z_faces = d['z_faces']
        n_lines = 15
        r_step = max(1, len(r_faces) // n_lines)
        z_step = max(1, len(z_faces) // n_lines)
        for r in r_faces[::r_step]:
            ax1.axhline(r, color='white', alpha=0.2, linewidth=0.3)
        for z in z_faces[::z_step]:
            ax1.axvline(z, color='white', alpha=0.2, linewidth=0.3)

    plt.tight_layout()
    fname1 = f'{save_prefix}_material_t_{t:.5f}ns.png'
    show(fname1, close_after=True, cbar_ax=cbar1.ax)
    print(f'Saved: {fname1}')

    # ---- Plot 2: Radiation temperature ----
    max_T_rad  = float(T_rad_2d.max())
    vmax_T_rad = np.ceil(max_T_rad * 100) / 100.0
    if T_bc is not None:
        vmax_T_rad = min(vmax_T_rad, 1.1 * T_bc)
    vmax_T_rad = max(vmax_T_rad, T_init_approx + 0.01)

    fig2, ax2 = plt.subplots(1, 1, figsize=(8, 3 * 1.275))
    im2 = ax2.pcolormesh(Z, _R, T_rad_2d, shading='auto', cmap='plasma',
                          vmin=T_init_approx, vmax=vmax_T_rad)
    ax2.set_xlabel('z (cm)', fontsize=12)
    ax2.set_ylabel('r (cm)', fontsize=12)
    cbar2 = plt.colorbar(im2, ax=ax2, orientation='horizontal',
                          location='top', pad=0.15,
                          label=r'$T_\mathrm{r}$ (keV)')
    ax2.set_aspect('equal')

    if show_mesh and 'r_faces' in d and 'z_faces' in d:
        r_faces = d['r_faces']
        z_faces = d['z_faces']
        for r in r_faces[::r_step]:
            ax2.axhline(r, color='white', alpha=0.2, linewidth=0.3)
        for z in z_faces[::z_step]:
            ax2.axvline(z, color='white', alpha=0.2, linewidth=0.3)

    plt.tight_layout()
    fname2 = f'{save_prefix}_radiation_t_{t:.5f}ns.png'
    show(fname2, close_after=True, cbar_ax=cbar2.ax)
    print(f'Saved: {fname2}')


# ---------------------------------------------------------------------------
# Fiducial-point temperature history
# ---------------------------------------------------------------------------

def plot_fiducial_history(solution_file):
    """Read solution NPZ and plot material / radiation temperature histories."""
    d = np.load(solution_file, allow_pickle=True)
    times        = d['times']
    fiducial_data = d['fiducial_data'].item()
    n_groups = len(d['energy_edges']) - 1

    markers = ['o', 's', '^', 'd', 'v']
    colors  = ['blue', 'red', 'green', 'purple', 'orange']

    # ---- Material temperature ----
    fig1, ax1 = plt.subplots(1, 1, figsize=(10, 7))
    for idx, (label, data) in enumerate(fiducial_data.items()):
        ax1.loglog(times, data['T_mat'],
                   marker=markers[idx % len(markers)],
                   color=colors[idx % len(colors)],
                   linewidth=2, markersize=6,
                   markevery=max(1, len(times) // 20),
                   label=label, alpha=0.8)
    ax1.set_xlabel('Time (ns)', fontsize=13, fontweight='bold')
    ax1.set_ylabel('Material Temperature T (keV)', fontsize=13, fontweight='bold')
    ax1.legend(fontsize=10, loc='best', framealpha=0.9)
    ax1.grid(True, which='both', alpha=0.3, linestyle='--')
    ax1.grid(True, which='minor', alpha=0.15, linestyle=':')
    plt.tight_layout()
    fname1 = f'crooked_pipe_{n_groups}g_noneq_fiducial_history_material.pdf'
    show(fname1, close_after=True)
    print(f'Saved: {fname1}')

    # ---- Radiation temperature ----
    fig2, ax2 = plt.subplots(1, 1, figsize=(10, 7))
    for idx, (label, data) in enumerate(fiducial_data.items()):
        ax2.loglog(times, data['T_rad'],
                   marker=markers[idx % len(markers)],
                   color=colors[idx % len(colors)],
                   linewidth=2, markersize=6,
                   markevery=max(1, len(times) // 20),
                   label=label, alpha=0.8)
    ax2.set_xlabel('Time (ns)', fontsize=13, fontweight='bold')
    ax2.set_ylabel(r'Radiation Temperature $T_r$ (keV)',
                   fontsize=13, fontweight='bold')
    ax2.legend(fontsize=10, loc='best', framealpha=0.9)
    ax2.grid(True, which='both', alpha=0.3, linestyle='--')
    ax2.grid(True, which='minor', alpha=0.15, linestyle=':')
    plt.tight_layout()
    fname2 = f'crooked_pipe_{n_groups}g_noneq_fiducial_history_radiation.pdf'
    show(fname2, close_after=True)
    print(f'Saved: {fname2}')


# ---------------------------------------------------------------------------
# Radiation spectra at fiducial points
# ---------------------------------------------------------------------------

def plot_spectra(spectra_file, save_prefix):
    """Plot radiation spectra at each fiducial point for every output snapshot."""
    d = np.load(spectra_file, allow_pickle=False)
    output_times   = d['output_times']   # (n_snaps,)
    print("Found times for spectra snapshots:", output_times)
    labels         = d['labels']          # (n_pts,)
    spectra        = d['spectra']         # (n_snaps, n_pts, n_groups)
    energy_edges   = d['energy_edges']
    energy_centers = d['energy_centers']

    n_snaps = len(output_times)
    cmap = plt.cm.plasma
    snap_colors = [cmap(i / max(n_snaps - 1, 1)) for i in range(n_snaps)]

    for pt_idx, pt_label in enumerate(labels):
        fig, ax = plt.subplots(figsize=(6, 4.5))
        for si, (col, t_s) in enumerate(zip(snap_colors, output_times)):
            spec = spectra[si, pt_idx, :]
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

        # Parse label for a readable title
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

        all_vals = spectra[:, pt_idx, :].ravel()
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
                norm=plt.Normalize(vmin=output_times[0], vmax=output_times[-1]))
            sm.set_array([])
            fig.colorbar(sm, ax=ax, label='time (ns)')
        plt.tight_layout()
        outname = f'{save_prefix}_spectra_pt{pt_idx + 1}.pdf'
        show(outname, close_after=True)
        print(f'Saved: {outname}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description='Generate crooked-pipe non-eq figures from saved NPZ files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument('--prefix', required=True,
                   help='File-name prefix, e.g. crooked_pipe_10g_noneq_refined_114x282')
    p.add_argument('--no-snapshots', action='store_true',
                   help='Skip per-time temperature colormaps')
    p.add_argument('--no-history', action='store_true',
                   help='Skip fiducial temperature history')
    p.add_argument('--no-spectra', action='store_true',
                   help='Skip radiation spectra plots')
    p.add_argument('--show-mesh', action='store_true',
                   help='Overlay mesh lines on colormaps')
    return p.parse_args()


def main():
    args = parse_args()
    prefix = args.prefix

    # ---- Colourmap snapshots ----
    if not args.no_snapshots:
        pattern = f'{prefix}_snapshot_t_*.npz'
        snap_files = sorted(glob.glob(pattern))
        if snap_files:
            print(f'\nPlotting {len(snap_files)} snapshot(s)...')
            for snap_file in snap_files:
                plot_snapshot(snap_file, prefix, show_mesh=args.show_mesh)
        else:
            print(f'No snapshot files found matching: {pattern}')

    # ---- Fiducial history ----
    if not args.no_history:
        sol_pattern = f'{prefix}_solution_*.npz'
        # Also try the naming convention that omits mesh dimensions in prefix
        sol_files = sorted(glob.glob(sol_pattern))
        if not sol_files:
            # Try without trailing mesh-tag: crooked_pipe_Xg_noneq_solution_MESH.npz
            base = prefix.rsplit('_', 2)[0] if prefix.count('_') >= 2 else prefix
            sol_files = sorted(glob.glob(f'{base}_solution_*.npz'))
        if sol_files:
            sol_file = sol_files[-1]   # use most recent
            print(f'\nPlotting fiducial history from: {sol_file}')
            plot_fiducial_history(sol_file)
        else:
            print(f'No solution NPZ found matching: {sol_pattern}')

    # ---- Spectra ----
    if not args.no_spectra:
        spectra_file = f'{prefix}_spectra.npz'
        if os.path.exists(spectra_file):
            print(f'\nPlotting spectra from: {spectra_file}')
            plot_spectra(spectra_file, prefix)
        else:
            print(f'Spectra file not found: {spectra_file}')

    print('\nDone.')


if __name__ == '__main__':
    main()
