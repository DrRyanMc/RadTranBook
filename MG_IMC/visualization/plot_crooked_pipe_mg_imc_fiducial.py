#!/usr/bin/env python3
"""
Plot fiducial-point temperature histories and group spectra from a
multigroup IMC crooked-pipe run.

Two kinds of output:
  1. Fiducial history  — T_mat(t) and T_rad(t) at each monitor point,
                         matching the style used for the diffusion runs.
  2. Spectra           — dE_r/dE vs photon energy at each fiducial point,
                         three curves per figure: t = 10, 20, 50 ns.

Usage (from workspace root):
    python MG_IMC/plot_crooked_pipe_mg_imc_fiducial.py
    python MG_IMC/plot_crooked_pipe_mg_imc_fiducial.py \\
        --fiducial  crooked_pipe_mg_imc_fiducial_10g_refined_114x282.npz \\
        --spectra   crooked_pipe_mg_imc_spectra_10g_refined_114x282.npz \\
        --output-base  crooked_pipe_mg_imc_10g_refined_114x282
"""

import sys
import os
import argparse

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup  (script lives in MG_IMC/)
# ---------------------------------------------------------------------------
_script_dir   = os.path.dirname(os.path.abspath(__file__))
_project_root = os.path.dirname(os.path.dirname(_script_dir))  # visualization -> MG_IMC -> RadTranBook
for _p in (_project_root,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.plotfuncs import show, font

# Physical constants
_A_RAD = 0.01372       # GJ / (cm^3 · keV^4)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
_COLORS    = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
_LINESTYLES = ['-', '--', '-.', ':']
_MARKERS   = ['o', 's', '^', 'd', 'v']

# Times at which to plot spectra
_SPECTRUM_TIMES = [10.0, 20.0, 50.0]   # ns

# Colors for the three spectrum snapshots
_SPEC_COLORS = ['tab:blue', 'tab:orange', 'tab:red']


def planck_spectral_density(E_keV, T_keV):
    """
    Equilibrium spectral radiation-energy density:
        dE_r/dE = a_rad * (15/pi^4) * T^3 * x^3/(exp(x)-1)   [GJ/cm^3/keV]
    where x = E/T.

    Safe against T -> 0 or E -> 0.
    """
    T_safe = np.maximum(T_keV, 1e-10)
    x = E_keV / T_safe
    # Avoid overflow in exp for large x
    with np.errstate(over='ignore', invalid='ignore'):
        planck = np.where(
            x < 500.0,
            x**3 / (np.exp(np.minimum(x, 500.0)) - 1.0),
            0.0,
        )
    return _A_RAD * (15.0 / np.pi**4) * T_safe**3 * planck


# ---------------------------------------------------------------------------
# Load helpers
# ---------------------------------------------------------------------------
def load_fiducial(path):
    """Return (times, labels, T_mat, T_rad) from an IMC fiducial NPZ."""
    d = np.load(path, allow_pickle=True)
    times  = d['times']
    labels = list(d['labels'])
    T_mat  = d['T_mat']   # (n_points, n_steps)
    T_rad  = d['T_rad']
    return times, labels, T_mat, T_rad


def load_spectra(path):
    """Return (output_times, labels, spectra, energy_edges, energy_centers)."""
    d = np.load(path, allow_pickle=True)
    return (
        d['output_times'],
        list(d['labels']),
        d['spectra'],        # (n_snapshots, n_points, n_groups)  GJ/cm^3/keV
        d['energy_edges'],
        d['energy_centers'],
    )


# ---------------------------------------------------------------------------
# Fiducial history plots  (diffusion style)
# ---------------------------------------------------------------------------
def _pretty_label(raw):
    """Turn 'r=0.00 z=2.75' into '(r,z) = (0.00, 2.75) cm'."""
    raw = raw.strip()
    try:
        parts = raw.split()
        rv = parts[0].split('=')[1]
        zv = parts[1].split('=')[1]
        return fr'$(r,z)=({rv},{zv})$ cm'
    except Exception:
        return raw


def plot_fiducial_history(times, labels, T_mat, T_rad, outbase):
    """
    Reproduce the diffusion-style panel layout:
      • one PDF with one subplot per fiducial point  (material T)
      • one PDF with one subplot per fiducial point  (radiation T)
      • one combined PDF for material T (all points on a single axis)
      • one combined PDF for radiation T
    """
    n_pts = len(labels)
    pretty = [_pretty_label(lb) for lb in labels]

    # Restrict to t > 0 so loglog is safe
    mask   = times > 0
    t_plot = times[mask]

    quantities = [
        (T_mat[:, mask], r'$T$ (keV)',              'material'),
        (T_rad[:, mask], r'$T_\mathrm{r}$ (keV)',   'radiation'),
    ]

    # ── Per-point panel figures ────────────────────────────────────────────
    for vals, ylabel, suffix in quantities:
        fig, axes = plt.subplots(
            1, n_pts,
            figsize=(3.5 * n_pts, 4.5),
            sharey=True,
        )
        if n_pts == 1:
            axes = [axes]

        for ax, label, color, marker, row in zip(
            axes, pretty, _COLORS, _MARKERS, vals
        ):
            ax.loglog(
                t_plot, row,
                color=color,
                linestyle='-',
                linewidth=2,
                marker=marker,
                markersize=3,
                markevery=max(1, len(t_plot) // 15),
                alpha=0.9,
                label=label,
            )
            ax.set_xlabel('time (ns)', fontsize=11)
            if ax is axes[0]:
                ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, which='both', alpha=0.25, linestyle='--')
            ax.grid(True, which='minor', alpha=0.12, linestyle=':')
            leg = ax.legend(prop=font, facecolor='white', edgecolor='none',
                            fontsize=9)
            leg.get_frame().set_alpha(None)

        plt.tight_layout()
        outname = f'{outbase}_fiducial_{suffix}.pdf'
        show(outname, close_after=True)
        print(f'Saved: {outname}')

    # ── Combined figures  (all points on one axis) ─────────────────────────
    for vals, ylabel, suffix in quantities:
        fig, ax = plt.subplots(figsize=(8, 5))

        for label, color, marker, row in zip(pretty, _COLORS, _MARKERS, vals):
            ax.loglog(
                t_plot, row,
                color=color,
                linestyle='-',
                linewidth=2,
                marker=marker,
                markersize=3,
                markevery=max(1, len(t_plot) // 15),
                alpha=0.9,
                label=label,
            )

        ax.set_xlabel('time (ns)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, which='both', alpha=0.25, linestyle='--')
        ax.grid(True, which='minor', alpha=0.12, linestyle=':')
        leg = ax.legend(prop=font, facecolor='white', edgecolor='none',
                        fontsize=10, loc='best')
        leg.get_frame().set_alpha(None)

        plt.tight_layout()
        outname = f'{outbase}_fiducial_{suffix}_combined.pdf'
        show(outname, close_after=True)
        print(f'Saved: {outname}')


# ---------------------------------------------------------------------------
# Spectrum plots
# ---------------------------------------------------------------------------
def plot_spectra(out_times, labels, spectra, energy_edges, energy_centers,
                 fid_times, fid_T_mat, fid_labels, outbase,
                 target_times=None):
    """
    One figure per fiducial point.  Each figure shows dE_r/dE vs E for
    `target_times` (default: [10, 20, 50] ns), plus an equilibrium Planck
    curve at the material temperature for reference.
    """
    if target_times is None:
        target_times = _SPECTRUM_TIMES

    n_pts    = len(labels)
    dE       = energy_edges[1:] - energy_edges[:-1]   # group widths (keV)
    E_fine   = np.logspace(np.log10(energy_edges[0]),
                           np.log10(energy_edges[-1]), 300)

    # For each target time, find the closest snapshot in out_times
    snap_indices = []
    snap_labels  = []
    for t_want in target_times:
        idx = int(np.argmin(np.abs(out_times - t_want)))
        snap_indices.append(idx)
        snap_labels.append(fr'$t = {out_times[idx]:.0f}$ ns')

    pretty_pts = [_pretty_label(lb) for lb in labels]

    for pt_idx in range(n_pts):
        fig, ax = plt.subplots(figsize=(6, 4.5))

        for snap_i, snap_label, color in zip(snap_indices, snap_labels, _SPEC_COLORS):
            spec_vals = spectra[snap_i, pt_idx, :]   # (n_groups,)  GJ/cm^3/keV

            # Step histogram style: plot at group centers with horizontal spans
            # spanning each group, showing the piecewise-constant spectrum
            ax.stairs(
                spec_vals,
                energy_edges,
                baseline=None,
                color=color,
                linewidth=1.8,
                label=snap_label,
            )

            # Scatter markers at group centers for clarity
            mask = spec_vals > 0
            if np.any(mask):
                ax.scatter(
                    energy_centers[mask], spec_vals[mask],
                    color=color, s=20, zorder=5,
                )

            # Equilibrium Planck curve at the material T at this fiducial point
            # Use the average T_mat around the snapshot time
            t_snap = out_times[snap_i]
            t_diff = np.abs(fid_times - t_snap)
            closest_step = int(np.argmin(t_diff))
            T_mat_val = float(fid_T_mat[pt_idx, closest_step])

            if T_mat_val > 1e-6:
                planck_vals = planck_spectral_density(E_fine, T_mat_val)
                ax.plot(
                    E_fine, planck_vals,
                    color=color,
                    linestyle=':',
                    linewidth=1.2,
                    alpha=0.6,
                )

        # Legend entry for Planck reference (once)
        ax.plot([], [], color='gray', linestyle=':', linewidth=1.2,
                label=r'$B(E,T_\mathrm{mat})$ (equil.)')

        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlabel(r'Photon energy $E$ (keV)', fontsize=12)
        ax.set_ylabel(r'$\mathrm{d}E_r/\mathrm{d}E$ (GJ cm$^{-3}$ keV$^{-1}$)',
                      fontsize=11)
        ax.set_title(pretty_pts[pt_idx], fontsize=12)
        ax.set_xlim(energy_edges[0] * 0.9, energy_edges[-1] * 1.1)

        # Clamp y-axis: show at most 12 decades; ignore pure-zero groups
        all_vals = np.concatenate([
            spectra[snap_i, pt_idx, :] for snap_i in snap_indices
        ])
        pos_vals = all_vals[all_vals > 0]
        if pos_vals.size > 0:
            ymax = float(np.max(pos_vals)) * 3.0
            ymin = max(float(np.min(pos_vals)) / 100.0, ymax * 1e-12)
            ax.set_ylim(ymin, ymax)
        
        ax.grid(True, which='both', alpha=0.25, linestyle='--')
        ax.grid(True, which='minor', alpha=0.12, linestyle=':')
        leg = ax.legend(prop=font, facecolor='white', edgecolor='none',
                        fontsize=9, loc='best')
        leg.get_frame().set_alpha(None)

        plt.tight_layout()
        # Build a short tag from the point label (strip spaces / special chars)
        pt_tag = pretty_pts[pt_idx].replace('$', '').replace(' ', '').replace(
            ',', '_').replace('(', '').replace(')', '').replace('=', '')
        outname = f'{outbase}_spectrum_pt{pt_idx+1}.pdf'
        show(outname, close_after=True)
        print(f'Saved: {outname}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description='Plot fiducial histories and spectra from MG-IMC crooked-pipe run.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--fiducial', type=str,
        default=os.path.join(_project_root,
                             'crooked_pipe_mg_imc_fiducial_10g_refined_114x282.npz'),
        help='NPZ file with fiducial temperature histories.',
    )
    parser.add_argument(
        '--spectra', type=str,
        default=os.path.join(_project_root,
                             'crooked_pipe_mg_imc_spectra_10g_refined_114x282.npz'),
        help='NPZ file with fiducial group spectra.',
    )
    parser.add_argument(
        '--output-base', type=str, default=None,
        help='Prefix for output PDF filenames. Default: auto from fiducial filename.',
    )
    parser.add_argument(
        '--spectrum-times', type=str, default='10,20,50',
        help='Comma-separated list of times (ns) at which to plot spectra.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve output base
    if args.output_base is None:
        base = os.path.basename(args.fiducial)
        # Strip '_fiducial_' and extension to get a sensible prefix
        base = base.replace('_fiducial_', '_').replace('.npz', '')
        args.output_base = os.path.join(_project_root, base)

    target_times = [float(x) for x in args.spectrum_times.split(',')]

    # ── Fiducial history ────────────────────────────────────────────────────
    print(f'\nLoading fiducial data: {args.fiducial}')
    times, fid_labels, T_mat, T_rad = load_fiducial(args.fiducial)
    print(f'  {len(fid_labels)} points, {len(times)} time steps, '
          f't ∈ [{times[0]:.3g}, {times[-1]:.3g}] ns')

    plot_fiducial_history(times, fid_labels, T_mat, T_rad, args.output_base)

    # ── Spectra ─────────────────────────────────────────────────────────────
    if os.path.exists(args.spectra):
        print(f'\nLoading spectra: {args.spectra}')
        out_times, spec_labels, spectra, energy_edges, energy_centers = \
            load_spectra(args.spectra)
        print(f'  snapshots at t = {out_times} ns')

        # Check that all target times are present
        for t in target_times:
            idx = int(np.argmin(np.abs(out_times - t)))
            print(f'  target t={t} ns  →  closest snapshot t={out_times[idx]:.2f} ns')

        plot_spectra(
            out_times, spec_labels, spectra, energy_edges, energy_centers,
            fid_times=times,
            fid_T_mat=T_mat,
            fid_labels=fid_labels,
            outbase=args.output_base,
            target_times=target_times,
        )
    else:
        print(f'\nSpectra file not found: {args.spectra}  (skipping spectrum plots)')

    print('\nDone.')


if __name__ == '__main__':
    main()
