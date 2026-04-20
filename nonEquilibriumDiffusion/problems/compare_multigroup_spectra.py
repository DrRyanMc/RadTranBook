#!/usr/bin/env python3
"""
Compare multigroup radiation spectra from runs with different numbers of groups.

Plots E_{r,g} (and optionally phi_g) vs group-centre photon energy at a chosen
time and spatial position, overlaying all supplied NPZ files.

Usage examples
--------------
# Auto-detect *_solutions.npz files; default time and position
python compare_multigroup_spectra.py

# Compare specific files at t=5 ns, r=1.0 cm
python compare_multigroup_spectra.py \
    marshak_wave_multigroup_powerlaw_2g_no_precond_timeBC_larsen_solutions.npz \
    marshak_wave_multigroup_powerlaw_10g_no_precond_timeBC_larsen_solutions.npz \
    marshak_wave_multigroup_powerlaw_50g_no_precond_timeBC_larsen_solutions.npz \
    --time 5.0 --r-position 1.0

# Also plot phi_g
python compare_multigroup_spectra.py --plot-phi ...
"""

import re
import sys
import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup — run from problems/ or from workspace root
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.plotfuncs import show, font

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def group_centers(energy_edges):
    """Geometric-mean centre energy for each group (keV)."""
    return np.sqrt(energy_edges[:-1] * energy_edges[1:])


def group_widths(energy_edges):
    """Width of each energy group (keV)."""
    return energy_edges[1:] - energy_edges[:-1]


def load_npz(path):
    """Load a structured NPZ file; raise ValueError for old list-of-dicts format."""
    d = np.load(path, allow_pickle=False)
    required = ['times', 'r', 'energy_edges', 'E_r_groups', 'phi_groups']
    missing = [k for k in required if k not in d]
    if missing or 'solutions' in d:
        raise ValueError(
            f"'{os.path.basename(path)}' uses the old list-of-dicts format.\n"
            "  Re-run the simulation to produce a structured NPZ with named arrays."
        )
    return d


def nearest_index(arr, value):
    return int(np.argmin(np.abs(arr - value)))


def label_from_filename(path):
    """Extract a readable label, e.g. '10g larsen' from the filename."""
    name = os.path.splitext(os.path.basename(path))[0]
    # Try to extract n_groups and flux-limiter from standard naming convention
    import re
    m_g = re.search(r'_(\d+)g_', name)
    m_fl = None #re.search(r'_(larsen|levermore_pomraning|max|none)(?:_|$)', name)
    label = m_g.group(1) + ' groups' if m_g else name
    if m_fl:
        label += f' ({m_fl.group(1)})'
    return label


# ---------------------------------------------------------------------------
# Main comparison function
# ---------------------------------------------------------------------------

def compare_spectra(npz_paths, target_time, target_r, plot_phi=False, outbase=None):
    """
    Load each NPZ file and plot E_{r,g} vs photon energy on a shared axis.

    Parameters
    ----------
    npz_paths : list[str]
    target_time : float or None  — ns; if None uses last snapshot in each file
    target_r : float or None     — cm; if None uses first cell
    plot_phi : bool              — if True also produce a phi_g panel
    outbase : str or None        — prefix for output PDF filenames
    """
    datasets = []
    for path in sorted(npz_paths):
        try:
            d = load_npz(path)
        except ValueError as exc:
            print(f"  SKIPPING {os.path.basename(path)}: {exc}")
            continue
        datasets.append((path, d))

    if not datasets:
        print("No usable NPZ files found.")
        return
    #sort dataset by number of groups
    datasets.sort(key=lambda x: int(re.search(r'_(\d+)g_', 
                                              os.path.basename(x[0])).group(1)) if re.search(r'_(\d+)g_', os.path.basename(x[0])) else 0)
    # Determine actual time / position from first dataset if not specified
    first_d = datasets[0][1]
    t_query = target_time if target_time is not None else float(first_d['times'][-1])
    r_query = target_r    if target_r    is not None else float(first_d['r'][0])

    print(f"\nComparing {len(datasets)} dataset(s) at t ≈ {t_query:.3f} ns, "
          f"r ≈ {r_query:.3f} cm\n")

    # ---------- colour cycle -----------
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    n_panels = 2 if plot_phi else 1
    fig, axes = plt.subplots(1, n_panels,
                             figsize=(6.5 * n_panels, 4.8),
                             squeeze=False)
    ax_Er  = axes[0, 0]
    ax_phi = axes[0, 1] if plot_phi else None

    used_times = []
    used_rs    = []
    markers = ['o', 's', 'D', '^', 'v', 'P', '*', 'X']
    linestyles = ['-', '--', ':', '-.']
    for idx, (path, d) in enumerate(datasets):
        times        = d['times']
        r            = d['r']
        energy_edges = d['energy_edges']
        E_r_groups   = d['E_r_groups']   # (n_saved, n_groups, n_cells)
        phi_groups   = d['phi_groups']    # (n_saved, n_groups, n_cells)

        t_idx = nearest_index(times,  t_query)
        r_idx = nearest_index(r,      r_query)

        t_actual = float(times[t_idx])
        r_actual = float(r[r_idx])
        used_times.append(t_actual)
        used_rs.append(r_actual)

        E_centers = group_centers(energy_edges)
        dE        = group_widths(energy_edges)
        n_groups  = len(E_centers)

        spectrum_Er  = E_r_groups[t_idx, :, r_idx]/dE
        spectrum_phi = phi_groups[t_idx, :, r_idx]

        label = label_from_filename(path)
        color = colors[idx % len(colors)]

        # Step plot for a histogram-like spectral display
        ax_Er.step(energy_edges, np.append(spectrum_Er, spectrum_Er[-1]),
                   where='post', color=color, linestyle=linestyles[idx % len(linestyles)], linewidth=2, label=label)
        ax_Er.plot(E_centers, spectrum_Er,
                   markers[0], color=color, markersize=4, alpha=0.6)

        if ax_phi is not None:
            ax_phi.step(energy_edges, np.append(spectrum_phi, spectrum_phi[-1]),
                        where='post', color=color, linestyle=linestyles[idx % len(linestyles)], linewidth=2, label=label)
            ax_phi.plot(E_centers, spectrum_phi,
                        markers[idx % len(markers)], color=color, markersize=4, alpha=0.6)

    #add the markers to the legend
    handles, labels = ax_Er.get_legend_handles_labels()
    new_handles = []
    for h in handles:
       if isinstance(h, plt.Line2D):
           new_handles.append(h)
    ax_Er.legend(new_handles, labels, prop=font, facecolor='white', edgecolor='none', fontsize=10)
    # Annotate actual time/position used (from first dataset)
    t_str = f't = {used_times[0]:.3f} ns' if used_times else ''
    r_str = f'r = {used_rs[0]:.3f} cm'   if used_rs    else ''

    # --- format E_r panel ---
    ax_Er.set_xlabel('photon energy (keV)', fontsize=12)
    ax_Er.set_ylabel(r'$E_{r,g}$ (GJ$/$cm$^{3}/$keV)', fontsize=12)
    #ax_Er.set_title(f'{t_str},  {r_str}', fontsize=11)
    ax_Er.set_xscale('log')
    ax_Er.set_yscale('log')
    #ax_Er.legend(prop=font, facecolor='white', edgecolor='none', fontsize=10)
    ax_Er.grid(True, which='both', alpha=0.3, linestyle='--')

    if ax_phi is not None:
        ax_phi.set_xlabel('photon energy (keV)', fontsize=12)
        ax_phi.set_ylabel(r'$\phi_g$ (GJ$\,$cm$^{-2}$$\,$ns$^{-1}$)', fontsize=12)
        ax_phi.set_title(f'{t_str},  {r_str}', fontsize=11)
        ax_phi.set_xscale('log')
        ax_phi.set_yscale('log')
        ax_phi.legend(prop=font, facecolor='white', edgecolor='none', fontsize=10)
        ax_phi.grid(True, which='both', alpha=0.3, linestyle='--')

    plt.tight_layout()

    # Derive output filename
    if outbase is None:
        outbase = 'multigroup_spectra_comparison'
    t_tag = f'_t{t_query:.2f}ns'.replace('.', 'p')
    r_tag = f'_r{r_query:.2f}cm'.replace('.', 'p')
    suffix = '_Er_phi' if plot_phi else '_Er'
    outname = f'{outbase}{t_tag}{r_tag}{suffix}.pdf'

    show(outname, close_after=True)
    print(f"Saved: {outname}")


# ---------------------------------------------------------------------------
# Spatial-sweep version: spectrum at multiple r positions for one file
# ---------------------------------------------------------------------------

def compare_spectra_across_space(npz_paths, target_time, r_positions, outbase=None):
    """
    For each NPZ file, plot the spectrum at several r positions on one figure,
    with one panel per file.
    """
    datasets = []
    for path in sorted(npz_paths):
        try:
            d = load_npz(path)
        except ValueError as exc:
            print(f"  SKIPPING {os.path.basename(path)}: {exc}")
            continue
        datasets.append((path, d))

    if not datasets:
        print("No usable NPZ files found.")
        return

    first_d = datasets[0][1]
    t_query = target_time if target_time is not None else float(first_d['times'][-1])

    n_files = len(datasets)
    fig, axes = plt.subplots(1, n_files, figsize=(6.0 * n_files, 4.8),
                             squeeze=False, sharey=True)

    pos_colors = plt.cm.viridis(np.linspace(0, 1, len(r_positions)))

    for col, (path, d) in enumerate(datasets):
        ax = axes[0, col]
        times        = d['times']
        r            = d['r']
        energy_edges = d['energy_edges']
        E_r_groups   = d['E_r_groups']

        t_idx = nearest_index(times, t_query)
        t_actual = float(times[t_idx])
        E_centers = group_centers(energy_edges)

        for pi, r_val in enumerate(r_positions):
            r_idx  = nearest_index(r, r_val)
            r_actual = float(r[r_idx])
            spectrum = E_r_groups[t_idx, :, r_idx]

            ax.step(energy_edges, np.append(spectrum, spectrum[-1]),
                    where='post', color=pos_colors[pi], linewidth=2,
                    label=f'r = {r_actual:.2f} cm')
            ax.plot(E_centers, spectrum,
                    'o', color=pos_colors[pi], markersize=4, alpha=0.6)

        label = label_from_filename(path)
        ax.set_xlabel('photon energy (keV)', fontsize=12)
        if col == 0:
            ax.set_ylabel(r'$E_{r,g}$ (GJ$\,$cm$^{-3}$)', fontsize=12)
        #ax.set_title(f'{label}\nt = {t_actual:.3f} ns', fontsize=10)
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.legend(prop=font, facecolor='white', edgecolor='none', fontsize=8)
        ax.grid(True, which='both', alpha=0.3, linestyle='--')

    plt.tight_layout()

    if outbase is None:
        outbase = 'multigroup_spectra_spatial'
    t_tag = f'_t{t_query:.2f}ns'.replace('.', 'p')
    outname = f'{outbase}{t_tag}_spatial_sweep.pdf'
    show(outname, close_after=True)
    print(f"Saved: {outname}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare multigroup radiation spectra across NPZ files.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        'files', nargs='*',
        help='NPZ solution files to compare. If omitted, auto-detects '
             '*_solutions.npz in the current directory.'
    )
    parser.add_argument(
        '--time', '-t', type=float, default=None,
        help='Target time (ns). Nearest snapshot is used. Default: last snapshot.'
    )
    parser.add_argument(
        '--r-position', '-r', type=float, default=None,
        help='Target spatial position r (cm). Default: r=0.0 cm.'
    )
    parser.add_argument(
        '--plot-phi', action='store_true',
        help='Also plot scalar intensity phi_g alongside E_r_g.'
    )
    parser.add_argument(
        '--spatial-sweep', action='store_true',
        help='Instead of a single r, plot spectra at multiple r positions '
             '(one panel per file). --r-sweep-positions controls the set.'
    )
    parser.add_argument(
        '--r-sweep-positions', type=str, default=None,
        help='Comma-separated r values (cm) for --spatial-sweep, '
             'e.g. "0.5,1.5,3.0,5.0". Default: 5 evenly spaced points.'
    )
    parser.add_argument(
        '--output-base', type=str, default=None,
        help='Prefix for output PDF filenames.'
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve NPZ files
    if args.files:
        npz_paths = args.files
    else:
        candidates = sorted(glob.glob('*_solutions.npz'))
        if not candidates:
            candidates = sorted(glob.glob('marshak_wave_multigroup_powerlaw_*.npz'))
        if not candidates:
            print("No NPZ files found. Specify files explicitly or run from the "
                  "directory containing *_solutions.npz files.")
            sys.exit(1)
        npz_paths = candidates
        print(f"Auto-detected {len(npz_paths)} file(s):")
        for p in npz_paths:
            print(f"  {p}")

    if args.spatial_sweep:
        # Parse r positions
        if args.r_sweep_positions:
            r_positions = [float(x.strip()) for x in args.r_sweep_positions.split(',')]
        else:
            # Default: 5 evenly spaced positions inferred from first loadable file
            for p in npz_paths:
                try:
                    d = load_npz(p)
                    r_arr = d['r']
                    r_positions = list(np.linspace(r_arr[0], r_arr[-1], 5))
                    break
                except ValueError:
                    continue
            else:
                r_positions = [0.5, 1.5, 3.0, 5.0, 6.5]

        compare_spectra_across_space(
            npz_paths, args.time, r_positions, outbase=args.output_base
        )
    else:
        compare_spectra(
            npz_paths,
            target_time=args.time,
            target_r=args.r_position,
            plot_phi=args.plot_phi,
            outbase=args.output_base
        )


if __name__ == '__main__':
    main()
