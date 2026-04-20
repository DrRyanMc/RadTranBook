#!/usr/bin/env python3
"""
Compare fiducial-point temperature histories from multigroup crooked-pipe NPZ
files (e.g. 2-group, 10-group, 50-group solutions).

Usage
-----
# Auto-detect refined multigroup solution files in the workspace root:
python plot_crooked_pipe_multigroup_compare.py

# Explicit files (any order – sorted by group count automatically):
python plot_crooked_pipe_multigroup_compare.py \
    crooked_pipe_2g_noneq_solution_refined_168x354.npz \
    crooked_pipe_10g_noneq_solution_refined_168x354.npz \
    crooked_pipe_50g_noneq_solution_refined_168x354.npz
"""

import sys
import os
import re
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup  (script lives in nonEquilibriumDiffusion/problems/)
# ---------------------------------------------------------------------------
_script_dir  = os.path.dirname(os.path.abspath(__file__))
_noneq_dir   = os.path.dirname(_script_dir)
_project_root = os.path.dirname(_noneq_dir)
for _p in (_noneq_dir, _project_root):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from utils.plotfuncs import show, font

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _group_count(path):
    """Return the integer group count encoded in the filename, or None."""
    m = re.search(r'_(\d+)g_', os.path.basename(path))
    return int(m.group(1)) if m else None


def short_label(path):
    """Return a human-readable label like '2 groups' or '10 groups'."""
    n = _group_count(path)
    if n is not None:
        return f'{n} groups'
    return os.path.splitext(os.path.basename(path))[0]


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    times = d['times']
    fiducial_data = d['fiducial_data'].item()
    return times, fiducial_data


# ---------------------------------------------------------------------------
# Plotting constants
# ---------------------------------------------------------------------------

_RUN_COLORS  = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
_LINESTYLES  = ['-', '--', '-.', ':']
_MARKERS     = ['o', 's', '^', 'd', 'v']
_POINT_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']


def _plot_combined_panel(ax, datasets, quantity, point_labels):
    """
    All fiducial points on one axis.
    Points → colours, runs → linestyles.
    """
    for ds_idx, (run_label, times, fiducial_data) in enumerate(datasets):
        ls = _LINESTYLES[ds_idx % len(_LINESTYLES)]
        for pt_idx, pt_label in enumerate(point_labels):
            T = np.asarray(fiducial_data[pt_label][quantity])
            color  = _POINT_COLORS[pt_idx % len(_POINT_COLORS)]
            marker = _MARKERS[pt_idx % len(_MARKERS)]
            ax.loglog(
                times, T,
                color=color,
                linestyle=ls,
                linewidth=1.8,
                marker=marker,
                markersize=4,
                markevery=max(1, len(times) // 15),
                alpha=0.85,
            )
    #set x lower limit to be 0.01 ns (10 ps) to avoid cluttering the plot with very early time points
    if (np.min(T)> 5e-3):
        ax.set_xlim(left=0.05)
    # Invisible lines for the linestyle/run legend
    for ds_idx, (run_label, _, _) in enumerate(datasets):
        ls = _LINESTYLES[ds_idx % len(_LINESTYLES)]
        ax.plot([], [], color='black', linestyle=ls, linewidth=2, label=run_label)


def plot_comparison(datasets, outbase):
    """
    Produces four PDF files:
      <outbase>_material.pdf        — one panel per fiducial point
      <outbase>_radiation.pdf
      <outbase>_material_combined.pdf   — all points on one axis
      <outbase>_radiation_combined.pdf
    """
    point_labels = list(datasets[0][2].keys())

    y_labels = {
        'T_mat': r'$T$ (keV)',
        'T_rad': r'$T_\mathrm{r}$ (keV)',
    }
    suffixes = {
        'T_mat': 'material',
        'T_rad': 'radiation',
    }

    # ------------------------------------------------------------------
    # Per-point panels  (each panel = one fiducial location, runs overlay)
    # ------------------------------------------------------------------
    for quantity, ylabel in y_labels.items():
        fig, axes = plt.subplots(
            1, len(point_labels),
            figsize=(3.5 * len(point_labels), 4.5),
            sharey=True,
        )
        if len(point_labels) == 1:
            axes = [axes]

        for ax, pt_label in zip(axes, point_labels):
            for ds_idx, (run_label, times, fiducial_data) in enumerate(datasets):
                T = np.asarray(fiducial_data[pt_label][quantity])
                ax.loglog(
                    times, T,
                    color=_RUN_COLORS[ds_idx % len(_RUN_COLORS)],
                    linestyle='-',
                    linewidth=2,
                    marker=_MARKERS[ds_idx % len(_MARKERS)],
                    markersize=4,
                    markevery=max(1, len(times) // 15),
                    alpha=0.85,
                    label=run_label,
                )

            loc_str = pt_label.split(':', 1)[1].strip() if ':' in pt_label else pt_label
            ax.set_xlabel('time (ns)', fontsize=11)
            if ax is axes[0]:
                ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, which='both', alpha=0.25, linestyle='--')
            ax.grid(True, which='minor', alpha=0.12, linestyle=':')
            leg = ax.legend(prop=font, facecolor='white', edgecolor='none',
                            fontsize=9)
            leg.get_frame().set_alpha(None)

        plt.tight_layout()
        outname = f'{outbase}_{suffixes[quantity]}.pdf'
        show(outname, close_after=True)
        print(f'Saved: {outname}')

    # ------------------------------------------------------------------
    # Combined figures  (all points, all runs)
    # ------------------------------------------------------------------
    for quantity, ylabel in y_labels.items():
        fig, ax = plt.subplots(figsize=(9, 5.5))

        _plot_combined_panel(ax, datasets, quantity, point_labels)

        ax.set_xlabel('time (ns)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, which='both', alpha=0.25, linestyle='--')
        ax.grid(True, which='minor', alpha=0.12, linestyle=':')
        leg = ax.legend(prop=font, facecolor='white', edgecolor='none',
                        fontsize=9, loc='best', ncol=1)
        leg.get_frame().set_alpha(None)

        plt.tight_layout()
        outname = f'{outbase}_{suffixes[quantity]}_combined.pdf'
        show(outname, close_after=True)
        print(f'Saved: {outname}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare multigroup crooked-pipe fiducial histories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'files', nargs='*',
        help='NPZ files to compare.  If omitted, auto-detects '
             'crooked_pipe_*g_noneq_solution_refined_*x*.npz files.',
    )
    parser.add_argument(
        '--output-base', type=str,
        default='crooked_pipe_multigroup_fiducial_compare',
        help='Prefix for output PDF filenames.',
    )
    # Search path for auto-detection (default: workspace root two levels up)
    parser.add_argument(
        '--search-dir', type=str, default=None,
        help='Directory to search for NPZ files (default: two levels above this script).',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.files:
        npz_paths = args.files
    else:
        search_dir = args.search_dir or _project_root
        pattern = os.path.join(search_dir, 'crooked_pipe_*g_noneq_solution_refined_*x*.npz')
        candidates = sorted(glob.glob(pattern))
        if not candidates:
            # fallback: look in cwd
            candidates = sorted(glob.glob('crooked_pipe_*g_noneq_solution_refined_*x*.npz'))
        npz_paths = candidates

    if len(npz_paths) < 2:
        print(f'Need at least 2 NPZ files; found: {npz_paths}')
        sys.exit(1)

    # Sort by group count so legend is ordered 2 → 10 → 50 etc.
    def sort_key(p):
        n = _group_count(p)
        return n if n is not None else 9999

    npz_paths = sorted(npz_paths, key=sort_key)

    print('Comparing:')
    datasets = []
    for path in npz_paths:
        label = short_label(path)
        print(f'  [{label}]  {path}')
        times, fiducial_data = load_npz(path)
        datasets.append((label, times, fiducial_data))

    plot_comparison(datasets, outbase=args.output_base)
    print('\nDone.')


if __name__ == '__main__':
    main()
