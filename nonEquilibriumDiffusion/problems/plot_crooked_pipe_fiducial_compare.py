#!/usr/bin/env python3
"""
Compare fiducial-point temperature histories from two crooked-pipe NPZ files
(typically the Larsen and Levermore-Pomraning flux-limiter runs).

Usage
-----
# Auto-detect the two refined NPZ files:
python plot_crooked_pipe_fiducial_compare.py

# Explicit files:
python plot_crooked_pipe_fiducial_compare.py \
    crooked_pipe_noneq_solution_refined_larsen_114x282.npz \
    crooked_pipe_noneq_solution_refined_levermore_pomraning_114x282.npz
"""

import sys
import os
import glob
import argparse

import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.plotfuncs import show, font

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def short_label(path):
    """Derive a short label from the filename, e.g. 'larsen' or 'L-P'."""
    name = os.path.splitext(os.path.basename(path))[0].lower()
    if 'levermore' in name or 'pomraning' in name:
        return 'Levermore-Pomraning'
    if 'larsen' in name:
        return 'Larsen'
    if 'max' in name:
        return 'Max'
    if 'none' in name:
        return 'None'
    return os.path.splitext(os.path.basename(path))[0]


def load_npz(path):
    d = np.load(path, allow_pickle=True)
    times = d['times']
    fiducial_data = d['fiducial_data'].item()
    return times, fiducial_data


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

# Solid lines for first file, dashed for second; each point gets its own colour
_POINT_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
_LINESTYLES   = ['-', '--', '-.', ':']
_MARKERS      = ['o', 's', '^', 'd', 'v']


def _plot_panel(ax, datasets, quantity, point_labels):
    """
    Draw one panel (material or radiation temperature).

    datasets : list[ (label_str, times_arr, fiducial_data_dict) ]
    quantity : 'T_mat' or 'T_rad'
    """
    for ds_idx, (run_label, times, fiducial_data) in enumerate(datasets):
        ls = _LINESTYLES[ds_idx % len(_LINESTYLES)]
        for pt_idx, pt_label in enumerate(point_labels):
            data = fiducial_data[pt_label]
            T = np.asarray(data[quantity])
            color = _POINT_COLORS[pt_idx % len(_POINT_COLORS)]
            marker = _MARKERS[pt_idx % len(_MARKERS)]

            # Only add a legend entry on the first dataset per point,
            # and one legend entry for the linestyle per dataset
            label = None
            if ds_idx == 0:
                label = pt_label.split(':')[1].strip()  # e.g. 'r=0.0, z=0.25'

            ax.loglog(
                times, T,
                color=color,
                linestyle=ls,
                linewidth=1.8,
                marker=marker,
                markersize=4,
                markevery=max(1, len(times) // 15),
                alpha=0.85,
                #label=label,
            )

    # Add invisible lines for the linestyle legend entries
    for ds_idx, (run_label, _, _) in enumerate(datasets):
        ls = _LINESTYLES[ds_idx % len(_LINESTYLES)]
        ax.plot([], [], color='black', linestyle=ls, linewidth=2,
                label=run_label)


def plot_comparison(datasets, outbase):
    """
    Two figures: material temperature and radiation temperature.
    Each figure has one panel per fiducial point and a shared legend.
    """
    # Collect the common set of point labels (from first dataset)
    point_labels = list(datasets[0][2].keys())

    y_labels = {
        'T_mat': 'T (keV)',
        'T_rad': r'$T_\mathrm{r}$ (keV)',
    }
    suffixes = {
        'T_mat': 'material',
        'T_rad': 'radiation',
    }

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
                ls = _LINESTYLES[ds_idx % len(_LINESTYLES)]
                T = np.asarray(fiducial_data[pt_label][quantity])
                color = 'tab:blue'  # single point per panel → one colour per run

                ax.loglog(
                    times, T,
                    color=_POINT_COLORS[ds_idx % len(_POINT_COLORS)],
                    linestyle='-',
                    linewidth=2,
                    marker=_MARKERS[ds_idx % len(_MARKERS)],
                    markersize=4,
                    markevery=max(1, len(times) // 15),
                    alpha=0.85,
                    label=run_label,
                )

            # Panel title: just the location part
            loc_str = pt_label.split(':', 1)[1].strip() if ':' in pt_label else pt_label
            #ax.set_title(loc_str, fontsize=10)
            ax.set_xlabel('time (ns)', fontsize=11)
            if ax is axes[0]:
                ax.set_ylabel(ylabel, fontsize=11)
            ax.grid(True, which='both', alpha=0.25, linestyle='--')
            ax.grid(True, which='minor', alpha=0.12, linestyle=':')
            ax.legend(prop=font, facecolor='white', edgecolor='none',framealpha=1.0, fontsize=9)

        plt.tight_layout()
        outname = f'{outbase}_{suffixes[quantity]}.pdf'
        show(outname, close_after=True)
        print(f'Saved: {outname}')

    # --------------------------------------------------------------------------
    # Combined figure: all five points on one set of axes, both runs overlaid
    # --------------------------------------------------------------------------
    for quantity, ylabel in y_labels.items():
        fig, ax = plt.subplots(figsize=(9, 5.5))

        _plot_panel(ax, datasets, quantity, point_labels)

        ax.set_xlabel('time (ns)', fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        ax.grid(True, which='both', alpha=0.25, linestyle='--')
        ax.grid(True, which='minor', alpha=0.12, linestyle=':')
        ax.legend(prop=font, facecolor='white', edgecolor='none', fontsize=9,
                  loc='best', ncol=1, framealpha=1.0)

        plt.tight_layout()
        outname = f'{outbase}_{suffixes[quantity]}_combined.pdf'
        show(outname, close_after=True)
        print(f'Saved: {outname}')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare crooked-pipe fiducial histories across two runs.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'files', nargs='*',
        help='Two NPZ files to compare. If omitted, auto-detects '
             'crooked_pipe_noneq_solution_refined_*npz files.',
    )
    parser.add_argument(
        '--output-base', type=str, default='crooked_pipe_fiducial_compare',
        help='Prefix for output PDF filenames.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.files:
        npz_paths = args.files
    else:
        candidates = sorted(glob.glob('crooked_pipe_noneq_solution_refined_*x*.npz'))
        # Prefer the ones with limiter names in them
        named = [c for c in candidates if 'larsen' in c or 'levermore' in c]
        npz_paths = named if named else candidates

    if len(npz_paths) < 2:
        print(f'Need at least 2 NPZ files; found: {npz_paths}')
        sys.exit(1)

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
