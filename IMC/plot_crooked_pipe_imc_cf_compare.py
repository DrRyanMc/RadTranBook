#!/usr/bin/env python3
"""
Compare fiducial-point temperature histories between Carter-Forest IMC and
regular IMC solutions for the crooked pipe problem.

Usage
-----
# Auto-detect a Carter-Forest run and a regular IMC run:
python plot_crooked_pipe_imc_cf_compare.py

# Explicit files:
python plot_crooked_pipe_imc_cf_compare.py \
    crooked_pipe_imc_solution_cf_refined_Nb1000000_168x354.npz \
    crooked_pipe_imc_solution_refined_Nb1000000_168x354.npz
"""

import sys
import os
import glob
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utils.plotfuncs import show, font

# ---------------------------------------------------------------------------
# Matplotlib style (matching converging Marshak wave plots)
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    # Typography
    "font.family": "sans-serif",
    "font.sans-serif": ["Univers LT Std", "TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 12,
    "font.variant": "small-caps",
    "axes.titlesize": 18,
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "it",

    # Figure
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",

    # Axes/spines
    "axes.edgecolor": "black",
    "axes.linewidth": 1.15,
    "axes.grid": False,

    # Ticks
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "out",
    "ytick.direction": "out",

    # Lines
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",

    # Legend
    "legend.frameon": False,
})

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_imc_npz(path):
    """Load IMC solution NPZ file."""
    d = np.load(path, allow_pickle=True)
    times = d['times']

    # IMC stores material T in fiducial_data and radiation T in fiducial_data_rad
    fiducial_mat = d['fiducial_data'].item()
    fiducial_rad = d['fiducial_data_rad'].item()

    # Combine into unified format: {point_label: {'T_mat': array, 'T_rad': array}}
    fiducial_data = {}
    for label in fiducial_mat.keys():
        fiducial_data[label] = {
            'T_mat': np.asarray(fiducial_mat[label]),
            'T_rad': np.asarray(fiducial_rad[label]),
        }

    return np.asarray(times), fiducial_data


def infer_run_label(path):
    """Infer human-readable method label from filename."""
    base = os.path.basename(path).lower()
    if '_cf_' in base or 'carter' in base:
        return 'Carter-Forest'
    if '_fc_' in base:
        return 'Fleck-Cummings IMC'
    return 'Fleck-Cummings'


def particle_count(path):
    """Extract NbXXXXX particle count for ranking candidates."""
    match = re.search(r'nb(\d+)', os.path.basename(path).lower())
    return int(match.group(1)) if match else -1


def pick_best(candidates):
    """Pick candidate with highest particle count, then lexicographically."""
    if not candidates:
        return None
    return sorted(candidates, key=lambda p: (particle_count(p), p))[-1]


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_POINT_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
_MARKERS = ['o', 's', '^', 'd', 'v']


def plot_comparison(datasets, outbase):
    """
    Create single combined plots with all fiducial points on one axis.
    Solid lines for first dataset, dashed for second dataset.

    datasets : list of (label_str, times_arr, fiducial_data_dict)
    """
    first_points = set(datasets[0][2].keys())
    second_points = set(datasets[1][2].keys())
    point_labels = sorted(first_points & second_points)

    if not point_labels:
        raise RuntimeError('No shared fiducial point labels between the two IMC files.')

    if first_points != second_points:
        print('Warning: fiducial labels differ between files; plotting shared labels only.')
        print(f'  Shared labels: {len(point_labels)}')

    for quantity, ylabel, suffix in [
        ('T_mat', 'temperature (keV)', 'material'),
        ('T_rad', r'radiation temperature (keV)', 'radiation'),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.5))

        for ds_idx, (run_label, times, fiducial_data) in enumerate(datasets):
            linestyle = '-' if ds_idx == 0 else '--'

            for pt_idx, pt_label in enumerate(point_labels):
                T = fiducial_data[pt_label][quantity]
                color = _POINT_COLORS[pt_idx % len(_POINT_COLORS)]
                marker = _MARKERS[pt_idx % len(_MARKERS)]

                ax.loglog(
                    times,
                    T,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    marker=marker,
                    markersize=4,
                    markevery=max(1, len(times) // 20),
                    alpha=0.85,
                )

        # Add invisible lines for method legend (CF vs regular)
        for ds_idx, (run_label, _, _) in enumerate(datasets):
            linestyle = '-' if ds_idx == 0 else '--'
            ax.plot([], [], color='black', linestyle=linestyle, linewidth=2.5, label=run_label)

        ax.set_xlabel('time (ns)')
        ax.set_ylabel(ylabel)

        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)

        ax.legend(fontsize=9, loc='best', ncol=1)

        plt.tight_layout()
        outname = f'{outbase}_{suffix}.pdf'
        plt.savefig(outname, dpi=600)
        print(f'Saved: {outname}')
        plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare Carter-Forest and regular IMC crooked-pipe fiducial histories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='Two IMC NPZ files to compare (Carter-Forest and regular). If omitted, auto-detects.',
    )
    parser.add_argument(
        '--output-base',
        type=str,
        default='crooked_pipe_imc_cf_compare',
        help='Prefix for output PDF filenames.',
    )
    return parser.parse_args()


def auto_detect_files():
    """Auto-detect one Carter-Forest IMC file and one regular IMC file."""
    cf_candidates = sorted(glob.glob('crooked_pipe_imc_solution_cf_*.npz'))

    regular_all = sorted(glob.glob('crooked_pipe_imc_solution_*.npz'))
    regular_candidates = [
        p for p in regular_all
        if '_cf_' not in p.lower() and '_fc_' not in p.lower()
    ]

    cf_file = pick_best(cf_candidates)
    regular_file = pick_best(regular_candidates)

    if not cf_file:
        print('Error: No Carter-Forest IMC files found (crooked_pipe_imc_solution_cf_*.npz)')
        sys.exit(1)
    if not regular_file:
        print('Error: No regular IMC files found (crooked_pipe_imc_solution_*.npz without _cf_/_fc_)')
        sys.exit(1)

    return [cf_file, regular_file]


def main():
    args = parse_args()

    if args.files:
        if len(args.files) != 2:
            print(f'Error: Need exactly 2 files (Carter-Forest and regular IMC), got {len(args.files)}')
            sys.exit(1)
        npz_paths = args.files
    else:
        npz_paths = auto_detect_files()

    print('Comparing:')
    print(f'  File 1: {npz_paths[0]}')
    print(f'  File 2: {npz_paths[1]}')

    datasets = []
    for path in npz_paths:
        times, fiducial_data = load_imc_npz(path)
        label = infer_run_label(path)
        print(f'  [{label}] {len(times)} time points')
        datasets.append((label, times, fiducial_data))

    print('\nGenerating plots...')
    plot_comparison(datasets, outbase=args.output_base)

    print('\nDone.')


if __name__ == '__main__':
    main()
