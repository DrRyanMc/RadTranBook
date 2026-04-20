#!/usr/bin/env python3
"""
Compare fiducial-point temperature histories between IMC and diffusion 
solutions for the crooked pipe problem.

Usage
-----
# Auto-detect the refined mesh solutions:
python plot_crooked_pipe_imc_diffusion_compare.py

# Explicit files:
python plot_crooked_pipe_imc_diffusion_compare.py \
    crooked_pipe_imc_solution_refined_Nb100000_168x354.npz \
    crooked_pipe_noneq_solution_refined_larsen_114x282.npz
"""

import sys
import os
import glob
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
            'T_rad': np.asarray(fiducial_rad[label])
        }
    
    return times, fiducial_data


def load_diffusion_npz(path):
    """Load diffusion solution NPZ file."""
    d = np.load(path, allow_pickle=True)
    times = d['times']
    
    # Diffusion already has format: {point_label: {'T_mat': array, 'T_rad': array}}
    fiducial_data = d['fiducial_data'].item()
    
    # Convert arrays to numpy if needed
    for label in fiducial_data.keys():
        fiducial_data[label]['T_mat'] = np.asarray(fiducial_data[label]['T_mat'])
        fiducial_data[label]['T_rad'] = np.asarray(fiducial_data[label]['T_rad'])
    
    return times, fiducial_data


def detect_file_type(path):
    """Determine if file is IMC or diffusion based on filename."""
    basename = os.path.basename(path).lower()
    if 'imc' in basename:
        return 'IMC'
    elif 'noneq' in basename:
        return 'Diffusion'
    else:
        # Try to detect by loading and checking keys
        d = np.load(path, allow_pickle=True)
        if 'fiducial_data_rad' in d.keys():
            return 'IMC'
        else:
            return 'Diffusion'


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_POINT_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
_MARKERS = ['o', 's', '^', 'd', 'v']


def plot_comparison(datasets, outbase):
    """
    Create single combined plots with all fiducial points on one axis.
    Solid lines for first dataset (IMC), dashed for second (diffusion).
    
    datasets : list of (label_str, times_arr, fiducial_data_dict)
    """
    # Collect point labels from first dataset
    point_labels = list(datasets[0][2].keys())
    
    # Create two figures: material and radiation temperature
    for quantity, ylabel, suffix in [
        ('T_mat', 'temperature (keV)', 'material'),
        ('T_rad', r'radiation temperature (keV)', 'radiation')
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        
        # Plot each dataset with different linestyle
        for ds_idx, (run_label, times, fiducial_data) in enumerate(datasets):
            linestyle = '-' if ds_idx == 0 else '--'
            
            # Plot each fiducial point
            for pt_idx, pt_label in enumerate(point_labels):
                T = fiducial_data[pt_label][quantity]
                color = _POINT_COLORS[pt_idx % len(_POINT_COLORS)]
                marker = _MARKERS[pt_idx % len(_MARKERS)]
                
                # Label format: only show point location on first dataset
                if ds_idx == 0:
                    label = pt_label.split(':', 1)[1].strip() if ':' in pt_label else pt_label
                else:
                    label = None
                
                ax.loglog(
                    times, T,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    marker=marker,
                    markersize=4,
                    markevery=max(1, len(times) // 20),
                    alpha=0.85,
                    #label=label,
                )
        
        # Add invisible lines for the linestyle legend (IMC vs diffusion)
        for ds_idx, (run_label, _, _) in enumerate(datasets):
            linestyle = '-' if ds_idx == 0 else '--'
            ax.plot([], [], color='black', linestyle=linestyle, linewidth=2.5,
                    label=run_label)
        
        ax.set_xlabel('time (ns)')
        ax.set_ylabel(ylabel)
        
        # Match converging Marshak wave style: remove top/right spines, thicken bottom/left
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
        # No legend frame (handled by rcParams)
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
        description='Compare IMC and diffusion crooked-pipe fiducial histories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'files', nargs='*',
        help='Two NPZ files to compare (IMC and diffusion). If omitted, auto-detects.',
    )
    parser.add_argument(
        '--output-base', type=str, default='crooked_pipe_imc_diffusion_compare',
        help='Prefix for output PDF filenames.',
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    if args.files:
        if len(args.files) != 2:
            print(f'Error: Need exactly 2 files (IMC and diffusion), got {len(args.files)}')
            sys.exit(1)
        npz_paths = args.files
    else:
        # Auto-detect: find one IMC refined and one diffusion refined
        imc_candidates = sorted(glob.glob('crooked_pipe_imc_solution_refined_*.npz'))
        diff_candidates = sorted(glob.glob('crooked_pipe_noneq_solution_refined_*.npz'))
        
        if not imc_candidates:
            print('Error: No IMC solution files found (crooked_pipe_imc_solution_refined_*.npz)')
            sys.exit(1)
        if not diff_candidates:
            print('Error: No diffusion solution files found (crooked_pipe_noneq_solution_refined_*.npz)')
            sys.exit(1)
        
        # Pick the ones with highest particle count or specific names
        imc_file = imc_candidates[-1]  # Last alphabetically (likely highest Nb)
        diff_file = [f for f in diff_candidates if 'larsen' in f.lower()]
        if not diff_file:
            diff_file = diff_candidates[-1]
        else:
            diff_file = diff_file[0]
        
        npz_paths = [imc_file, diff_file]
    
    print('Comparing:')
    print(f'  File 1: {npz_paths[0]}')
    print(f'  File 2: {npz_paths[1]}')
    
    # Load files
    datasets = []
    for path in npz_paths:
        file_type = detect_file_type(path)
        
        if file_type == 'IMC':
            times, fiducial_data = load_imc_npz(path)
            label = 'IMC'
        else:
            times, fiducial_data = load_diffusion_npz(path)
            label = 'Diffusion'
        
        print(f'  [{label}] {file_type} - {len(times)} time points')
        datasets.append((label, times, fiducial_data))
    
    # Create comparison plots
    print('\nGenerating plots...')
    plot_comparison(datasets, outbase=args.output_base)
    
    print('\nDone.')


if __name__ == '__main__':
    main()
