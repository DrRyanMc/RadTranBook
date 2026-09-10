#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Compare fiducial-point temperature histories between IMC, diffusion,
and discrete-ordinates (SN) solutions for the crooked pipe problem.

Usage
-----
# Auto-detect IMC, diffusion, and SN solution files:
python plot_crooked_pipe_imc_diffusion_compare.py

# Explicit files:
python plot_crooked_pipe_imc_diffusion_compare.py \
    crooked_pipe_imc_solution_refined_Nb100000_168x354.npz \
    crooked_pipe_noneq_solution_refined_larsen_114x282.npz \
    crooked_pipe_sn_114x282.npz
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
            'T_rad': np.asarray(fiducial_rad[label])
        }
    
    return np.asarray(times), fiducial_data


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
    
    return np.asarray(times), fiducial_data


def load_sn_npz(path):
    """Load discrete ordinates (SN) solution NPZ file."""
    d = np.load(path, allow_pickle=True)
    times = np.asarray(d['fid_times'])
    labels = [str(x) for x in np.asarray(d['fid_labels']).tolist()]

    fid_tmat_arr = np.asarray(d['fid_Tmat'])
    fid_trad_arr = np.asarray(d['fid_Trad'])

    if fid_tmat_arr.ndim != 2 or fid_trad_arr.ndim != 2:
        raise ValueError(f"Unexpected SN fiducial array shape in {path}")
    if fid_tmat_arr.shape[0] != len(times) or fid_trad_arr.shape[0] != len(times):
        raise ValueError(f"SN fiducial history length mismatch in {path}")
    if fid_tmat_arr.shape[1] != len(labels) or fid_trad_arr.shape[1] != len(labels):
        raise ValueError(f"SN fiducial label count mismatch in {path}")

    fiducial_data = {}
    for j, label in enumerate(labels):
        fiducial_data[label] = {
            'T_mat': fid_tmat_arr[:, j],
            'T_rad': fid_trad_arr[:, j],
        }

    return times, fiducial_data


def detect_file_type(path):
    """Determine if file is IMC, diffusion, or SN."""
    basename = os.path.basename(path).lower()
    if 'imc' in basename:
        return 'IMC'
    elif ('_sn_' in basename) or basename.startswith('sn_') or basename.startswith('crooked_pipe_sn'):
        return 'SN'
    elif 'noneq' in basename:
        return 'Diffusion'
    else:
        # Try to detect by loading and checking keys
        d = np.load(path, allow_pickle=True)
        keys = set(d.keys())
        if 'fiducial_data_rad' in keys:
            return 'IMC'
        if 'fiducial_data' in keys:
            return 'Diffusion'
        if {'fid_times', 'fid_labels', 'fid_Tmat', 'fid_Trad'}.issubset(keys):
            return 'SN'
        raise ValueError(f"Could not determine file type for {path}")


def _canonical_label(label):
    """Normalize label text so fiducial points can be matched across methods."""
    s = str(label).lower().strip()
    if ':' in s:
        s = s.split(':', 1)[1].strip()

    m = re.search(r'r\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*z\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', s)
    if m:
        rv = float(m.group(1))
        zv = float(m.group(2))
        return f'r={rv:.6f},z={zv:.6f}'

    return ' '.join(s.split())


def _format_display_label(label):
    """Readable fiducial label for logging/plot legend when needed."""
    s = str(label)
    return s.split(':', 1)[1].strip() if ':' in s else s


def _find_common_label_maps(datasets):
    """Return method -> {canonical_label: original_label} for shared fiducials only."""
    maps = {}
    for method, _, fid in datasets:
        cur_map = {}
        for original_label in fid.keys():
            canon = _canonical_label(original_label)
            if canon in cur_map:
                continue
            cur_map[canon] = original_label
        maps[method] = cur_map

    common = None
    for method in maps:
        keyset = set(maps[method].keys())
        common = keyset if common is None else (common & keyset)

    if not common:
        raise RuntimeError('No shared fiducial labels found across selected files.')

    common = sorted(common)
    out = {method: {canon: maps[method][canon] for canon in common} for method in maps}
    return common, out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_POINT_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
_MARKERS = ['o', 's', '^', 'd', 'v']
_LINESTYLE_BY_METHOD = {
    'IMC': '-',
    'Diffusion': '--',
    'SN': '-.',
}


def plot_comparison(datasets, outbase, SN_label='SN', IMC_label='IMC', Diffusion_label='Diffusion'):
    """
    Create combined plots with all fiducial points on one axis.
    Line styles indicate method (IMC, diffusion, SN).
    
    datasets : list of (method_label, times_arr, fiducial_data_dict)
    """
    common_canon_labels, label_maps = _find_common_label_maps(datasets)
    method_order = [method for method, _, _ in datasets]
    
    # Create two figures: material and radiation temperature
    for quantity, ylabel, suffix in [
        ('T_mat', 'temperature (keV)', 'material'),
        ('T_rad', r'radiation temperature (keV)', 'radiation')
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.5))
        
        # Plot each dataset with method-dependent linestyle
        for method, times, fiducial_data in datasets:
            linestyle = _LINESTYLE_BY_METHOD.get(method, ':')
            
            # Plot each fiducial point
            for pt_idx, canon_label in enumerate(common_canon_labels):
                pt_label = label_maps[method][canon_label]
                if pt_label not in fiducial_data:
                    print(f'Warning: fiducial label "{pt_label}" not found in {method} data; skipping.')
                    continue
                if quantity not in fiducial_data[pt_label]:
                    print(f'Warning: quantity "{quantity}" not found for fiducial label "{pt_label}" in {method} data; skipping.')
                    continue
                T = fiducial_data[pt_label][quantity]
                color = _POINT_COLORS[pt_idx % len(_POINT_COLORS)]
                marker = _MARKERS[pt_idx % len(_MARKERS)]
                
                ax.loglog(
                    times, T,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.8,
                    marker=marker,
                    markersize=4,
                    markevery=max(1, len(times) // 20),
                    alpha=0.85,
                )

        # Add invisible lines for the linestyle legend (method identity)
        for method_name in method_order:
            #set label to the user-provided label if available, otherwise use the default method name
            if method_name == 'IMC':
                label = IMC_label
            elif method_name == 'Diffusion':
                label = Diffusion_label
            elif method_name == 'SN':
                label = rf"S$_{{{SN_label}}}$"
            else:
                label = method_name
            linestyle = _LINESTYLE_BY_METHOD.get(method_name, ':')
            ax.plot([], [], color='black', linestyle=linestyle, linewidth=2.5, label=label)
        
        ax.set_xlabel('time (ns)')
        ax.set_ylabel(ylabel)
        
        # Match converging Marshak wave style: remove top/right spines, thicken bottom/left
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        
        # No legend frame (handled by rcParams)
        ax.legend(fontsize=12, loc='best', ncol=1)
        
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
        description='Compare IMC, diffusion, and SN crooked-pipe fiducial histories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'files', nargs='*',
        help='Two or three NPZ files to compare. If omitted, auto-detects IMC + diffusion + SN.',
    )
    parser.add_argument(
        '--output-base', type=str, default='crooked_pipe_imc_diffusion_sn_compare',
        help='Prefix for output PDF filenames.',
    )
    #add a parameter for the label for SN, IMC, and diffusion. If not provided, use the default labels.
    parser.add_argument(
        '--SN-label', nargs=1, type=str, default=['SN'],
        help='Label for the SN method. If not provided, defaults are used.',
    )   
    parser.add_argument(
        '--IMC-label', nargs=1, type=str, default=['IMC'],
        help='Label for the IMC method. If not provided, defaults are used.',
    )
    parser.add_argument(
        '--Diffusion-label', nargs=1, type=str, default=['Diffusion'],
        help='Label for the Diffusion method. If not provided, defaults are used.',
    )
    return parser.parse_args()


def auto_detect_paths():
    """Auto-detect one IMC file, one diffusion file, and one SN file."""
    imc_candidates = sorted(glob.glob('crooked_pipe_imc_solution_refined_*.npz'))
    diff_candidates = sorted(glob.glob('crooked_pipe_noneq_solution_refined_*.npz'))

    sn_candidates = sorted(glob.glob('crooked_pipe_sn_*.npz'))
    if not sn_candidates:
        sn_candidates = sorted(glob.glob('*sn*crooked*pipe*.npz'))
    if not sn_candidates:
        sn_candidates = sorted(glob.glob('sn_*crooked*pipe*.npz'))

    if not imc_candidates:
        print('Error: No IMC solution files found (crooked_pipe_imc_solution_refined_*.npz)')
        sys.exit(1)
    if not diff_candidates:
        print('Error: No diffusion solution files found (crooked_pipe_noneq_solution_refined_*.npz)')
        sys.exit(1)
    if not sn_candidates:
        print('Error: No SN solution files found (e.g. crooked_pipe_sn_*.npz)')
        sys.exit(1)

    imc_file = imc_candidates[-1]
    diff_pref = [f for f in diff_candidates if 'larsen' in f.lower()]
    diff_file = diff_pref[0] if diff_pref else diff_candidates[-1]
    sn_file = sn_candidates[-1]

    return [imc_file, diff_file, sn_file]


def method_sort_key(path):
    method = detect_file_type(path)
    order = {'IMC': 0, 'Diffusion': 1, 'SN': 2}
    return order.get(method, 99)


def main():
    args = parse_args()
    
    if args.files:
        if len(args.files) < 2 or len(args.files) > 3:
            print(f'Error: Need 2 or 3 files, got {len(args.files)}')
            sys.exit(1)
        npz_paths = sorted(args.files, key=method_sort_key)
    else:
        npz_paths = auto_detect_paths()
    
    print('Comparing:')
    for idx, path in enumerate(npz_paths, start=1):
        print(f'  File {idx}: {path}')
    
    # Load files
    datasets = []
    for path in npz_paths:
        file_type = detect_file_type(path)
        
        if file_type == 'IMC':
            times, fiducial_data = load_imc_npz(path)
            label = 'IMC'
        elif file_type == 'Diffusion':
            times, fiducial_data = load_diffusion_npz(path)
            label = 'Diffusion'
        elif file_type == 'SN':
            times, fiducial_data = load_sn_npz(path)
            label = 'SN'
        else:
            print(f'Error: Unsupported file type for {path}')
            sys.exit(1)
        
        print(f'  [{label}] {file_type} - {len(times)} time points')
        datasets.append((label, times, fiducial_data))

    methods = [m for m, _, _ in datasets]
    if ('IMC' not in methods) or ('Diffusion' not in methods):
        print('Error: At minimum, comparison requires IMC and diffusion files.')
        sys.exit(1)
    if 'SN' not in methods:
        print('Warning: SN file not provided; plotting only IMC and diffusion.')
    else:
        print('  Three-method comparison enabled: IMC vs Diffusion vs SN')
    
    # Create comparison plots
    print('\nGenerating plots...')
    plot_comparison(datasets, outbase=args.output_base, SN_label=args.SN_label[0], IMC_label=args.IMC_label[0], Diffusion_label=args.Diffusion_label[0])
    
    print('\nDone.')


if __name__ == '__main__':
    main()
