#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Compare fiducial-point temperature histories between crooked-pipe P3, P7,
IMC, and S16 solutions.

Usage
-----
# Auto-detect the standard crooked-pipe result files:
python plot_crooked_pipe_p3_p7_imc_s16_compare.py

# Explicit files (any order):
python plot_crooked_pipe_p3_p7_imc_s16_compare.py \
    results/crooked_pipe_gray/P3/crooked_pipe_pn_148x266.npz \
    results/crooked_pipe_gray/P7/crooked_pipe_pn_148x266.npz \
    DiscreteOrdinates2D/results/crooked_pipe/IMC/crooked_pipe_imc_solution_fc_refined_Nb400000_dtmax10.0_168x354.npz \
    DiscreteOrdinates2D/results/crooked_pipe/S16_fine/crooked_pipe_sn_168x354.npz
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
# Matplotlib style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Univers LT Std", "TeX Gyre Heros", "Helvetica", "Arial", "DejaVu Sans"],
    "font.size": 12,
    "axes.labelsize": 12,
    "font.variant": "small-caps",
    "axes.titlesize": 18,
    "mathtext.fontset": "dejavusans",
    "mathtext.default": "it",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "savefig.bbox": "tight",
    "axes.edgecolor": "black",
    "axes.linewidth": 1.15,
    "axes.grid": False,
    "xtick.color": "black",
    "ytick.color": "black",
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.major.size": 6,
    "ytick.major.size": 6,
    "xtick.direction": "out",
    "ytick.direction": "out",
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
    "legend.frameon": False,
})


def load_fiducial_npz(path):
    """Load crooked-pipe fiducial NPZ file with fid_times/fid_Tmat/fid_Trad."""
    d = np.load(path, allow_pickle=True)
    required = {'fid_times', 'fid_labels', 'fid_Tmat', 'fid_Trad'}
    if not required.issubset(set(d.keys())):
        raise ValueError(f"File does not match fiducial format: {path}")

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


def load_imc_npz(path):
    """Load IMC solution NPZ file."""
    d = np.load(path, allow_pickle=True)
    required = {'times', 'fiducial_data', 'fiducial_data_rad'}
    if not required.issubset(set(d.keys())):
        raise ValueError(f"File does not match IMC fiducial format: {path}")

    times = np.asarray(d['times'])
    fiducial_mat = d['fiducial_data'].item()
    fiducial_rad = d['fiducial_data_rad'].item()

    fiducial_data = {}
    for label in fiducial_mat.keys():
        fiducial_data[str(label)] = {
            'T_mat': np.asarray(fiducial_mat[label]),
            'T_rad': np.asarray(fiducial_rad[label]),
        }
    return times, fiducial_data


def detect_file_type(path):
    """Classify a path as P3, P7, IMC, or S16/SN."""
    path_lower = path.replace('\\', '/').lower()
    basename = os.path.basename(path_lower)
    print(f'Detecting file type for {path} ...')
    if 'p3' in path_lower or basename.startswith('p3_') or 'crooked_pipe_gray/p3' in path_lower:
        return 'P3'
    if 'p9' in path_lower or basename.startswith('p9_') or 'crooked_pipe_gray/p9' in path_lower:
        return 'P9'
    if 'p7' in path_lower or basename.startswith('p7_') or 'crooked_pipe_gray/p7' in path_lower:
        return 'P7'
    if '/imc/' in path_lower or 'imc' in basename:
        return 'IMC'
    if '/s16/' in path_lower or 's16' in basename or ('_sn_' in basename):
        return 'S16'

    d = np.load(path, allow_pickle=True)
    keys = set(d.keys())
    if {'fid_times', 'fid_labels', 'fid_Tmat', 'fid_Trad'}.issubset(keys):
        return 'S16'
    if {'times', 'fiducial_data', 'fiducial_data_rad'}.issubset(keys):
        return 'IMC'
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


def _find_common_label_maps(datasets):
    """Return method -> {canonical_label: original_label} for shared fiducials only."""
    maps = {}
    for method, _, fid in datasets:
        cur_map = {}
        for original_label in fid.keys():
            canon = _canonical_label(original_label)
            if canon not in cur_map:
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


_POINT_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown']
_MARKERS = ['o', 's', '^', 'd', 'v', 'P']
_LINESTYLE_BY_METHOD = {
    'P3': ":",  # custom dash pattern
    'P7': '-.',
    'P9': '--',
    'IMC': '-',
    'S16': '-',
}


def plot_comparison(datasets, outbase):
    """Create material and radiation plots for the four-way comparison."""
    common_canon_labels, label_maps = _find_common_label_maps(datasets)
    method_order = [method for method, _, _ in datasets]

    for quantity, ylabel, suffix in [
        ('T_mat', 'temperature (keV)', 'material'),
        ('T_rad', r'radiation temperature (keV)', 'radiation'),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.5))

        for method, times, fiducial_data in datasets:
            linestyle = _LINESTYLE_BY_METHOD.get(method, ':')

            for pt_idx, canon_label in enumerate(common_canon_labels):
                pt_label = label_maps[method][canon_label]
                if pt_label not in fiducial_data:
                    print(f'Warning: fiducial label "{pt_label}" not found in {method} data; skipping.')
                    continue
                if quantity not in fiducial_data[pt_label]:
                    print(f'Warning: quantity "{quantity}" not found for fiducial label "{pt_label}" in {method} data; skipping.')
                    continue

                T = np.asarray(fiducial_data[pt_label][quantity])
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

        for method_name in method_order:
            linestyle = _LINESTYLE_BY_METHOD.get(method_name, ':')
            #if method is PN or SN make it have a latex subscript
            if method_name in ['P3', 'P7', 'S16', 'P9']:
                method_name = f"{method_name[0]}$_{{{method_name[1:]}}}$"
            ax.plot([], [], color='black', linestyle=linestyle, linewidth=2.5, label=method_name)

        ax.set_xlabel('time (ns)')
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        #make the legend wider
        ax.legend(fontsize=12, loc='best', ncol=1, handlelength=3.5, handletextpad=1.0)

        plt.tight_layout()
        outname = f'{outbase}_{suffix}.pdf'
        plt.savefig(outname, dpi=600)
        print(f'Saved: {outname}')
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare crooked-pipe P3, P7, IMC, and S16 fiducial histories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='Four NPZ files to compare. If omitted, auto-detects P3 + P7 + IMC + S16.',
    )
    parser.add_argument(
        '--output-base',
        type=str,
        default='crooked_pipe_p3_p7_imc_s16_compare',
        help='Prefix for output PDF filenames.',
    )
    return parser.parse_args()


def auto_detect_paths():
    """Auto-detect the standard crooked-pipe P3/P7/IMC/S16 files."""
    candidates = [
        os.path.join(project_root, 'results', 'crooked_pipe_gray', 'P3', 'crooked_pipe_pn_148x266.npz'),
        os.path.join(project_root, 'results', 'crooked_pipe_gray', 'P7', 'crooked_pipe_pn_148x266.npz'),
        os.path.join(project_root, 'DiscreteOrdinates2D', 'results', 'crooked_pipe', 'IMC', 'crooked_pipe_imc_solution_fc_refined_Nb400000_dtmax10.0_168x354.npz'),
        os.path.join(project_root, 'DiscreteOrdinates2D', 'results', 'crooked_pipe', 'S16_fine', 'crooked_pipe_sn_168x354.npz'),
    ]
    missing = [p for p in candidates if not os.path.exists(p)]
    if missing:
        print('Error: Missing expected crooked-pipe files:')
        for path in missing:
            print(f'  {path}')
        sys.exit(1)
    return candidates


def method_sort_key(path):
    method = detect_file_type(path)
    order = {'P3': 4, 'P7': 3, 'IMC': 0, 'S16': 1, 'P9': 2}
    return order.get(method, 99)


def main():
    args = parse_args()

    if args.files:
        if len(args.files) != 4:
            print(f'Error: Need exactly 4 files, got {len(args.files)}')
            sys.exit(1)
        npz_paths = sorted(args.files, key=method_sort_key)
    else:
        npz_paths = auto_detect_paths()

    print('Comparing:')
    for idx, path in enumerate(npz_paths, start=1):
        print(f'  File {idx}: {path}')

    datasets = []
    for path in npz_paths:
        file_type = detect_file_type(path)

        if file_type in ('P3', 'P7', 'S16', 'P9'):
            times, fiducial_data = load_fiducial_npz(path)
            label = file_type
            print(f'  [{label}] {len(times)} time points')
        elif file_type == 'IMC':
            times, fiducial_data = load_imc_npz(path)
            label = 'IMC'
        else:
            print(f'Error: Unsupported file type for {path}')
            sys.exit(1)

        print(f'  [{label}] {len(times)} time points')
        datasets.append((label, times, fiducial_data))

    print('\nGenerating plots...')
    plot_comparison(datasets, outbase=args.output_base)
    print('\nDone.')


if __name__ == '__main__':
    main()