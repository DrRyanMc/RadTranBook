#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Compare fiducial-point temperature histories between three SN-order
crooked-pipe solutions (for example S4, S8, and S16).

Usage
-----
# Auto-detect common refined runs in current directory tree:
python plot_crooked_pipe_sn_orders_compare.py

# Explicit files (any order):
python plot_crooked_pipe_sn_orders_compare.py \
    S4_fine/crooked_pipe_sn_168x354.npz \
    S8_fine/crooked_pipe_sn_168x354.npz \
    S16_fine/crooked_pipe_sn_168x354.npz
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


def load_sn_npz(path):
    """Load discrete ordinates (SN) solution NPZ file."""
    d = np.load(path, allow_pickle=True)
    required = {'fid_times', 'fid_labels', 'fid_Tmat', 'fid_Trad'}
    if not required.issubset(set(d.keys())):
        raise ValueError(f"File does not match SN fiducial format: {path}")

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
    return np.asarray(times), fiducial_data


def _canonical_label(label):
    """Normalize label text so fiducial points can be matched across runs."""
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
    """Return run -> {canonical_label: original_label} for shared fiducials only."""
    maps = {}
    for run_label, _, fid in datasets:
        cur_map = {}
        for original_label in fid.keys():
            canon = _canonical_label(original_label)
            if canon not in cur_map:
                cur_map[canon] = original_label
        maps[run_label] = cur_map

    common = None
    for run_label in maps:
        keyset = set(maps[run_label].keys())
        common = keyset if common is None else (common & keyset)

    if not common:
        raise RuntimeError('No shared fiducial labels found across selected SN files.')

    common = sorted(common)
    out = {run_label: {canon: maps[run_label][canon] for canon in common} for run_label in maps}
    return common, out


def _extract_sn_order(path):
    """Try to infer S_N order from path (e.g., .../S8_... or ..._S16_...)."""
    text = path.replace('\\\\', '/').lower()
    m = re.search(r'(^|[/_])s(\d+)([/_]|$)', text)
    if m:
        return int(m.group(2))
    return None


def _run_label(path, fallback_index):
    order = _extract_sn_order(path)
    if order is not None:
        return rf'S$_{{{order}}}$'
    return f'SN run {fallback_index + 1}'


def _sort_paths_by_order(paths):
    def key_func(p):
        order = _extract_sn_order(p)
        return (order if order is not None else 10**9, p)
    return sorted(paths, key=key_func)


def auto_detect_paths():
    """Auto-detect SN files, preferring S4/S8/S16 refined outputs if present."""
    cands = []
    cands.extend(glob.glob('S*/crooked_pipe_sn_*.npz'))
    cands.extend(glob.glob('crooked_pipe_sn_*.npz'))
    cands = sorted(set(cands))

    if not cands:
        print('Error: No SN solution files found (e.g. S8_fine/crooked_pipe_sn_*.npz).')
        sys.exit(1)

    by_order = {}
    for p in cands:
        order = _extract_sn_order(p)
        if order is None:
            continue
        if order not in by_order:
            by_order[order] = []
        by_order[order].append(p)

    preferred = []
    for order in (4, 8, 16):
        if order in by_order:
            preferred.append(sorted(by_order[order])[-1])

    if len(preferred) == 3:
        return preferred

    cands_sorted = _sort_paths_by_order(cands)
    if len(cands_sorted) < 3:
        print('Error: Need at least 3 SN files to compare.')
        sys.exit(1)
    return cands_sorted[:3]


_POINT_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
_MARKERS = ['o', 's', '^', 'd', 'v']
_LINESTYLES = ['-', '--', '-.']


def plot_comparison(datasets, outbase):
    """
    Create combined plots with all fiducial points on one axis.
    Colors indicate fiducial points; line styles indicate SN order.

    datasets : list of (run_label, times_arr, fiducial_data_dict)
    """
    common_canon_labels, label_maps = _find_common_label_maps(datasets)

    for quantity, ylabel, suffix in [
        ('T_mat', 'temperature (keV)', 'material'),
        ('T_rad', r'radiation temperature (keV)', 'radiation'),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.5))

        for ds_idx, (run_label, times, fiducial_data) in enumerate(datasets):
            linestyle = _LINESTYLES[ds_idx % len(_LINESTYLES)]

            for pt_idx, canon_label in enumerate(common_canon_labels):
                pt_label = label_maps[run_label][canon_label]
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

        for ds_idx, (run_label, _, _) in enumerate(datasets):
            linestyle = _LINESTYLES[ds_idx % len(_LINESTYLES)]
            ax.plot([], [], color='black', linestyle=linestyle, linewidth=2.5, label=run_label)

        ax.set_xlabel('time (ns)')
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.legend(fontsize=12, loc='best', ncol=1)

        plt.tight_layout()
        outname = f'{outbase}_{suffix}.pdf'
        plt.savefig(outname, dpi=600)
        print(f'Saved: {outname}')
        plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare three SN-order crooked-pipe fiducial histories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        'files',
        nargs='*',
        help='Three SN NPZ files to compare. If omitted, auto-detects preferred S4/S8/S16 files.',
    )
    parser.add_argument(
        '--output-base',
        type=str,
        default='crooked_pipe_sn_orders_compare',
        help='Prefix for output PDF filenames.',
    )
    return parser.parse_args()


def main():
    args = parse_args()

    if args.files:
        if len(args.files) != 3:
            print(f'Error: Need exactly 3 SN files, got {len(args.files)}')
            sys.exit(1)
        npz_paths = _sort_paths_by_order(args.files)
    else:
        npz_paths = auto_detect_paths()

    print('Comparing SN runs:')
    for idx, path in enumerate(npz_paths, start=1):
        print(f'  File {idx}: {path}')

    datasets = []
    for idx, path in enumerate(npz_paths):
        times, fiducial_data = load_sn_npz(path)
        label = _run_label(path, idx)
        print(f'  [{label}] {len(times)} time points')
        datasets.append((label, times, fiducial_data))

    print('\nGenerating plots...')
    plot_comparison(datasets, outbase=args.output_base)
    print('\nDone.')


if __name__ == '__main__':
    main()
