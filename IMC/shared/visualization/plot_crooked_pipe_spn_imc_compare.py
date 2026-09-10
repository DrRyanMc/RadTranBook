#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[3]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Compare fiducial-point temperature histories between crooked-pipe SP3, SP7,
IMC, and S16 solutions.

Usage
-----
# Auto-detect the standard crooked-pipe result files:
python plot_crooked_pipe_spn_imc_compare.py

# Explicit files (any order, SP3 + SP7 + IMC + S16):
python plot_crooked_pipe_spn_imc_compare.py \
    results/crooked_pipe_gray/SP3/crooked_pipe_spn_148x266.npz \
    results/crooked_pipe_gray/SP7/crooked_pipe_spn_148x266.npz \
    DiscreteOrdinates2D/results/crooked_pipe/IMC/crooked_pipe_imc_solution_fc_refined_Nb400000_dtmax10.0_168x354.npz \
    DiscreteOrdinates2D/results/crooked_pipe/S16_fine/crooked_pipe_sn_168x354.npz

# SP3 + SP7 + IMC only (no S16):
python plot_crooked_pipe_spn_imc_compare.py --no-s16
"""

import sys
import os
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(_HERE)
if project_root not in sys.path:
    sys.path.insert(0, project_root)

try:
    from utils.plotfuncs import show, font
except ImportError:
    pass

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


# ---------------------------------------------------------------------------
# Loaders  (SPN and PN files share the same NPZ layout)
# ---------------------------------------------------------------------------

def load_fiducial_npz(path):
    """Load crooked-pipe fiducial NPZ (PN or SPN format)."""
    d = np.load(path, allow_pickle=True)
    required = {'fid_times', 'fid_labels', 'fid_Tmat', 'fid_Trad'}
    if not required.issubset(set(d.keys())):
        raise ValueError(f"File does not match fiducial format: {path}")

    times = np.asarray(d['fid_times'])
    labels = [str(x) for x in np.asarray(d['fid_labels']).tolist()]
    fid_tmat_arr = np.asarray(d['fid_Tmat'])
    fid_trad_arr = np.asarray(d['fid_Trad'])

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


# ---------------------------------------------------------------------------
# File-type detection
# ---------------------------------------------------------------------------

def detect_file_type(path):
    """Classify a path as SP3, SP7, IMC, or S16."""
    p = path.replace('\\', '/').lower()
    base = os.path.basename(p)

    if '/sp1' in p or 'sp1_' in base or '_sp1' in base:
        return 'SP1'
    if '/sp3' in p or 'sp3_' in base or '_sp3' in base:
        return 'SP3'
    if '/sp7' in p or 'sp7_' in base or '_sp7' in base:
        return 'SP7'
    if '/p9' in p or 'p9_' in base or '_p9' in base:
        return 'P9'
    if '/imc/' in p or 'imc' in base:
        return 'IMC'
    if '/s16/' in p or 's16' in base or '_sn_' in base:
        return 'S16'

    # Fallback: inspect keys
    d = np.load(path, allow_pickle=True)
    keys = set(d.keys())
    if {'fid_times', 'fid_labels', 'fid_Tmat', 'fid_Trad'}.issubset(keys):
        return 'S16'
    if {'times', 'fiducial_data', 'fiducial_data_rad'}.issubset(keys):
        return 'IMC'
    raise ValueError(f"Could not determine file type for {path}")


# ---------------------------------------------------------------------------
# Label canonicalisation
# ---------------------------------------------------------------------------

def _canonical_label(label):
    s = str(label).lower().strip()
    if ':' in s:
        s = s.split(':', 1)[1].strip()
    m = re.search(
        r'r\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*z\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', s)
    if m:
        return f'r={float(m.group(1)):.6f},z={float(m.group(2)):.6f}'
    return ' '.join(s.split())


def _find_common_label_maps(datasets):
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


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_POINT_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange', 'tab:brown']
_MARKERS = ['o', 's', '^', 'd', 'v', 'P']

_LINESTYLE_BY_METHOD = {
    'SP1': (0, (1, 1)),
    'SP3': ':',
    'SP7': '-.',
    'P9':  '-',
    'S16': (0, (3, 1, 1, 1)),
    'IMC': '-',
}

_LATEX_LABEL = {
    'SP1': r'SP$_1$',
    'SP3': r'SP$_3$',
    'SP7': r'SP$_7$',
    'P9':  r'P$_9$',
    'S16': r'S$_{16}$',
    'IMC': 'IMC',
}

_METHOD_ORDER = ['IMC', 'S16', 'P9', 'SP7', 'SP3', 'SP1']


def plot_comparison(datasets, outbase):
    common_canon_labels, label_maps = _find_common_label_maps(datasets)
    method_order = [m for m, _, _ in datasets]

    for quantity, ylabel, suffix in [
        ('T_mat', 'temperature (keV)', 'material'),
        ('T_rad', 'radiation temperature (keV)', 'radiation'),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.5))

        for method, times, fiducial_data in datasets:
            linestyle = _LINESTYLE_BY_METHOD.get(method, ':')

            for pt_idx, canon_label in enumerate(common_canon_labels):
                pt_label = label_maps[method][canon_label]
                if pt_label not in fiducial_data:
                    continue
                if quantity not in fiducial_data[pt_label]:
                    continue

                T = np.asarray(fiducial_data[pt_label][quantity])
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

        for method_name in method_order:
            linestyle = _LINESTYLE_BY_METHOD.get(method_name, ':')
            display = _LATEX_LABEL.get(method_name, method_name)
            ax.plot([], [], color='black', linestyle=linestyle,
                    linewidth=2.5, label=display)

        ax.set_xlabel('time (ns)')
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.legend(fontsize=12, loc='best', ncol=1,
                  handlelength=3.5, handletextpad=1.0)

        plt.tight_layout()
        outname = f'{outbase}_{suffix}.pdf'
        plt.savefig(outname, dpi=600)
        print(f'Saved: {outname}')
        plt.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def method_sort_key(path):
    method = detect_file_type(path)
    order = {'IMC': 0, 'S16': 1, 'P9': 2, 'SP7': 3, 'SP3': 4, 'SP1': 5}
    return order.get(method, 99)


def auto_detect_paths(include_s16=True, include_imc=True, include_p9=False):
    candidates = [
        os.path.join(project_root, 'results', 'crooked_pipe_gray', 'SP3',
                     'crooked_pipe_spn_148x266.npz'),
        os.path.join(project_root, 'results', 'crooked_pipe_gray', 'SP7',
                     'crooked_pipe_spn_148x266.npz'),
        os.path.join(project_root, 'results', 'crooked_pipe_gray', 'SP1_no_filt',
                     'crooked_pipe_spn_148x266.npz'),
    ]
    if include_imc:
        candidates.append(
            os.path.join(project_root, 'DiscreteOrdinates2D', 'results', 'crooked_pipe',
                         'IMC', 'crooked_pipe_imc_solution_fc_refined_Nb400000_dtmax10.0_168x354.npz'))
    if include_s16:
        candidates.append(
            os.path.join(project_root, 'DiscreteOrdinates2D', 'results', 'crooked_pipe',
                         'S16_fine', 'crooked_pipe_sn_168x354.npz'))
    if include_p9:
        candidates.append(
            os.path.join(project_root, 'results', 'crooked_pipe_gray', 'P9',
                         'crooked_pipe_pn_148x266.npz'))

    missing = [p for p in candidates if not os.path.exists(p)]
    if missing:
        print('Error: Missing expected crooked-pipe files:')
        for p in missing:
            print(f'  {p}')
        sys.exit(1)
    return candidates


def parse_args():
    parser = argparse.ArgumentParser(
        description='Compare crooked-pipe SP3, SP7, IMC, and S16 fiducial histories.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('files', nargs='*',
                        help='NPZ files to compare (auto-detected if omitted).')
    parser.add_argument('--output-base', type=str,
                        default='crooked_pipe_spn_imc_compare',
                        help='Prefix for output PDF filenames.')
    parser.add_argument('--no-s16', action='store_true',
                        help='Omit S16 from the comparison.')
    parser.add_argument('--no-imc', action='store_true',
                        help='Omit IMC from the comparison.')
    parser.add_argument('--p9', action='store_true',
                        help='Include filtered P9 in the comparison.')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.files:
        npz_paths = sorted(args.files, key=method_sort_key)
    else:
        npz_paths = auto_detect_paths(include_s16=not args.no_s16,
                                      include_imc=not args.no_imc,
                                      include_p9=args.p9)
        npz_paths = sorted(npz_paths, key=method_sort_key)

    print('Comparing:')
    for idx, path in enumerate(npz_paths, 1):
        print(f'  File {idx}: {path}')

    datasets = []
    for path in npz_paths:
        file_type = detect_file_type(path)
        if file_type in ('SP1', 'SP3', 'SP7', 'S16', 'P9'):
            times, fiducial_data = load_fiducial_npz(path)
        elif file_type == 'IMC':
            times, fiducial_data = load_imc_npz(path)
        else:
            print(f'Error: Unsupported file type for {path}')
            sys.exit(1)
        print(f'  [{file_type}] {len(times)} time points, '
              f'{len(fiducial_data)} fiducial points')
        datasets.append((file_type, times, fiducial_data))

    print('\nGenerating plots...')
    plot_comparison(datasets, outbase=args.output_base)
    print('Done.')


if __name__ == '__main__':
    main()
