#!/usr/bin/env python3
"""
Compare crooked-pipe SP_N (SP3/SP7 with local filter), S16, and IMC
fiducial-point temperature histories.

Usage
-----
# Auto-detect standard paths:
python plot_crooked_pipe_spn_compare.py

# Explicit SP3 + SP7 + S16 + IMC files:
python plot_crooked_pipe_spn_compare.py \\
    results/crooked_pipe_gray/SP3/crooked_pipe_spn_148x266.npz \\
    results/crooked_pipe_gray/SP7/crooked_pipe_spn_148x266.npz \\
    DiscreteOrdinates2D/results/crooked_pipe/S16_fine/crooked_pipe_sn_168x354.npz \\
    DiscreteOrdinates2D/results/crooked_pipe/IMC/crooked_pipe_imc_solution_fc_refined_Nb400000_dtmax10.0_168x354.npz
"""

import sys
import os
import re
import argparse

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl

# ---------------------------------------------------------------------------
# Project root (two levels up from IMC/ where the PN companion script lives)
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(_HERE)))  # project root
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    from utils.plotfuncs import show, font as _font
    _HAS_PLOTFUNCS = True
except ImportError:
    _HAS_PLOTFUNCS = False

# ---------------------------------------------------------------------------
# Matplotlib style
# ---------------------------------------------------------------------------
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 12,
    'axes.labelsize': 12,
    'axes.edgecolor': 'black',
    'axes.linewidth': 1.15,
    'axes.grid': False,
    'lines.linewidth': 1.8,
    'legend.frameon': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
})

# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def _load_fiducial_npz(path):
    """Load SPN / PN / SN NPZ: fid_times, fid_labels, fid_Tmat, fid_Trad."""
    d = np.load(path, allow_pickle=True)
    times  = np.asarray(d['fid_times'])
    labels = [str(x) for x in np.asarray(d['fid_labels']).tolist()]
    Tmat_arr = np.asarray(d['fid_Tmat'])
    Trad_arr = np.asarray(d['fid_Trad'])
    fiducial_data = {
        lab: {'T_mat': Tmat_arr[:, j], 'T_rad': Trad_arr[:, j]}
        for j, lab in enumerate(labels)
    }
    return times, fiducial_data


def _load_imc_npz(path):
    """Load IMC NPZ: times, fiducial_data, fiducial_data_rad."""
    d = np.load(path, allow_pickle=True)
    times       = np.asarray(d['times'])
    fid_mat     = d['fiducial_data'].item()
    fid_rad     = d['fiducial_data_rad'].item()
    fiducial_data = {
        str(lab): {'T_mat': np.asarray(fid_mat[lab]),
                   'T_rad': np.asarray(fid_rad[lab])}
        for lab in fid_mat
    }
    return times, fiducial_data


def _detect_type(path):
    pl = path.replace('\\', '/').lower()
    base = os.path.basename(pl)
    if '/sp3/' in pl or 'sp3_' in base or '_sp3' in base: return 'SP3'
    if '/sp7/' in pl or 'sp7_' in base or '_sp7' in base: return 'SP7'
    if '/s16/' in pl or 's16' in base or '_sn_' in base:  return 'S16'
    if '/imc/' in pl or 'imc'  in base:                    return 'IMC'
    d = np.load(path, allow_pickle=True)
    keys = set(d.keys())
    if {'times', 'fiducial_data'}.issubset(keys): return 'IMC'
    return 'S16'   # fallback to fiducial NPZ format


def _canon(label):
    s = str(label).lower().strip()
    if ':' in s: s = s.split(':', 1)[1].strip()
    m = re.search(
        r'r\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*z\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', s)
    if m:
        return f'r={float(m.group(1)):.6f},z={float(m.group(2)):.6f}'
    return ' '.join(s.split())


def _common_labels(datasets):
    maps = {}
    for method, _, fid in datasets:
        m = {}
        for orig in fid:
            c = _canon(orig)
            m.setdefault(c, orig)
        maps[method] = m
    common = None
    for m in maps:
        common = set(maps[m]) if common is None else common & set(maps[m])
    common = sorted(common)
    out = {method: {c: maps[method][c] for c in common} for method in maps}
    return common, out


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

_POINT_COLORS = ['tab:blue', 'tab:red', 'tab:green', 'tab:purple', 'tab:orange']
_MARKERS      = ['o', 's', '^', 'd', 'v']
_LS = {'SP3': ':', 'SP7': '-.', 'S16': '--', 'IMC': '-'}

_LATEX_LABELS = {
    'SP3': r'SP$_3$',
    'SP7': r'SP$_7$',
    'S16': r'S$_{16}$',
    'IMC': 'IMC',
}


def plot_comparison(datasets, outbase):
    common_canon, label_maps = _common_labels(datasets)
    method_order = [m for m, _, _ in datasets]

    for quantity, ylabel, suffix in [
        ('T_mat', 'material temperature (keV)', 'material'),
        ('T_rad', r'radiation temperature (keV)', 'radiation'),
    ]:
        fig, ax = plt.subplots(figsize=(6, 4.5))

        for method, times, fid in datasets:
            ls = _LS.get(method, ':')
            for pt_idx, canon in enumerate(common_canon):
                orig = label_maps[method][canon]
                if orig not in fid or quantity not in fid[orig]:
                    continue
                T     = np.asarray(fid[orig][quantity])
                color  = _POINT_COLORS[pt_idx % len(_POINT_COLORS)]
                marker = _MARKERS[pt_idx % len(_MARKERS)]
                ax.loglog(times, T, color=color, linestyle=ls, linewidth=1.8,
                          marker=marker, markersize=4,
                          markevery=max(1, len(times) // 20), alpha=0.85)

        # Legend: one entry per method (black line, varying style)
        for method in method_order:
            ax.plot([], [], color='black', linestyle=_LS.get(method, ':'),
                    linewidth=2.5, label=_LATEX_LABELS.get(method, method))

        ax.set_xlabel('time (ns)')
        ax.set_ylabel(ylabel)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.legend(fontsize=12, loc='best', ncol=1, handlelength=3.5, handletextpad=1.0)
        plt.tight_layout()

        outname = f'{outbase}_{suffix}.pdf'
        plt.savefig(outname, dpi=600)
        print(f'Saved: {outname}')
        plt.close()


# ---------------------------------------------------------------------------
# Auto-detect default paths
# ---------------------------------------------------------------------------

def _auto_paths():
    candidates = {
        'SP3': os.path.join(_ROOT, 'results', 'crooked_pipe_gray', 'SP3_local_filt',
                            'crooked_pipe_spn_148x266.npz'),
        'SP7': os.path.join(_ROOT, 'results', 'crooked_pipe_gray', 'SP7_local_filt',
                            'crooked_pipe_spn_148x266.npz'),
        'S16': os.path.join(_ROOT, 'DiscreteOrdinates2D', 'results', 'crooked_pipe',
                            'S16_fine', 'crooked_pipe_sn_168x354.npz'),
        'IMC': os.path.join(_ROOT, 'DiscreteOrdinates2D', 'results', 'crooked_pipe',
                            'IMC', 'crooked_pipe_imc_solution_fc_refined_Nb400000_dtmax10.0_168x354.npz'),
    }
    missing = {k: v for k, v in candidates.items() if not os.path.exists(v)}
    if missing:
        print('Missing expected files:')
        for k, v in missing.items():
            print(f'  [{k}] {v}')
        sys.exit(1)
    return list(candidates.values())


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare crooked-pipe SP_N fiducial histories vs SN and IMC.')
    parser.add_argument('files', nargs='*',
                        help='NPZ files to compare (SP3, SP7, S16, IMC in any order).')
    parser.add_argument('--output-base', default='crooked_pipe_spn_compare',
                        help='Prefix for output PDF filenames.')
    args = parser.parse_args()

    npz_paths = args.files if args.files else _auto_paths()
    if not npz_paths:
        print('No files provided and auto-detection failed.')
        sys.exit(1)

    print('Comparing:')
    datasets = []
    for path in npz_paths:
        ftype = _detect_type(path)
        loader = _load_imc_npz if ftype == 'IMC' else _load_fiducial_npz
        times, fid = loader(path)
        print(f'  [{ftype:4s}] {len(times)} time points  —  {path}')
        datasets.append((ftype, times, fid))

    plot_comparison(datasets, outbase=args.output_base)
    print('Done.')


if __name__ == '__main__':
    main()
