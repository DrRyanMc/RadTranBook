#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Compare refinement-test fiducial histories across S_N quadrature orders.

Loads the NPZ files written by save_fiducials_npz() from multiple result
directories and overlays them on the same axes.

Default usage (run from results/refinement/):
    python ../../problems/plot_refinement_fiducial_compare.py

Explicit directories:
    python ../../problems/plot_refinement_fiducial_compare.py \\
        --dirs N4_I100 N8_I100 N16_I100 \\
        --labels '$S_4$' '$S_8$' '$S_{16}$'
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ── plotfuncs (optional) ───────────────────────────────────────────────────
_here = os.path.dirname(os.path.abspath(__file__))
try:
    sys.path.insert(0, os.path.join(_here, '..', '..'))
    from utils.plotfuncs import hide_spines as _hide_spines, font as _font
    HAS_PLOTFUNCS = True
except ImportError:
    HAS_PLOTFUNCS = False


# ── default line styles per solution (lowest → highest order) ─────────────
DEFAULT_STYLES = ['-.', '--', '-']           # S4, S8, S16
DEFAULT_LABELS = [r'$S_4$', r'$S_8$', r'$S_{16}$']

DEFAULT_PN_DIRS = ['P1', 'P3', 'P7']
DEFAULT_PN_STYLES = ['-.', '--', '-']
DEFAULT_PN_LABELS = [r'$P_1$', r'$P_3$', r'$P_7$']

# ── one color per fiducial point (matches plot_fiducial_history) ──────────
FID_COLORS = ['blue', 'red', 'green', 'purple']


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_fiducials(npz_path):
    """Return (times, labels, Tmat_list, Trad_list) from a fiducials NPZ."""
    d = np.load(npz_path, allow_pickle=True)
    labels = list(d['fid_labels'])
    times  = d['fid_times']
    Tmat   = [d[f'fid_Tmat_{j}'] for j in range(len(labels))]
    Trad   = [d[f'fid_Trad_{j}'] for j in range(len(labels))]
    return times, labels, Tmat, Trad


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _make_legend_handles(fid_labels, sol_labels, sol_styles):
    """Build two groups of proxy legend handles."""
    # Group 1: colored solid lines for fiducial points
    fid_handles = [
        Line2D([0], [0], color=FID_COLORS[j % len(FID_COLORS)],
               lw=2, ls='-', label=lbl)
        for j, lbl in enumerate(fid_labels)
    ]
    # Group 2: black lines with varying styles for solution orders
    sol_handles = [
        Line2D([0], [0], color='k', lw=2, ls=ls, label=lbl)
        for ls, lbl in zip(sol_styles, sol_labels)
    ]
    return fid_handles, sol_handles


def plot_compare(solutions, field='T_mat', savefile='refinement_compare_T_mat.pdf'):
    """
    Overlay fiducial histories for all solutions on one axes.

    Parameters
    ----------
    solutions : list of dicts, each with keys
        'times', 'fid_labels', 'Tmat', 'Trad', 'label', 'ls'
    field     : 'T_mat' or 'T_rad'
    savefile  : output filename
    """
    fp = _font if HAS_PLOTFUNCS else None

    fig, ax = plt.subplots(figsize=(8, 5))

    fid_labels = solutions[0]['fid_labels']   # same for all

    for sol in solutions:
        data_list = sol['Tmat'] if field == 'T_mat' else sol['Trad']
        for j, Tv in enumerate(data_list):
            ts = sol['times']
            ax.loglog(
                ts, Tv,
                color=FID_COLORS[j % len(FID_COLORS)],
                ls=sol['ls'],
                lw=2,
                alpha=0.85,
            )

    ylabel = ('Material Temperature (keV)' if field == 'T_mat'
              else 'Radiation Temperature (keV)')
    ax.set_xlabel('Time (ns)',  fontproperties=fp, fontsize=12)
    ax.set_ylabel(ylabel,        fontproperties=fp, fontsize=12)
    ax.grid(True, which='both', alpha=0.3, ls='--')

    # Two-group legend
    fid_handles, sol_handles = _make_legend_handles(
        fid_labels,
        [s['label'] for s in solutions],
        [s['ls']    for s in solutions],
    )
    leg = ax.legend(
        handles=fid_handles + sol_handles,
        fontsize=9, ncol=1,
        title='Point / Order',
        title_fontsize=9,
    )
    if HAS_PLOTFUNCS:
        for txt in leg.get_texts():
            txt.set_fontproperties(_font)
            txt.set_fontsize(9)

    plt.tight_layout()
    if HAS_PLOTFUNCS:
        _hide_spines()
    plt.savefig(savefile, dpi=600, bbox_inches='tight')
    print(f'Saved: {savefile}')
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Compare refinement-test fiducial histories across S_N orders.')
    parser.add_argument(
        '--mode', choices=['sn', 'pn'], default='sn',
        help="Comparison family defaults: 'sn' (S4/S8/S16) or 'pn' (P1/P3/P7).")
    parser.add_argument(
        '--dirs', nargs='+',
        default=None,
        help='Result directories to compare.')
    parser.add_argument(
        '--labels', nargs='+',
        default=None,
        help='Legend labels for each directory.')
    parser.add_argument(
        '--styles', nargs='+',
        default=None,
        help='Line styles for each directory (default: -. -- -).')
    parser.add_argument(
        '--npz', default=None,
        help='NPZ filename within each directory.')
    parser.add_argument(
        '--outprefix', default='refinement_compare',
        help='Output file prefix.')
    args = parser.parse_args()

    if args.mode == 'pn':
        if args.dirs is None:
            args.dirs = DEFAULT_PN_DIRS
        if args.labels is None:
            args.labels = DEFAULT_PN_LABELS
        if args.styles is None:
            args.styles = DEFAULT_PN_STYLES
        if args.npz is None:
            args.npz = 'refinement_test_p{order}_fiducials.npz'
    else:
        if args.dirs is None:
            args.dirs = ['N4_I100', 'N8_I100', 'N16_I100']
        if args.labels is None:
            args.labels = DEFAULT_LABELS
        if args.styles is None:
            args.styles = DEFAULT_STYLES
        if args.npz is None:
            args.npz = 'refinement_test_sn_fiducials.npz'

    if len(args.labels) != len(args.dirs):
        parser.error('--labels must have the same length as --dirs')
    if len(args.styles) != len(args.dirs):
        parser.error('--styles must have the same length as --dirs')

    # Load all solutions
    solutions = []
    for d, lbl, ls in zip(args.dirs, args.labels, args.styles):
        npz_name = args.npz
        if '{order}' in npz_name:
            order_token = ''.join(ch for ch in os.path.basename(d) if ch.isdigit())
            npz_name = npz_name.format(order=order_token)
        npz_path = os.path.join(d, npz_name)
        if not os.path.exists(npz_path):
            print(f'WARNING: {npz_path} not found — skipping.')
            continue
        times, fid_labels, Tmat, Trad = load_fiducials(npz_path)
        solutions.append(dict(
            times=times, fid_labels=fid_labels,
            Tmat=Tmat, Trad=Trad,
            label=lbl, ls=ls,
        ))
        print(f'Loaded {npz_path}  ({len(fid_labels)} fiducials, {len(times)} steps)')

    if not solutions:
        print('No NPZ files found.')
        return

    plot_compare(solutions, field='T_mat',
                 savefile=f'{args.outprefix}_T_mat.pdf')
    plot_compare(solutions, field='T_rad',
                 savefile=f'{args.outprefix}_T_rad.pdf')


if __name__ == '__main__':
    main()
