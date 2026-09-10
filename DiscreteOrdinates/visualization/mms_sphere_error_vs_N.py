#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Angular accuracy of the spherical LD-S_N method vs. S_N order.

Opening figure for Section 10.8.  Using the manufactured solution
I(r,μ) = m r μ + b, we plot the relative error in the angular flux ψ as a
function of the S_N order N for several fixed spatial meshes I.

For each spatial resolution I ∈ {10, 100, 500} we sweep the even orders
N = 2, 4, …, 32 and record

    L²(ψ)   = sqrt( mean[ (ψ_num − ψ_exact)² / ψ_exact² ] )      (filled symbols)
    L∞(ψ)   = max[  |ψ_num − ψ_exact|     / |ψ_exact| ]          (open symbols)

The behaviour this figure reveals (and which the surrounding text explains):

  * N = 2 is machine-exact (the lone interior angular edge sits at μ = 0 where
    the odd part m r μ vanishes, and the boundary edges sit at μ = ±1 where
    1 − μ² = 0).
  * For N ≥ 4 a finite angular-closure error appears.  Refining the spatial
    mesh I lowers the error only until it hits the angular floor set by the
    conservative metric defect |α_{n+1/2} − (1−μ²_{n+1/2})| = O(N⁻²).
  * L∞ (open) sits above L² (filled) and need not improve with N: the worst
    error localises at the forward pole μ_n → +1 at r = 0 (pole amplification),
    even while the mean (L²) keeps falling.

Run
---
    cd DiscreteOrdinates
    python problems/mms_sphere_error_vs_N.py
    python problems/mms_sphere_error_vs_N.py --I 10 100 500 --Nmax 32
"""

import sys
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_sn_dir = os.path.dirname(_this_dir)            # DiscreteOrdinates/
sys.path.insert(0, _sn_dir)
sys.path.insert(0, _this_dir)

# Project root for the shared plotting style (utils.plotfuncs).
project_root = os.path.dirname(_sn_dir)
sys.path.insert(0, project_root)

from mms_sphere_sn import run_mms
from utils.plotfuncs import show, hide_spines, font


# ---------------------------------------------------------------------------
# Error collection
# ---------------------------------------------------------------------------

def psi_errors(I, N, R, sigma, m, b):
    """Return (L2, Linf) relative error of ψ for one (I, N) pair."""
    psi_num, psi_exact, _, _, _, _, _ = run_mms(I, N, R, sigma, m, b)
    rel = np.abs(psi_num - psi_exact) / np.maximum(np.abs(psi_exact), 1e-14)
    L2 = float(np.sqrt(np.mean(rel ** 2)))
    Linf = float(np.max(rel))
    return L2, Linf


def collect(I_vals, N_vals, R, sigma, m, b):
    """Build {I: (L2[:], Linf[:])} over the requested N values."""
    data = {}
    for I in I_vals:
        L2s, Linfs = [], []
        for N in N_vals:
            L2, Linf = psi_errors(I, N, R, sigma, m, b)
            L2s.append(L2)
            Linfs.append(Linf)
        data[I] = (np.array(L2s), np.array(Linfs))
    return data


def print_table(I_vals, N_vals, data):
    for I in I_vals:
        L2s, Linfs = data[I]
        print(f"\n  I = {I} radial cells")
        print(f"    {'N':>4}  {'L2(psi)':>12}  {'Linf(psi)':>12}")
        print("    " + "-" * 32)
        for k, N in enumerate(N_vals):
            print(f"    {N:>4}  {L2s[k]:>12.3e}  {Linfs[k]:>12.3e}")


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def _draw_curves(ax, I_vals, N_vals, data, colors, markers, floor, label=True):
    """Plot every (I) pair onto ax; filled = L², open = L∞.

    Each spatial mesh I gets a distinct colour AND a distinct marker, so the
    curves remain distinguishable in greyscale.
    """
    for k, I in enumerate(I_vals):
        c = colors[k % len(colors)]
        mk = markers[k % len(markers)]
        L2s, Linfs = data[I]
        # Filled symbols / solid line  -> L2
        ax.semilogy(N_vals, np.maximum(L2s, floor), '-', marker=mk, color=c,
                    lw=1.6, ms=6, label=(fr'$J = {I}$' if label else None))
        # Open symbols / dashed line   -> Linf
        ax.semilogy(N_vals, np.maximum(Linfs, floor), '--', marker=mk, color=c,
                    lw=1.4, ms=6, markerfacecolor='white', markeredgecolor=c)


def plot_error_vs_N(I_vals, N_vals, data, save_name):
    """Error vs N: filled = L², open = L∞, one colour per spatial mesh I.

    Main axes show the full N range; an inset zooms on N = 2–6 so the
    machine-exact N = 2 point and the onset of the angular-closure error
    are visible while the main panel shows the large-N trend.
    """
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
    markers = ['o', 's', '^', 'D', 'v']
    N_vals = np.asarray(N_vals)

    fig, ax = plt.subplots(figsize=(7, 5))

    floor = 1e-16
    _draw_curves(ax, I_vals, N_vals, data, colors, markers, floor, label=True)

    # Proxy handles to explain the fill convention (colour/marker-independent).
    proxy_l2 = plt.Line2D([], [], color='gray', marker='o', ls='-',
                          ms=6, label=r'$L_2$ (filled)')
    proxy_linf = plt.Line2D([], [], color='gray', marker='o', ls='--',
                            ms=6, markerfacecolor='white',
                            markeredgecolor='gray', label=r'$L_\infty$ (open)')

    ax.set_xlabel(r'$S_N$ order $N$')
    ax.set_ylabel(r'Relative error in $I(r,\mu)$')
    ax.set_ylim(bottom=1e-7)
    ax.set_xlim(left=3)
    ax.set_xscale('log')
    xticks = [4, 8, 10,20, 30]
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(t) for t in xticks])
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    handles, _ = ax.get_legend_handles_labels()
    leg = ax.legend(handles=handles + [proxy_l2, proxy_linf],
                    prop=font, frameon=False, ncol=2, loc='best')
    ax.add_artist(leg)
    ax.grid(True, which='both', alpha=0.25)

    # -----------------------------------------------------------------
    # Reference triangle: 2nd-order (N^-2) decay of the L² error.
    # On log-log axes a power law N^p is a straight line of slope p.
    # -----------------------------------------------------------------
    x1, x2 = 16, 32.
    y1 = 9.0e-4
    y2 = y1 * (x2 / x1) ** (-2.0)
    ax.plot([x1, x2], [y2, y2], '-', color='0.35', lw=1.2)   # horizontal leg
    ax.plot([x1, x1], [y1, y2], '-', color='0.35', lw=1.2)   # vertical leg
    ax.plot([x1, x2], [y1, y2], '-', color='0.35', lw=1.2)   # hypotenuse
    ax.text(np.sqrt(x1 * x2), y2 * 0.9, '1', color='0.35',
            ha='center', va='top', fontproperties=font, fontsize=10)
    ax.text(x1 * .97, np.sqrt(y1 * y2), '2', color='0.35',
            ha='right', va='center', fontproperties=font, fontsize=10)

    # -----------------------------------------------------------------
    # Inset: zoom on small N (2–8), where N = 2 is machine-exact.
    # Placed in the lower-right, below the curves.
    # -----------------------------------------------------------------
    N_inset = N_vals[N_vals <= 8]
    data_inset = {I: (data[I][0][:len(N_inset)], data[I][1][:len(N_inset)])
                  for I in I_vals}

    axin = ax.inset_axes([0.18, 0.08, 0.35, 0.42])
    _draw_curves(axin, I_vals, N_inset, data_inset, colors, markers, floor,
                 label=False)
    axin.set_xticks(N_inset)
    axin.set_xscale('linear')
    axin.tick_params(labelsize=9)
    axin.grid(True, which='both', alpha=0.25)

    plt.tight_layout()
    show(save_name, close_after=True)
    print(f"\n  Saved  {save_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Angular accuracy vs S_N order for the spherical LD-S_N '
                    'method (opening figure for Section 10.8).')
    parser.add_argument('--I', type=int, nargs='+', default=[10, 100, 500],
                        help='Spatial meshes (radial cells); default 10 100 500')
    parser.add_argument('--Nmax', type=int, default=32,
                        help='Largest even S_N order (default 32)')
    parser.add_argument('--R', type=float, default=3.0, help='Outer radius (cm)')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Absorption opacity σ (cm⁻¹)')
    parser.add_argument('--m', type=float, default=0.5,
                        help='Slope m for I = m r μ + b')
    parser.add_argument('--b', type=float, default=2.0,
                        help='Offset b for I = m r μ + b')
    parser.add_argument('--out', default='mms_sph_error_vs_N.pdf',
                        help='Output figure file name')
    args = parser.parse_args()

    N_vals = np.arange(2, args.Nmax + 1, 2)

    print("\nAngular accuracy vs S_N order — spherical LD-S_N (Section 10.8)")
    print(f"  I_exact(r,μ) = {args.m}·r·μ + {args.b}")
    print(f"  σ = {args.sigma} cm⁻¹,  R = {args.R} cm")
    print(f"  spatial meshes I = {args.I}")
    print(f"  N = {list(N_vals)}")

    data = collect(args.I, N_vals, args.R, args.sigma, args.m, args.b)
    print_table(args.I, N_vals, data)
    plot_error_vs_N(args.I, N_vals, data, args.out)

    print("\nDone.")


if __name__ == '__main__':
    main()
