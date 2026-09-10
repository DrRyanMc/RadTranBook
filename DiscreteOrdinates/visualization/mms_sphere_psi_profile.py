#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Angular flux ψ_n(r) for the spherical LD-S_N manufactured solution.

Companion figure for Section 10.8.  The manufactured intensity is
I(r,μ) = m r μ + b, which along any fixed ordinate μ_n is an exact straight
line in r.  On a coarse spatial mesh the numerical ψ_n(r) bends visibly away
from these straight lines near the origin: the forward ordinate μ → +1 dips
below every other ordinate and then crosses back over them as r grows.  This
near-origin transient is a spatial LD effect amplified by the 1/r geometry
(it vanishes as O(h) under radial refinement); the conservative angular closure
itself preserves constants exactly, so ψ_n(0) = b in the semi-discrete limit.

This script plots ψ_n(r) for every ordinate of an S_8 calculation.  All curves
are drawn in black; a single annotation arrow indicates the direction of
increasing μ_n across the family of curves, and an inset zooms on the
near-origin region where the traces bend away from the exact lines.

Run
---
    cd DiscreteOrdinates
    python problems/mms_sphere_psi_profile.py
    python problems/mms_sphere_psi_profile.py --N 8 --I 10
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
from utils.plotfuncs import show, font


# ---------------------------------------------------------------------------
# Build ψ_n(r) on a per-edge mesh
# ---------------------------------------------------------------------------

def psi_profiles(I, N, R, sigma, m, b):
    """Return (r, psi_num[:, n], psi_exact[:, n], MU); r = interleaved edges."""
    psi_num, _, r_left, dr, MU, _, _ = run_mms(I, N, R, sigma, m, b)
    # Interleave left/right edges so the discontinuous LD trace is visible.
    r = np.empty(2 * I)
    r[0::2] = r_left
    r[1::2] = r_left + dr
    psi = np.empty((2 * I, N))
    for n in range(N):
        psi[0::2, n] = psi_num[:, n, 0]
        psi[1::2, n] = psi_num[:, n, 1]
    # Exact solution I(r, μ) = m r μ + b along each ordinate.
    psi_ex = m * np.outer(r, MU) + b
    return r, psi, psi_ex, MU


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_psi_profiles(I, N, R, sigma, m, b, save_name):
    """ψ_n(r) for all ordinates, drawn in black, with a μ-direction arrow.

    Solid black = numerical LD-S_N trace; dotted = exact I(r,μ) = m r μ + b
    along each ordinate.  The extreme ordinates are labelled at the right
    edge, and an inset zooms on the near-origin region where the numerical
    traces bend away from the exact straight lines.
    """
    r, psi, psi_ex, MU = psi_profiles(I, N, R, sigma, m, b)

    fig, ax = plt.subplots(figsize=(7, 5))

    for n in range(N):
        ax.plot(r, psi[:, n], '-', color='black', lw=1.2)
        ax.plot(r, psi_ex[:, n], ':', color='black', lw=1.0)

    ax.set_xlabel(r'radius $r$ (cm)')
    ax.set_ylabel(r'specific intensity $I_n(r)$')
    ax.set_xlim(0.0, R)
    ax.set_ylim(bottom=0.0)
    ax.grid(True, which='both', alpha=0.25)

    # Proxy legend distinguishing numerical vs. exact.
    proxy_num = plt.Line2D([], [], color='black', ls='-', lw=1.2,
                           label=r'LD-S$_N$')
    proxy_ex = plt.Line2D([], [], color='black', ls=':', lw=1.0,
                          label=r'exact $\,m r \mu + b$')
    ax.legend(handles=[proxy_num, proxy_ex], prop=font, frameon=False,
              loc='upper left')

    # -----------------------------------------------------------------
    # Label the extreme ordinates at the right edge.
    # -----------------------------------------------------------------
    order = np.argsort(MU)
    n_lo = order[0]      # most negative μ
    n_hi = order[-1]     # most positive μ
    ax.text(R * 1.005, psi[-1, n_hi], fr'$\mu \approx {MU[n_hi]:+.2f}$',
            color='black', ha='left', va='center',
            fontproperties=font, fontsize=10)
    ax.text(R * 1.005, psi[-1, n_lo], fr'$\mu \approx {MU[n_lo]:+.2f}$',
            color='black', ha='left', va='center',
            fontproperties=font, fontsize=10)

    # -----------------------------------------------------------------
    # Annotation arrow showing the direction of increasing μ.
    # -----------------------------------------------------------------
    r_idx = int(0.82 * (len(r) - 1))
    x_a = 3.0 #r[r_idx]
    y_lo = psi[r_idx, n_lo]
    y_hi = psi[r_idx, n_hi]

    ax.annotate('', xy=(x_a, y_hi), xytext=(x_a, y_lo),
                arrowprops=dict(arrowstyle='-|>', color='tab:red', lw=1.8))
    ax.text(x_a * 1.02, 0.5 * (y_lo + y_hi),
            r'increasing $\mu$', color='tab:red',
            rotation=90, ha='left', va='center',
            fontproperties=font, fontsize=11)

    #ax.set_title(fr'$S_{{{N}}}$ angular flux, $I = {I}$ cells',
    #             fontproperties=font, fontsize=13)

    # -----------------------------------------------------------------
    # Inset: zoom on the near-origin region r ∈ [0, 0.6] where the
    # numerical traces bend away from the exact straight lines.
    # -----------------------------------------------------------------
    r_zoom = 6/I
    mask = r <= r_zoom
    axin = ax.inset_axes([0.1, 0.1, 0.25, 0.25])
    for n in range(N):
        axin.plot(r[mask], psi[mask, n], '-', color='black', lw=1.1)
        axin.plot(r[mask], psi_ex[mask, n], ':', color='black', lw=0.9)
    axin.set_xlim(0.0, r_zoom)
    axin.tick_params(labelsize=8)
    axin.set_title('Zoom near $r=0$', fontproperties=font, fontsize=10)
    axin.grid(True, which='both', alpha=0.25)

    plt.tight_layout()
    show(save_name, close_after=True)
    print(f"\n  Saved  {save_name}")
    print(f"  μ ordinates: {np.array2string(np.sort(MU), precision=4)}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Angular-flux profiles ψ_n(r) for the spherical LD-S_N '
                    'manufactured solution (companion to mms_sphere_error_vs_N).')
    parser.add_argument('--I', type=int, default=10,
                        help='Spatial mesh (radial cells); default 10 '
                             '(coarse, so the near-origin dip/crossover shows)')
    parser.add_argument('--N', type=int, default=8,
                        help='S_N order (default 8)')
    parser.add_argument('--R', type=float, default=3.0, help='Outer radius (cm)')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Absorption opacity σ (cm⁻¹)')
    parser.add_argument('--m', type=float, default=0.5,
                        help='Slope m for I = m r μ + b')
    parser.add_argument('--b', type=float, default=2.0,
                        help='Offset b for I = m r μ + b')
    parser.add_argument('--out', default='mms_sph_psi_profile.pdf',
                        help='Output figure file name')
    args = parser.parse_args()

    print("\nAngular-flux profiles ψ_n(r) — spherical LD-S_N (Section 10.8)")
    print(f"  I_exact(r,μ) = {args.m}·r·μ + {args.b}")
    print(f"  σ = {args.sigma} cm⁻¹,  R = {args.R} cm")
    print(f"  mesh I = {args.I},  order N = {args.N}")

    plot_psi_profiles(args.I, args.N, args.R, args.sigma, args.m, args.b,
                      args.out)

    print("\nDone.")


if __name__ == '__main__':
    main()
