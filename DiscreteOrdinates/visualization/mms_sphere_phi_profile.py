#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Scalar flux φ(r) for the spherical LD-S_N manufactured solution.

Companion figure for Section 10.8.  The manufactured intensity
I(r,μ) = m r μ + b has an exactly constant scalar flux,

    φ_exact(r) = ∫_{-1}^{1} (m r μ + b) dμ = 2 b,

because the odd (m r μ) part integrates to zero.  Any departure of the
numerical φ(r) from this flat line is therefore a *direct* picture of the
angular-closure error and where it lives in space.

This script plots φ(r) for S_8 and S_32 at three spatial meshes
(I = 10, 100, 500 radial cells).  It makes two points visible at a glance:

  * The error is concentrated near the origin r → 0, where the curved-geometry
    term (1−μ²)/r is most singular; the bulk r ≳ R/2 sits right on 2b.
  * Refining the spatial mesh sharpens/contains the near-origin feature
    (explaining why the L∞ error depends on the cell count) while the
    large-N angular floor in the bulk is essentially mesh-independent.

Run
---
    cd DiscreteOrdinates
    python problems/mms_sphere_phi_profile.py
    python problems/mms_sphere_phi_profile.py --I 10 100 500 --N 8 32
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
# Build φ(r) on a per-edge mesh
# ---------------------------------------------------------------------------

def phi_profile(I, N, R, sigma, m, b):
    """Return (r, phi_num) sampled at every cell edge (left/right)."""
    _, _, r_left, dr, _, phi_num, _ = run_mms(I, N, R, sigma, m, b)
    # Interleave left/right edges so the discontinuous LD trace is visible.
    r = np.empty(2 * I)
    phi = np.empty(2 * I)
    r[0::2] = r_left
    r[1::2] = r_left + dr
    phi[0::2] = phi_num[:, 0]
    phi[1::2] = phi_num[:, 1]
    return r, phi


# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------

def plot_phi_profile(I_vals, N_vals, R, sigma, m, b, save_name):
    """Two panels (one per S_N order); φ(r) for each spatial mesh."""
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
    markers = ['o', 's', '^', 'D', 'v']
    phi_exact = 2.0 * b

    fig, axes = plt.subplots(1, len(N_vals), figsize=(11, 4.5), sharey=True)
    if len(N_vals) == 1:
        axes = [axes]

    for ax, N in zip(axes, N_vals):
        for k, I in enumerate(I_vals):
            r, phi = phi_profile(I, N, R, sigma, m, b)
            ax.plot(r, phi, '-', marker=markers[k % len(markers)],
                    color=colors[k % len(colors)], lw=1.4,
                    ms=4, markevery=max(1, len(r) // 16),
                    label=fr'$I = {I}$')
        ax.axhline(phi_exact, color='0.35', ls=':', lw=1.3,
                   label=fr'exact $\,\varphi = 2b = {phi_exact:g}$')
        ax.set_xlabel(r'radius $r$ (cm)')
        ax.set_title(fr'$S_{{{N}}}$', fontproperties=font, fontsize=13)
        ax.set_xlim(0.0, R)
        ax.grid(True, which='both', alpha=0.25)

    axes[0].set_ylabel(r'scalar flux $\varphi(r)$')
    axes[0].legend(prop=font, frameon=False, loc='best')

    plt.tight_layout()
    show(save_name, close_after=True)
    print(f"\n  Saved  {save_name}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Scalar-flux profiles φ(r) for the spherical LD-S_N '
                    'manufactured solution (companion to mms_sphere_error_vs_N).')
    parser.add_argument('--I', type=int, nargs='+', default=[10, 100, 500],
                        help='Spatial meshes (radial cells); default 10 100 500')
    parser.add_argument('--N', type=int, nargs='+', default=[8, 32],
                        help='S_N orders to show, one panel each (default 8 32)')
    parser.add_argument('--R', type=float, default=3.0, help='Outer radius (cm)')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Absorption opacity σ (cm⁻¹)')
    parser.add_argument('--m', type=float, default=0.5,
                        help='Slope m for I = m r μ + b')
    parser.add_argument('--b', type=float, default=2.0,
                        help='Offset b for I = m r μ + b')
    parser.add_argument('--out', default='mms_sph_phi_profile.pdf',
                        help='Output figure file name')
    args = parser.parse_args()

    print("\nScalar-flux profiles φ(r) — spherical LD-S_N (Section 10.8)")
    print(f"  I_exact(r,μ) = {args.m}·r·μ + {args.b}")
    print(f"  φ_exact = 2b = {2.0 * args.b:g}")
    print(f"  σ = {args.sigma} cm⁻¹,  R = {args.R} cm")
    print(f"  meshes I = {args.I},  orders N = {args.N}")

    plot_phi_profile(args.I, args.N, args.R, args.sigma, args.m, args.b,
                     args.out)

    print("\nDone.")


if __name__ == '__main__':
    main()
