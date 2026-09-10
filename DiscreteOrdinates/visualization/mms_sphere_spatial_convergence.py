#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Spatial (O(h^2)) convergence study for the 1-D spherical LD-S_N method.

Companion to ``mms_sphere_sn.py`` and ``mms_sphere_angular_convergence.py``
(Section 10.8).

Motivation
----------
The linear-in-mu manufactured solution I = m r mu + b is a clean spatial test
ONLY at N = 2: for N >= 4 the conservative angular closure introduces an
O(1/N^2) error floor that masks the spatial discretisation error.  To verify
the *spatial* order of accuracy independently of the angular scheme we choose
an ANGULARLY ISOTROPIC manufactured solution that is also spatially smooth and
non-linear in r:

    I_exact(r) = b + A * (1 - (r/R)^2)          (independent of mu)

Because the solution is isotropic, the angular redistribution term
(1 - mu^2)/r dI/dmu vanishes identically, and the discrete S_N closure
represents it without consistency error at *every* N.  The only remaining
error is therefore the spatial LD discretisation, which converges at its
formal second order.

Manufactured source (from the steady transport equation, dI/dmu = 0):

    Q(r, mu) = mu * I'(r) + sigma * I(r),      I'(r) = -2 A r / R^2

Result
------
Both the L-infinity and L2 errors converge as O(h^2) at all N, with NO angular
floor as N is increased — the complementary picture to
``mms_sphere_angular_convergence.py``.

Run
---
    cd DiscreteOrdinates
    python problems/mms_sphere_spatial_convergence.py
    python problems/mms_sphere_spatial_convergence.py --N 4 8 16 --I 16 32 64 128 256 512
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_sn_dir = os.path.dirname(_this_dir)        # DiscreteOrdinates/
sys.path.insert(0, _sn_dir)

from DiscreteOrdinates.src.sn_solver import _get_quadrature
from DiscreteOrdinates.src.sn_solver_ld_sphere import single_sweep_psi_sph_ld


# ---------------------------------------------------------------------------
# Manufactured solution (isotropic, spatially non-linear)
# ---------------------------------------------------------------------------

def I_exact(r, A, b, R):
    """Isotropic manufactured intensity I(r) = b + A(1 - (r/R)^2)."""
    return b + A * (1.0 - (r / R) ** 2)


def dI_dr(r, A, R):
    """Radial derivative I'(r) = -2 A r / R^2."""
    return -2.0 * A * r / R ** 2


def Q_mms(r, mu, sigma, A, b, R):
    """Manufactured source Q(r, mu) = mu I'(r) + sigma I(r)."""
    return mu * dI_dr(r, A, R) + sigma * I_exact(r, A, b, R)


# ---------------------------------------------------------------------------
# Single-mesh solve
# ---------------------------------------------------------------------------

def run_mms(I, N, R, sigma, A, b, fix=0):
    """Solve the isotropic MMS problem; return (psi_num, psi_exact)."""
    MU, _ = _get_quadrature(N)
    dr = np.full(I, R / I, dtype=np.float64)
    r_left = np.arange(I, dtype=np.float64) * (R / I)
    r_right = r_left + dr

    sigma_arr = np.full((I, 2), sigma, dtype=np.float64)

    source_n = np.empty((I, N, 2), dtype=np.float64)
    for n in range(N):
        source_n[:, n, 0] = Q_mms(r_left, MU[n], sigma, A, b, R)
        source_n[:, n, 1] = Q_mms(r_right, MU[n], sigma, A, b, R)

    source_g = np.empty((I, 2), dtype=np.float64)
    source_g[:, 0] = Q_mms(r_left, -1.0, sigma, A, b, R)
    source_g[:, 1] = Q_mms(r_right, -1.0, sigma, A, b, R)

    bc_outer = np.zeros((N, 2), dtype=np.float64)
    for n in range(N):
        if MU[n] < 0.0:
            bc_outer[n, 0] = float(I_exact(R, A, b, R))
    bc_g_outer = float(I_exact(R, A, b, R))

    psi_num, _, _ = single_sweep_psi_sph_ld(
        I, r_left, dr, source_n, source_g, sigma_arr, N,
        bc_outer, bc_g_outer, bc_inner=None, fix=fix)

    psi_exact = np.empty((I, N, 2), dtype=np.float64)
    for n in range(N):
        psi_exact[:, n, 0] = I_exact(r_left, A, b, R)
        psi_exact[:, n, 1] = I_exact(r_right, A, b, R)

    return psi_num, psi_exact


# ---------------------------------------------------------------------------
# Convergence data
# ---------------------------------------------------------------------------

def collect(N_vals, I_vals, R, sigma, A, b):
    """Return {N: {'h', 'Linf', 'L2'}} over the mesh sequence."""
    results = {}
    print(f"  {'N':>4}  {'I':>5}  {'h':>10}  {'Linf':>11}  {'L2':>11}  {'rate(L2)':>9}")
    print("  " + "-" * 60)
    for N in N_vals:
        h, Linf, L2 = [], [], []
        prev = None
        for I in I_vals:
            psi_num, psi_exact = run_mms(I, N, R, sigma, A, b)
            rel = np.abs(psi_num - psi_exact) / np.maximum(np.abs(psi_exact), 1e-14)
            li = float(np.max(rel))
            l2 = float(np.sqrt(np.mean(rel ** 2)))
            hh = R / I
            rate = "" if prev is None else f"{np.log2(prev / l2):.2f}"
            print(f"  {N:>4}  {I:>5}  {hh:>10.4e}  {li:>11.3e}  {l2:>11.3e}  {rate:>9}")
            h.append(hh); Linf.append(li); L2.append(l2); prev = l2
        results[N] = {'h': np.array(h), 'Linf': np.array(Linf), 'L2': np.array(L2)}
    return results


def fit_order(h, err):
    """Least-squares order p in err ~ h^p."""
    p = np.polyfit(np.log(h), np.log(err), 1)
    return float(p[0])


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _setup_style():
    matplotlib.rcParams.update({
        'font.size': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_spatial(results, N_vals, save_prefix):
    """Log-log L2 error vs h with an O(h^2) reference line."""
    _setup_style()
    colors = ['tab:blue', 'tab:red', 'tab:green', 'tab:orange', 'tab:purple']
    markers = ['o', 's', '^', 'D', 'v']

    fig, ax = plt.subplots(figsize=(8, 5.2))

    for ni, N in enumerate(N_vals):
        res = results[N]
        p = fit_order(res['h'], res['L2'])
        ax.loglog(res['h'], res['L2'],
                  marker=markers[ni % len(markers)], ls='-',
                  color=colors[ni % len(colors)], lw=1.6, ms=6,
                  label=fr'$N = {N}$  ($p = {p:.2f}$)')

    # -----------------------------------------------------------------
    # Reference triangle: 2nd-order (h^2) decay of the L² error.
    # On log-log axes a power law h^p is a straight line of slope p.
    # (Same right-triangle fiducial as mms_sphere_error_vs_N.py.)
    # -----------------------------------------------------------------
    all_h = results[N_vals[0]]['h']
    x1, x2 = float(np.min(all_h)) * 2.0, float(np.min(all_h)) * 4.0
    all_L2 = np.concatenate([results[N]['L2'] for N in N_vals])
    y1 = float(np.min(all_L2)) * 3                # lower corner (smaller h)
    y2 = y1 * (x2 / x1) ** 2.0                         # upper corner (larger h)
    ax.plot([x1, x2], [y1, y1], '-', color='0.35', lw=1.2)   # horizontal leg
    ax.plot([x2, x2], [y1, y2], '-', color='0.35', lw=1.2)   # vertical leg
    ax.plot([x1, x2], [y1, y2], '-', color='0.35', lw=1.2)   # hypotenuse
    ax.text(np.sqrt(x1 * x2), y1 * 0.92, '1', color='0.35',
            ha='center', va='top', fontsize=10)
    ax.text(x2 * 1.12, np.sqrt(y1 * y2), '2', color='0.35',
            ha='right', va='center', fontsize=10)

    ax.set_xlabel(r'$\Delta r = R/I$')
    ax.set_ylabel(r'$L_2$ relative error in $I$')
    # ax.set_title('Spatial convergence — isotropic MMS\n'
    #              r'$I_{\mathrm{exact}}(r)=b+A\,[1-(r/R)^2]$ '
    #              r'(angular closure exact at all $N$)')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_linewidth(1.5)
    ax.invert_xaxis()
    fig.tight_layout()

    fig.savefig(save_prefix + '.png', dpi=160, bbox_inches='tight')
    fig.savefig(save_prefix + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"\n  Saved  {save_prefix}.png  /  .pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Spatial O(h^2) convergence study for the spherical LD-SN '
                    'method using an isotropic manufactured solution.')
    parser.add_argument('--N', type=int, nargs='+', default=[4, 8, 16],
                        help='S_N orders to test (default: 4 8 16)')
    parser.add_argument('--I', type=int, nargs='+',
                        default=[ 32, 64, 128, 256,512],
                        help='Cell counts for refinement (default: 16..512)')
    parser.add_argument('--R', type=float, default=3.0, help='Outer radius (cm)')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Absorption opacity sigma (cm^-1)')
    parser.add_argument('--A', type=float, default=1.0,
                        help='Amplitude A for I = b + A(1-(r/R)^2)')
    parser.add_argument('--b', type=float, default=2.0,
                        help='Offset b for I = b + A(1-(r/R)^2)')
    parser.add_argument('--prefix', default='mms_sph_spatial',
                        help='Output file prefix')
    args = parser.parse_args()

    print("\nSpatial convergence study — spherical LD-S_N (Section 10.8)")
    print(f"  I_exact(r) = {args.b} + {args.A}*(1 - (r/{args.R})^2)   (isotropic)")
    print(f"  Q(r,mu)    = mu*I'(r) + {args.sigma}*I(r)")
    print(f"  sigma = {args.sigma} cm^-1,  R = {args.R} cm\n")

    results = collect(args.N, args.I, args.R, args.sigma, args.A, args.b)

    print("\n  Fitted spatial orders (L2):")
    for N in args.N:
        p = fit_order(results[N]['h'], results[N]['L2'])
        print(f"    N = {N:>3}:  p = {p:.3f}")

    print("\n--- Figure ---")
    plot_spatial(results, args.N, args.prefix)

    print("\nDone.")


if __name__ == '__main__':
    main()
