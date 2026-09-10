#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Angular (S_N) convergence study for the 1-D spherical LD-S_N method.

Companion to ``mms_sphere_sn.py`` (Section 10.8).  Where that script verifies
the solver against the manufactured solution I(r,μ)=m r μ + b, this script
*quantifies and visualises* the angular discretisation error that remains
after the spatial and source terms are exact.

Two distinct effects are documented, both of which are worth discussing in the
text:

1.  CLOSURE CONSISTENCY ERROR  —  O(1/N²).
    The conservative spherical S_N scheme replaces the true angular metric
    (1−μ²) at each angular edge μ_{n+1/2} by a discrete coefficient α_{n+1/2}
    built from the recursion

        α_{n+1/2} = α_{n−1/2} − 4 μ_n w_n ,   α_{1/2} = α_{N+1/2} = 0 .

    This recursion is exactly the *constant-preservation* condition (it
    telescopes to zero at μ = ±1).  Enforcing it forces α_{n+1/2} ≠ 1−μ²_{n+1/2};
    the defect decays only as O(Δμ²) = O(1/N²), NOT spectrally, even though the
    underlying Gauss quadrature is spectrally accurate.  This sets the angular
    floor of the scalar flux φ(r), independent of spatial refinement.

2.  POLE / ORIGIN AMPLIFICATION  —  grows with N in L∞.
    The worst-case angular-flux error always lives at the most-forward
    ordinate (μ_n → +1) at the origin (r = 0), where (1−μ²)/r is most
    singular.  As N grows μ_max → 1 and this localized error *increases*,
    even as the mean (scalar) solution keeps improving.  This is a less
    well-known limitation of refining the S_N order for curved geometry.

Run
---
    cd DiscreteOrdinates
    python problems/mms_sphere_angular_convergence.py
    python problems/mms_sphere_angular_convergence.py --N 4 8 16 32 64 128 --I 256
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
sys.path.insert(0, _this_dir)

from DiscreteOrdinates.src.sn_solver import _get_quadrature
from DiscreteOrdinates.src.sn_solver_ld_sphere import _compute_sph_quad_data
from mms_sphere_sn import run_mms


# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

def _setup_style():
    matplotlib.rcParams.update({
        'font.size': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def _finish_axes(ax):
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_linewidth(1.5)
    ax.grid(True, which='both', alpha=0.25)


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def metric_defect(N):
    """Return (Linf, L2) of |α_{n+1/2} − (1−μ²_{n+1/2})| over interior edges."""
    mu_edges, alpha, _ = _compute_sph_quad_data(N)
    true = 1.0 - mu_edges ** 2
    defect = (alpha - true)[1:-1]            # interior edges only
    return float(np.max(np.abs(defect))), float(np.sqrt(np.mean(defect ** 2)))


def collect(N_vals, I_fine, R, sigma, m, b):
    """Gather metric defect and MMS errors over a list of S_N orders."""
    data = {k: [] for k in (
        'N', 'metric_Linf', 'metric_L2',
        'psi_Linf', 'psi_L2', 'phi_L2',
        'argmax_mu', 'argmax_r')}

    for N in N_vals:
        mLinf, mL2 = metric_defect(N)

        psi_num, psi_exact, r_left, dr, MU, phi_num, phi_exact = \
            run_mms(I_fine, N, R, sigma, m, b)

        rel = np.abs(psi_num - psi_exact) / np.maximum(np.abs(psi_exact), 1e-14)
        relph = np.abs(phi_num - phi_exact) / np.maximum(np.abs(phi_exact), 1e-14)

        idx = np.unravel_index(int(np.argmax(rel)), rel.shape)
        j, n, lr = idx
        r_here = r_left[j] + (dr[j] if lr == 1 else 0.0)

        data['N'].append(N)
        data['metric_Linf'].append(mLinf)
        data['metric_L2'].append(mL2)
        data['psi_Linf'].append(float(np.max(rel)))
        data['psi_L2'].append(float(np.sqrt(np.mean(rel ** 2))))
        data['phi_L2'].append(float(np.sqrt(np.mean(relph ** 2))))
        data['argmax_mu'].append(float(MU[n]))
        data['argmax_r'].append(float(r_here))

    for k in data:
        data[k] = np.array(data[k], dtype=float)
    return data


def fit_slope(N, err, tail_from=None):
    """Least-squares power-law slope p in err ~ N^p (optionally tail only)."""
    mask = np.ones_like(N, dtype=bool) if tail_from is None else (N >= tail_from)
    if mask.sum() < 2:
        return np.nan
    p = np.polyfit(np.log(N[mask]), np.log(err[mask]), 1)
    return float(p[0])


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_metric_defect(data, save_prefix):
    """Metric defect |α − (1−μ²)| vs N with O(1/N²) reference."""
    _setup_style()
    N = data['N']
    fig, ax = plt.subplots(figsize=(6.5, 5))

    ax.loglog(N, data['metric_Linf'], 'o-', color='C0', lw=1.6, ms=6,
              label=r'$\max_n|\alpha_{n+1/2}-(1-\mu_{n+1/2}^2)|$')
    ax.loglog(N, data['metric_L2'], 's--', color='C1', lw=1.6, ms=6,
              label=r'$L^2$ defect')

    ref = data['metric_Linf'][0] * (N / N[0]) ** (-2.0)
    ax.loglog(N, ref, ':k', lw=1.2, alpha=0.6, label=r'$\mathcal{O}(N^{-2})$')

    p_inf = fit_slope(N, data['metric_Linf'])
    p_2 = fit_slope(N, data['metric_L2'])
    ax.set_xlabel(r'$N$ (S$_N$ order)')
    ax.set_ylabel(r'angular-metric defect')
    ax.set_title('Closure consistency error\n'
                 fr'(fitted slopes: $L^\infty\sim N^{{{p_inf:.2f}}}$, '
                 fr'$L^2\sim N^{{{p_2:.2f}}}$)')
    ax.legend(fontsize=10)
    _finish_axes(ax)
    fig.tight_layout()
    _save(fig, f'{save_prefix}_metric_defect')


def plot_phi_convergence(data, save_prefix):
    """Scalar-flux L2 error vs N (the smooth, well-behaved quantity)."""
    _setup_style()
    N = data['N']
    fig, ax = plt.subplots(figsize=(6.5, 5))

    ax.loglog(N, data['phi_L2'], 'o-', color='C2', lw=1.6, ms=6,
              label=r'$L^2$ error in $\varphi(r)$')

    ref2 = data['phi_L2'][0] * (N / N[0]) ** (-2.0)
    ref1 = data['phi_L2'][0] * (N / N[0]) ** (-1.0)
    ax.loglog(N, ref2, ':k', lw=1.2, alpha=0.6, label=r'$\mathcal{O}(N^{-2})$')
    ax.loglog(N, ref1, '--k', lw=1.0, alpha=0.4, label=r'$\mathcal{O}(N^{-1})$')

    p = fit_slope(N, data['phi_L2'])
    ax.set_xlabel(r'$N$ (S$_N$ order)')
    ax.set_ylabel(r'$L^2$ relative error in $\varphi$')
    ax.set_title('Scalar-flux angular convergence\n'
                 fr'(fitted slope $\sim N^{{{p:.2f}}}$; floors on spatial error)')
    ax.legend(fontsize=10)
    _finish_axes(ax)
    fig.tight_layout()
    _save(fig, f'{save_prefix}_phi_convergence')


def plot_pole_amplification(data, save_prefix):
    """L∞(ψ) growth and its location (μ_n → 1 at r = 0)."""
    _setup_style()
    N = data['N']
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: psi Linf (grows) vs phi L2 (decays) on the same axes
    ax1.loglog(N, data['psi_Linf'], 'o-', color='C3', lw=1.6, ms=6,
               label=r'$L^\infty$ error in $\psi$ (worst ordinate)')
    ax1.loglog(N, data['phi_L2'], 's--', color='C2', lw=1.6, ms=6,
               label=r'$L^2$ error in $\varphi$ (mean)')
    ax1.set_xlabel(r'$N$ (S$_N$ order)')
    ax1.set_ylabel('relative error')
    ax1.set_title('Pole amplification vs. mean convergence')
    ax1.legend(fontsize=10)
    _finish_axes(ax1)

    # Right: the ordinate where the worst error lives, μ_n → 1
    ax2.semilogx(N, data['argmax_mu'], 'o-', color='C3', lw=1.6, ms=6)
    ax2.axhline(1.0, color='k', ls=':', lw=1, alpha=0.6)
    ax2.set_xlabel(r'$N$ (S$_N$ order)')
    ax2.set_ylabel(r'$\mu_n$ of worst-error ordinate')
    ax2.set_ylim(0.0, 1.05)
    ax2.set_title('Worst error localizes at the forward pole\n'
                  r'($\mu_n \to +1$, always at $r = 0$)')
    _finish_axes(ax2)

    fig.tight_layout()
    _save(fig, f'{save_prefix}_pole_amplification')


def _save(fig, name):
    fig.savefig(name + '.png', dpi=160, bbox_inches='tight')
    fig.savefig(name + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved  {name}.png  /  .pdf")


# ---------------------------------------------------------------------------
# Table
# ---------------------------------------------------------------------------

def print_table(data):
    print(f"\n  {'N':>5}  {'metricLinf':>11}  {'metricL2':>11}  "
          f"{'psiLinf':>10}  {'psiL2':>10}  {'phiL2':>10}  "
          f"{'argmax μ':>9}  {'r':>6}")
    print("  " + "-" * 92)
    for i in range(len(data['N'])):
        print(f"  {int(data['N'][i]):>5}  "
              f"{data['metric_Linf'][i]:>11.3e}  {data['metric_L2'][i]:>11.3e}  "
              f"{data['psi_Linf'][i]:>10.3e}  {data['psi_L2'][i]:>10.3e}  "
              f"{data['phi_L2'][i]:>10.3e}  "
              f"{data['argmax_mu'][i]:>+9.5f}  {data['argmax_r'][i]:>6.3f}")

    N = data['N']
    print("\n  Power-law slopes  err ~ N^p  (all N | tail N≥16):")
    for key, label in [('metric_Linf', 'metric L∞'),
                       ('metric_L2', 'metric L²'),
                       ('phi_L2', 'φ  L²'),
                       ('psi_L2', 'ψ  L²'),
                       ('psi_Linf', 'ψ  L∞')]:
        p_all = fit_slope(N, data[key])
        p_tail = fit_slope(N, data[key], tail_from=16)
        print(f"    {label:10s}  p = {p_all:+.3f}  |  {p_tail:+.3f}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Angular S_N convergence study for the spherical LD-SN '
                    'method (companion to mms_sphere_sn.py).')
    parser.add_argument('--N', type=int, nargs='+',
                        default=[4, 8, 16, 32, 64, 128],
                        help='S_N orders to test (default: 4 8 16 32 64 128)')
    parser.add_argument('--I', type=int, default=256,
                        help='Fixed (fine) spatial mesh so spatial error is '
                             'negligible (default: 256)')
    parser.add_argument('--R', type=float, default=3.0, help='Outer radius (cm)')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Absorption opacity σ (cm⁻¹)')
    parser.add_argument('--m', type=float, default=0.5,
                        help='Slope m for I = m r μ + b')
    parser.add_argument('--b', type=float, default=2.0,
                        help='Offset b for I = m r μ + b')
    parser.add_argument('--prefix', default='mms_sph_angular',
                        help='Output file prefix')
    args = parser.parse_args()

    print("\nAngular S_N convergence study — spherical LD-S_N (Section 10.8)")
    print(f"  I_exact(r,μ) = {args.m}·r·μ + {args.b}")
    print(f"  σ = {args.sigma} cm⁻¹,  R = {args.R} cm,  fixed mesh I = {args.I}")

    data = collect(args.N, args.I, args.R, args.sigma, args.m, args.b)
    print_table(data)

    print("\n--- Figures ---")
    plot_metric_defect(data, args.prefix)
    plot_phi_convergence(data, args.prefix)
    plot_pole_amplification(data, args.prefix)

    print("\nDone.")


if __name__ == '__main__':
    main()
