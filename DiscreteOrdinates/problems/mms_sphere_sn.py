#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Manufactured Solution (MMS) verification for the 1-D spherical LD-S_N solver.

Section 10.8 of the textbook.

Problem
-------
The 1-D spherical steady-state radiative transfer equation (C_v → ∞, T → 0):

    μ ∂I/∂r  +  (1 − μ²)/r · ∂I/∂μ  +  σ I(r,μ) = Q(r,μ)

Manufactured solution:
    I_exact(r, μ) = m r μ + b

Derivation of Q:
    μ ∂I/∂r  = μ · mμ         = m μ²
    (1−μ²)/r · ∂I/∂μ = (1−μ²)/r · mr = m(1−μ²)
    Streaming sum            = m μ² + m(1−μ²) = m

    Q(r,μ) = m + σ(mr μ + b)           = m + σ I_exact



No scattering (σ_s = 0), so the problem is direct (one sweep suffices).

Run
---
    cd DiscreteOrdinates
    python problems/mms_sphere_sn.py              # default: N=8, I=64, R=3
    python problems/mms_sphere_sn.py --N 4 8 16 --I 16 32 64 128 --R 3
    python problems/mms_sphere_sn.py --no-convergence   # quick single-run plot
"""

import sys
import os
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
import warnings

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_sn_dir   = os.path.dirname(_this_dir)        # DiscreteOrdinates/
sys.path.insert(0, _sn_dir)

from DiscreteOrdinates.src.sn_solver_ld_sphere import (
    single_sweep_psi_sph_ld,
    single_sweep_phi_sph_ld,
)
from DiscreteOrdinates.src.sn_solver import _get_quadrature


# ---------------------------------------------------------------------------
# Analytic solution
# ---------------------------------------------------------------------------

def I_exact(r, mu, m, b):
    """Manufactured intensity I(r, μ) = m r μ + b."""
    return m * r * mu + b


def Q_mms(r, mu, sigma, m, b):
    """Manufactured source Q(r, μ) = m + σ(mr μ + b).

    Derived from the spherical streaming operator applied to I_exact:
        L[I_exact] = m  (the μ² and (1−μ²) terms combine to 1)
    so  Q = L[I_exact] + σ I_exact = m + σ(mr μ + b).
    """
    return m + sigma * (m * r * mu + b)


# ---------------------------------------------------------------------------
# Single-mesh solve
# ---------------------------------------------------------------------------

def run_mms(I, N, R, sigma, m, b, fix=1):
    """Solve the MMS problem and return (psi_num, psi_exact, r_edges, MU, errors).

    Parameters
    ----------
    I : int       Number of radial cells.
    N : int       Number of discrete ordinates (S_N order).
    R : float     Outer radius (cm).
    sigma : float Absorption opacity (cm⁻¹).
    m, b : float  Manufactured-solution parameters.
    fix : int     Positivity fixup flag.

    Returns
    -------
    psi_num   : (I, N, 2)  Numerical angular flux.
    psi_exact : (I, N, 2)  Exact angular flux evaluated at cell edges.
    r_left    : (I,)       Left-edge radii.
    dr        : (I,)       Cell widths.
    MU        : (N,)       Gauss-Legendre ordinates.
    phi_num   : (I, 2)     Numerical scalar flux.
    phi_exact : (I, 2)     Exact scalar flux  ∫ I_exact dμ.
    """
    MU, W = _get_quadrature(N)

    dr     = np.full(I, R / I, dtype=np.float64)
    r_left = np.arange(I, dtype=np.float64) * (R / I)   # r_{j-1/2}
    r_right = r_left + dr                                 # r_{j+1/2}

    # sigma_hat for steady state: no 1/(c dt) term
    sigma_arr = np.full((I, 2), sigma, dtype=np.float64)

    # -----------------------------------------------------------------
    # Build anisotropic source arrays  Q(r_{j,edge}, μ_n)
    # source_n[j, n, 0] = Q at left  edge of cell j for ordinate n
    # source_n[j, n, 1] = Q at right edge of cell j for ordinate n
    # -----------------------------------------------------------------
    source_n = np.empty((I, N, 2), dtype=np.float64)
    for n in range(N):
        source_n[:, n, 0] = Q_mms(r_left,  MU[n], sigma, m, b)
        source_n[:, n, 1] = Q_mms(r_right, MU[n], sigma, m, b)

    # source_g is the source for the starting direction (μ = −1)
    source_g = np.empty((I, 2), dtype=np.float64)
    source_g[:, 0] = Q_mms(r_left,  -1.0, sigma, m, b)
    source_g[:, 1] = Q_mms(r_right, -1.0, sigma, m, b)

    # -----------------------------------------------------------------
    # Boundary conditions: incoming intensities from the exact solution
    # -----------------------------------------------------------------
    # Outer wall (r = R): inward-moving ordinates (μ_n < 0) and g (μ = −1)
    bc_outer    = np.zeros((N, 2), dtype=np.float64)
    bc_g_outer  = float(I_exact(R, -1.0, m, b))
    for n in range(N):
        if MU[n] < 0.0:
            bc_outer[n, 0] = float(I_exact(R, MU[n], m, b))

    # Inner boundary: full sphere (r_left[0] = 0), so origin regularity
    # condition I_n(0) = g(0) is applied automatically by the kernel.

    # -----------------------------------------------------------------
    # Single sweep (no scattering → one sweep = exact solve)
    # -----------------------------------------------------------------
    psi_num, phi_num, g_num = single_sweep_psi_sph_ld(
        I, r_left, dr,
        source_n, source_g,
        sigma_arr, N,
        bc_outer, bc_g_outer,
        bc_inner=None,   # full sphere
        fix=fix,
    )

    # -----------------------------------------------------------------
    # Exact solution at cell edges
    # -----------------------------------------------------------------
    psi_exact = np.empty((I, N, 2), dtype=np.float64)
    for n in range(N):
        psi_exact[:, n, 0] = I_exact(r_left,  MU[n], m, b)
        psi_exact[:, n, 1] = I_exact(r_right, MU[n], m, b)

    # Exact scalar flux:  φ(r) = ∫_{-1}^{1} I dμ = ∫_{-1}^{1} (mrμ+b) dμ = 2b
    # (the mrμ term integrates to zero; W sums to 1 so φ = 2·sum(W)·b = 2b)
    phi_exact = np.full((I, 2), 2.0 * b, dtype=np.float64)

    return psi_num, psi_exact, r_left, dr, MU, phi_num, phi_exact


# ---------------------------------------------------------------------------
# Convergence study
# ---------------------------------------------------------------------------

def convergence_study(N_vals, I_vals, R, sigma, m, b):
    """Compute L∞ and L² errors of psi over a grid of (N, I) values."""
    results = {}
    header = f"  {'N':>4}  {'I':>6}  {'L∞(ψ)':>12}  {'L²(ψ)':>12}  {'L∞(φ)':>12}"
    print(header)
    print("  " + "-" * (len(header) - 2))

    for N in N_vals:
        results[N] = {'I': [], 'Linf_psi': [], 'L2_psi': [], 'Linf_phi': []}
        for I in I_vals:
            psi_num, psi_exact, _, dr, _, phi_num, phi_exact = \
                run_mms(I, N, R, sigma, m, b)

            diff_psi = np.abs(psi_num - psi_exact)
            scale    = np.maximum(np.abs(psi_exact), 1e-14)
            rel_psi  = diff_psi / scale

            diff_phi = np.abs(phi_num - phi_exact)
            scale_ph = np.maximum(np.abs(phi_exact), 1e-14)
            rel_phi  = diff_phi / scale_ph

            Linf_psi = float(np.max(rel_psi))
            L2_psi   = float(np.sqrt(np.mean(rel_psi**2)))
            Linf_phi = float(np.max(rel_phi))

            results[N]['I'].append(I)
            results[N]['Linf_psi'].append(Linf_psi)
            results[N]['L2_psi'].append(L2_psi)
            results[N]['Linf_phi'].append(Linf_phi)
            print(f"  {N:>4}  {I:>6}  {Linf_psi:>12.3e}  {L2_psi:>12.3e}  {Linf_phi:>12.3e}")

    return results


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _setup_style():
    matplotlib.rcParams.update({
        'font.size': 12,
        'axes.spines.top':   False,
        'axes.spines.right': False,
    })


def plot_angular_flux(psi_num, psi_exact, r_left, dr, MU, N,
                      m, b, sigma, R, save_prefix='mms_sph'):
    """Plot I_n(r) numerical vs analytic for each ordinate on one figure.

    Each cell contributes two points (left and right LD DOFs), plotted as
    connected segments.  Discontinuities between cells are shown by the gaps
    when the left DOF of one cell differs from the right DOF of the previous.
    """
    _setup_style()
    r_right = r_left + dr

    cmap   = get_cmap('tab20')
    colors = [plt.cm.tab20(i / N) for i in range(N)]

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    ax_neg, ax_pos = axes

    for n in range(N):
        mu_n  = MU[n]
        color = colors[n]
        label = fr'$\mu_{n+1} = {mu_n:+.4f}$'

        # Interleave left/right DOFs: r = [rL_0, rR_0, rL_1, rR_1, ...]
        # Insert NaN between cells to break the line at discontinuities.
        I = len(r_left)
        r_plot     = np.empty(3 * I)
        I_plot_num = np.empty(3 * I)
        I_plot_ex  = np.empty(3 * I)
        r_plot[0::3]     = r_left
        r_plot[1::3]     = r_right
        r_plot[2::3]     = np.nan
        I_plot_num[0::3] = psi_num[:, n, 0]
        I_plot_num[1::3] = psi_num[:, n, 1]
        I_plot_num[2::3] = np.nan
        I_plot_ex[0::3]  = psi_exact[:, n, 0]
        I_plot_ex[1::3]  = psi_exact[:, n, 1]
        I_plot_ex[2::3]  = np.nan

        ax = ax_neg if mu_n < 0 else ax_pos
        ax.plot(r_plot, I_plot_ex, '-', color=color, lw=2, label=label)
        ax.plot(r_plot, I_plot_num, 'o', color=color, ms=3, alpha=0.6)

    for ax, title_suffix in zip(axes, ['negative ordinates (μ < 0)',
                                        'positive ordinates (μ > 0)']):
        ax.set_xlabel(r'$r$ (cm)')
        ax.set_ylabel(r'$I_n(r)$')
        ax.set_title(fr'$I(r,\mu)$ — {title_suffix}' + '\n'
                     fr'(lines = analytic, dots = numerical, $N={N}$)')
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.25)
        for spine in ('bottom', 'left'):
            ax.spines[spine].set_linewidth(1.5)

    fig.suptitle(
        fr'MMS spherical S$_N$ — $I_{{exact}}(r,\mu) = {m}\,r\mu + {b}$, '
        fr'$\sigma = {sigma}$ cm$^{{-1}}$, $R = {R}$ cm, $N = {N}$',
        fontsize=11,
    )
    fig.tight_layout()

    fname = f'{save_prefix}_N{N}_angular_flux'
    fig.savefig(fname + '.png', dpi=160, bbox_inches='tight')
    fig.savefig(fname + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved  {fname}.png  /  .pdf")


def plot_scalar_flux(phi_num, phi_exact, r_left, dr,
                     m, b, sigma, R, N, save_prefix='mms_sph'):
    """Plot scalar flux φ(r) and the relative error."""
    _setup_style()
    r_cen = r_left + 0.5 * dr
    phi_c_num = 0.5 * (phi_num[:, 0] + phi_num[:, 1])
    phi_c_ex  = 0.5 * (phi_exact[:, 0] + phi_exact[:, 1])
    rel_err   = np.abs(phi_c_num - phi_c_ex) / (np.abs(phi_c_ex) + 1e-30)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    ax1.plot(r_cen, phi_c_ex,  '-k', lw=2, label='Exact')
    ax1.plot(r_cen, phi_c_num, 'o',  ms=3, alpha=0.7, label='Numerical')
    ax1.set_xlabel(r'$r$ (cm)')
    ax1.set_ylabel(r'$\varphi(r)$')
    ax1.set_title(fr'Scalar flux ($N={N}$, $I={len(r_left)}$)')
    ax1.legend()
    ax1.grid(True, alpha=0.25)

    ax2.semilogy(r_cen, np.maximum(rel_err, 1e-17), '-b', lw=1.5)
    ax2.set_xlabel(r'$r$ (cm)')
    ax2.set_ylabel(r'relative error $|\varphi - \varphi_{ex}| / |\varphi_{ex}|$')
    ax2.set_title('Scalar flux relative error')
    ax2.grid(True, alpha=0.25)

    for ax in (ax1, ax2):
        for spine in ('bottom', 'left'):
            ax.spines[spine].set_linewidth(1.5)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle(
        fr'MMS — $I_{{exact}} = {m}\,r\mu + {b}$, $\sigma = {sigma}$ cm$^{{-1}}$, $R = {R}$ cm',
        fontsize=11,
    )
    fig.tight_layout()

    fname = f'{save_prefix}_N{N}_scalar_flux'
    fig.savefig(fname + '.png', dpi=160, bbox_inches='tight')
    fig.savefig(fname + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved  {fname}.png  /  .pdf")


def plot_convergence(results, N_vals, I_vals, save_prefix='mms_sph'):
    """Log-log convergence plot of L∞ψ vs mesh size h = R/I.

    Note: For the linear-in-μ manufactured solution the dominant error is
    angular discretisation (SN approximation of the (1-μ²)/r ∂I/∂μ term),
    not spatial.  The errors are therefore roughly constant as I increases at
    fixed N.  To see spatial O(h²) convergence, use m=0 (isotropic solution).
    """
    _setup_style()
    colors = [plt.cm.tab10(i / len(N_vals)) for i in range(len(N_vals))]

    fig, ax = plt.subplots(figsize=(7, 5))

    for ni, N in enumerate(N_vals):
        res = results[N]
        hs  = 1.0 / np.array(res['I'], dtype=float)   # h = R/I (R=1 for normalised)
        ax.loglog(hs, res['Linf_psi'], '-o', color=colors[ni],
                  label=fr'$N = {N}$', lw=1.5, ms=5)

    # Reference lines for 1st and 2nd order
    hs_ref = np.array([1.0 / I_vals[0], 1.0 / I_vals[-1]])
    ax.loglog(hs_ref, 2e-1 * hs_ref,      '--k', lw=1, alpha=0.5, label=r'$\mathcal{O}(h)$')
    ax.loglog(hs_ref, 2e-1 * hs_ref**2,   ':k',  lw=1, alpha=0.5, label=r'$\mathcal{O}(h^2)$')

    ax.set_xlabel(r'$h = R/I$')
    ax.set_ylabel(r'$L^\infty$ relative error in $\psi$')
    ax.set_title('MMS convergence — spherical LD-S$_N$')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_linewidth(1.5)

    fig.tight_layout()
    fname = f'{save_prefix}_convergence'
    fig.savefig(fname + '.png', dpi=160, bbox_inches='tight')
    fig.savefig(fname + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved  {fname}.png  /  .pdf")
    """L∞ and L² error vs N at a fixed fine spatial mesh I=I_fine."""

def plot_N_convergence(N_vals, I_fine, R, sigma, m, b, save_prefix='mms_sph'):
    """Error vs SN order N at a fixed fine spatial mesh, showing angular convergence."""
    _setup_style()
    Linf_vals = []
    L2_vals   = []
    print(f"\n--- N-convergence (I={I_fine} fixed) ---")
    print(f"  {'N':>4}  {'L∞(ψ)':>12}  {'L²(ψ)':>12}")
    for N in N_vals:
        psi_num, psi_exact, _, _, _, _, _ = run_mms(I_fine, N, R, sigma, m, b)
        diff   = np.abs(psi_num - psi_exact)
        scale  = np.maximum(np.abs(psi_exact), 1e-14)
        rel    = diff / scale
        Linf   = float(np.max(rel))
        L2     = float(np.sqrt(np.mean(rel**2)))
        Linf_vals.append(Linf)
        L2_vals.append(L2)
        print(f"  {N:>4}  {Linf:>12.3e}  {L2:>12.3e}")

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(N_vals, Linf_vals, 'o-b', lw=1.5, ms=6, label=r'$L^\infty$')
    ax.loglog(N_vals, L2_vals,   's--r', lw=1.5, ms=6, label=r'$L^2$')
    ax.set_xlabel('$N$ (SN order)')
    ax.set_ylabel('Relative error in $\\psi$')
    ax.set_title(fr'Angular discretisation error vs $N$ ($I={I_fine}$, $R={R}$)')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_linewidth(1.5)
    fig.tight_layout()
    fname = f'{save_prefix}_N_convergence'
    fig.savefig(fname + '.png', dpi=160, bbox_inches='tight')
    fig.savefig(fname + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved  {fname}.png  /  .pdf")

# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='MMS verification for 1-D spherical LD-SN solver (Section 10.8)')
    parser.add_argument('--N',  type=int, nargs='+', default=[4, 8, 16],
                        help='S_N orders to test (default: 4 8 16)')
    parser.add_argument('--I',  type=int, nargs='+', default=[16, 32, 64, 128],
                        help='Number of cells for convergence study (default: 16 32 64 128)')
    parser.add_argument('--R',  type=float, default=3.0,
                        help='Outer radius in cm (default: 3.0)')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Absorption opacity σ (cm⁻¹, default: 1.0)')
    parser.add_argument('--m',  type=float, default=0.5,
                        help='Slope parameter m for I = mrμ+b (default: 0.5)')
    parser.add_argument('--b',  type=float, default=2.0,
                        help='Offset parameter b for I = mrμ+b (default: 2.0)')
    parser.add_argument('--plot-N', type=int, default=None,
                        help='S_N order to use for the angular-flux detail plot. '
                             'Defaults to the last value in --N.')
    parser.add_argument('--plot-I', type=int, default=64,
                        help='Number of cells for the detail plots (default: 64)')
    parser.add_argument('--no-convergence', action='store_true',
                        help='Skip the convergence study; only produce detail plots.')
    parser.add_argument('--prefix', default='mms_sph',
                        help='Output file prefix (default: mms_sph)')
    parser.add_argument('--fix', type=int, default=1,
                        help='Positivity fixup flag 0/1 (default: 1)')
    args = parser.parse_args()

    R     = args.R
    sigma = args.sigma
    m     = args.m
    b     = args.b
    fix   = args.fix
    N_plot = args.plot_N if args.plot_N is not None else args.N[-1]
    I_plot = args.plot_I

    # Validate that I_exact >= 0 everywhere in the domain (required for fix=1)
    I_min_check = I_exact(R, -1.0, m, b)     # most negative: r=R, μ=−1
    if I_min_check < 0.0:
        print(f"WARNING: I_exact(R,−1) = {I_min_check:.3f} < 0; "
              f"the positivity fixup will alter the solution.  "
              f"Consider increasing b so that b > m·R.")

    print(f"\nMMS spherical S_N  (Section 10.8)")
    print(f"  I_exact(r,μ) = {m}·r·μ + {b}")
    print(f"  Q(r,μ)       = {m} + {sigma}·({m}·r·μ + {b})   [eq 10.223]")
    print(f"  σ = {sigma} cm⁻¹,  R = {R} cm")

    # ── Detail plots at a single (N, I) ──────────────────────────────────────
    print(f"\n--- Detail plot: N={N_plot}, I={I_plot} ---")
    psi_num, psi_exact, r_left, dr, MU, phi_num, phi_exact = \
        run_mms(I_plot, N_plot, R, sigma, m, b, fix=fix)

    # Print per-ordinate L∞ errors
    print(f"  {'n':>3}  {'μ_n':>9}  {'L∞|ψ_n − ψ_ex|':>18}  {'max|ψ_n|':>12}")
    for n in range(N_plot):
        err_n = float(np.max(np.abs(psi_num[:, n, :] - psi_exact[:, n, :])))
        mag_n = float(np.max(np.abs(psi_exact[:, n, :])))
        print(f"  {n:>3}  {MU[n]:>+9.6f}  {err_n:>18.3e}  {mag_n:>12.3e}")

    plot_angular_flux(psi_num, psi_exact, r_left, dr, MU, N_plot,
                      m, b, sigma, R, save_prefix=args.prefix)
    plot_scalar_flux(phi_num, phi_exact, r_left, dr,
                     m, b, sigma, R, N_plot, save_prefix=args.prefix)

    # ── Convergence study ─────────────────────────────────────────────────────
    if not args.no_convergence:
        print(f"\n--- Convergence study: N ∈ {args.N}, I ∈ {args.I} ---")
        results = convergence_study(args.N, args.I, R, sigma, m, b)
        plot_convergence(results, args.N, args.I, save_prefix=args.prefix)
        N_fine_vals = sorted(set(args.N + [2, 4, 8, 16, 32]))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot_N_convergence(N_fine_vals, args.plot_I, R, sigma, m, b,
                               save_prefix=args.prefix)

    print("\nDone.")


if __name__ == '__main__':
    main()
