#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Sourced transparent core inside an opaque absorbing shell — steady,
non-participating spherical transport benchmark.

Geometry (two concentric regions inside a sphere of radius ``R``):

    r < r1   :  sigma = sigma_in  (≈ 0, transparent),  q = q_in  (source)
    r1<r<R   :  sigma = sigma_out (large, opaque),      q = 0
    r = R    :  psi_b = 0          (vacuum / no inflow)

This is a *steady, non-participating* medium: the external isotropic source
``q`` is fixed and the opacity is pure absorption (no scattering, no thermal
emission).  Because nothing couples the radiation field back to the material,
there is no need to time-march a thermal problem with a huge ``Cv`` and a tiny
``T`` to suppress emission — the steady-state transport solve already *is* the
non-participating limit.  We therefore solve the fixed-source steady transport
equation directly:

    mu/r^2 d/dr(r^2 psi_n) + (1/r) d_mu[(1-mu^2) psi]_n + sigma psi_n = q/2 ,

with a (degenerate) source iteration that converges in a single sweep because
sigma_s = 0.

The physics: a nearly transparent, uniformly emitting core radiates into a
very opaque shell.  The scalar flux is large and smooth across the core, then
decays through the shell over roughly one absorption mean-free-path
(1/sigma_out); with an optically thick shell almost nothing leaks out of the
vacuum boundary, so essentially all of the emitted source is absorbed in the
shell.

Run
---
    cd DiscreteOrdinates
    python problems/sphere_shell_source_sn.py
    python problems/sphere_shell_source_sn.py --N 16 --I 400 --R 1.0 --r1 0.4 \
        --sigma_out 100 --sigma_in 1e-8 --q_in 1.0

This reproduces the Type (ii) two-region absorbing problem of
Machorro, J. Comput. Phys. 223 (2007) 67--81 (Section 5.2 / Fig. 5.1 and 6.3):
    rmid = 0.4,  sigma_in = 1e-8,  sigma_out = 100,  q_in = 1,  R = 1,
    vacuum outer boundary psi(R, mu<0) = 0.
The source q is the per-direction right-hand side of Eq. (1.1) (the same value
for every ordinate), and the scalar flux is phi(r) = int_{-1}^{1} psi dmu.  The
exact center value is phi(0) = 2 q_in rmid = 0.8.
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
_sn_dir = os.path.dirname(_this_dir)            # DiscreteOrdinates/
sys.path.insert(0, _sn_dir)

from DiscreteOrdinates.src.sn_solver import _get_quadrature
from DiscreteOrdinates.src.sn_solver_ld_sphere import (
    single_sweep_psi_sph_ld,
    _compute_geometric_moments,
)


# ---------------------------------------------------------------------------
# Two-region mesh (interface snapped to a cell boundary)
# ---------------------------------------------------------------------------

def build_mesh(I, R, r1):
    """Uniform-in-each-region mesh with the interface on a cell boundary.

    Returns (r_left, dr, core_mask) where core_mask[j] is True for cells in
    the transparent core (r < r1).
    """
    frac = r1 / R
    I_in = int(round(I * frac))
    I_in = max(1, min(I - 1, I_in))
    I_out = I - I_in

    dr_in = r1 / I_in
    dr_out = (R - r1) / I_out

    dr = np.concatenate([np.full(I_in, dr_in), np.full(I_out, dr_out)])
    r_left = np.concatenate([[0.0], np.cumsum(dr)[:-1]])
    core_mask = np.arange(I) < I_in
    return r_left, dr, core_mask


# ---------------------------------------------------------------------------
# Steady fixed-source solve (pure absorber: one sweep is exact)
# ---------------------------------------------------------------------------

def solve(I, N, R, r1, sigma_in, sigma_out, q_in, fix=1,
          tol=1e-12, max_its=200):
    """Solve the steady non-participating problem; return diagnostics dict."""
    MU, W = _get_quadrature(N)
    r_left, dr, core_mask = build_mesh(I, R, r1)
    r_right = r_left + dr

    # Per-cell (edge-replicated) opacity and isotropic source.
    sigma_cell = np.where(core_mask, sigma_in, sigma_out)
    q_cell = np.where(core_mask, q_in, 0.0)

    sigma_hat = np.repeat(sigma_cell[:, None], 2, axis=1)        # (I, 2)
    q_iso = np.repeat(q_cell[:, None], 2, axis=1)                # (I, 2)

    # Vacuum outer boundary: no inflow on any incoming ordinate.
    bc_outer = np.zeros((N, 2))
    bc_g_outer = 0.0

    # Pure absorber (sigma_s = 0): the external source does not depend on phi,
    # so source iteration converges in a single sweep.  We still loop for
    # robustness / generality.
    phi = np.zeros((I, 2))
    psi = None
    for it in range(max_its):
        # The source q is the per-direction right-hand side of the transport
        # equation (Machorro 2007, Eq. 1.1): the same value enters every
        # ordinate's balance, so source_n = q (NOT q/2).  The angle-integrated
        # emission is therefore int_{-1}^{1} q dmu = 2 q.
        src = q_iso
        source_n = np.repeat(src[:, None, :], N, axis=1)        # (I, N, 2)
        source_g = src.copy()                                    # (I, 2)

        psi, phi_new, _ = single_sweep_psi_sph_ld(
            I, r_left, dr,
            np.ascontiguousarray(source_n),
            np.ascontiguousarray(source_g),
            np.ascontiguousarray(sigma_hat),
            N, bc_outer, bc_g_outer, bc_inner=None, fix=fix,
        )
        err = np.max(np.abs(phi_new - phi)) / (np.max(np.abs(phi_new)) + 1e-30)
        phi = phi_new
        if err < tol:
            break
    niters = it + 1

    return dict(r_left=r_left, r_right=r_right, dr=dr, core_mask=core_mask,
                phi=phi, psi=psi, MU=MU, W=W, niters=niters,
                sigma_cell=sigma_cell, q_cell=q_cell)


# ---------------------------------------------------------------------------
# Energy balance:  emitted source = absorbed + leaked  (steady state)
# ---------------------------------------------------------------------------

def energy_balance(res, R, q_in):
    """Return (emitted, absorbed, leaked) integrated over the sphere.

    The integrals are evaluated with the LD scheme's *own* r^2-weighted
    geometric moments

        M_l = int_0^1 r(xi)^2 b_L(xi) dxi ,   M_r = int_0^1 r(xi)^2 b_R(xi) dxi

    (the same M_l, M_r that appear in the cell solve), so the discrete steady
    balance ``emitted = absorbed + leaked`` closes to iteration-tolerance on
    ANY mesh -- the spherical LD scheme is conservative by construction.

    Why not a naive shell integral?  Using phi_cell = 1/2 (phi_L + phi_R) with
    the exact shell volume (4 pi/3)(r_R^3 - r_L^3) is NOT what the scheme
    integrates: phi(r) is linear across the cell and the r^2 Jacobian weights
    the outer edge more, so the reaction moment is sigma * dr (M_l phi_L +
    M_r phi_R), not sigma * 1/2 (phi_L + phi_R) * V.  The midpoint form leaves
    an O(dr^2) truncation residual (a few percent on a coarse mesh) that only
    vanishes under refinement -- it is a post-processing inconsistency, not a
    conservation defect of the method.
    """
    r_left = res['r_left']
    dr = res['dr']
    M_l, M_r = _compute_geometric_moments(r_left, dr)
    phi_l = res['phi'][:, 0]
    phi_r = res['phi'][:, 1]

    # Emission: angle-integrated source moment.  q is the per-direction source
    # (constant in mu), so the angular integral contributes a factor 2.
    emitted = 4.0 * np.pi * np.sum(2.0 * res['q_cell'] * dr * (M_l + M_r))
    # Absorption: r^2-weighted LD reaction moment (identical to the cell solve,
    # and preserved exactly by the conservative positivity fixup).
    absorbed = 4.0 * np.pi * np.sum(
        res['sigma_cell'] * dr * (M_l * phi_l + M_r * phi_r))

    # Net outward current at the outer wall:  J = int mu psi dmu = sum 2 w mu psi
    # (weights sum to 1, so Delta mu_n = 2 w_n).  This is exactly what the
    # discrete radial streaming term telescopes to.
    psi_R = res['psi'][-1, :, 1]            # outer face of last cell, each n
    J_R = 2.0 * np.sum(res['W'] * res['MU'] * psi_R)   # net current (per area)
    leaked = 4.0 * np.pi * R ** 2 * J_R
    return emitted, absorbed, leaked


# ---------------------------------------------------------------------------
# Plot
# ---------------------------------------------------------------------------

def _setup_style():
    matplotlib.rcParams.update({
        'font.size': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot(res, R, r1, save_prefix):
    _setup_style()
    r_left = res['r_left']
    r_right = res['r_right']
    # Plot node values: stack left/right edges so the LD profile is visible.
    r_plot = np.empty(2 * len(r_left))
    r_plot[0::2] = r_left
    r_plot[1::2] = r_right
    phi_plot = np.empty(2 * len(r_left))
    phi_plot[0::2] = res['phi'][:, 0]
    phi_plot[1::2] = res['phi'][:, 1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    for ax in (ax1, ax2):
        ax.plot(r_plot, phi_plot, '-', color='tab:blue', lw=1.6)
        ax.axvline(r1, color='0.6', ls='--', lw=1.0, label=fr'interface $r_1={r1:g}$')
        ax.set_xlabel(r'radius $r$')
        ax.set_ylabel(r'scalar flux $\varphi(r)$')
        for spine in ('bottom', 'left'):
            ax.spines[spine].set_linewidth(1.5)
        ax.grid(True, which='both', alpha=0.25)

    ax1.set_title('Linear scale')
    ax1.legend(fontsize=10)
    ax2.set_yscale('log')
    ax2.set_title('Log scale (shell decay)')

    fig.suptitle('Sourced transparent core in an opaque shell — steady, '
                 'non-participating', y=1.02)
    fig.tight_layout()
    fig.savefig(save_prefix + '.png', dpi=160, bbox_inches='tight')
    fig.savefig(save_prefix + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved  {save_prefix}.png  /  .pdf")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Steady non-participating spherical transport: sourced '
                    'transparent core inside an opaque absorbing shell.')
    parser.add_argument('--N', type=int, default=16, help='S_N order (default 16)')
    parser.add_argument('--I', type=int, default=30,
                        help='Total radial cells (default 30)')
    parser.add_argument('--R', type=float, default=1.0, help='Outer radius (default 1.0)')
    parser.add_argument('--r1', type=float, default=0.4,
                        help='Core radius / interface (default 0.4, paper rmid)')
    parser.add_argument('--sigma_in', type=float, default=1e-8,
                        help='Core opacity (default 1e-8, transparent)')
    parser.add_argument('--sigma_out', type=float, default=100.0,
                        help='Shell opacity (default 100, opaque)')
    parser.add_argument('--q_in', type=float, default=1.0,
                        help='Core isotropic source (default 1.0)')
    parser.add_argument('--no-fix', action='store_true',
                        help='Disable the conservative positivity fixup')
    parser.add_argument('--prefix', default='sphere_shell_source',
                        help='Output figure prefix')
    args = parser.parse_args()

    print("\nSteady non-participating spherical transport benchmark")
    print(f"  R = {args.R},  r1 = {args.r1}")
    print(f"  sigma_in = {args.sigma_in:g} (core),  sigma_out = {args.sigma_out:g} (shell)")
    print(f"  q_in = {args.q_in} (core),  vacuum outer BC")
    print(f"  S_{args.N},  I = {args.I} cells")
    print(f"  shell optical depth tau = {args.sigma_out * (args.R - args.r1):g}\n")

    res = solve(args.I, args.N, args.R, args.r1,
                args.sigma_in, args.sigma_out, args.q_in,
                fix=0 if args.no_fix else 1)

    print(f"  source iteration converged in {res['niters']} sweep(s)")
    phi_cen = 0.5 * (res['phi'][:, 0] + res['phi'][:, 1])
    print(f"  phi(0)   = {phi_cen[0]:.6e}")
    print(f"  max phi  = {phi_cen.max():.6e}")
    print(f"  phi(R)   = {phi_cen[-1]:.6e}")

    emitted, absorbed, leaked = energy_balance(res, args.R, args.q_in)
    print("\n  Energy balance (steady state: emitted = absorbed + leaked):")
    print(f"    emitted  = {emitted:.6e}")
    print(f"    absorbed = {absorbed:.6e}")
    print(f"    leaked   = {leaked:.6e}")
    resid = emitted - absorbed - leaked
    print(f"    residual = {resid:.3e}  "
          f"(relative {abs(resid) / (abs(emitted) + 1e-30):.3e})")

    print("\n--- Figure ---")
    plot(res, args.R, args.r1, args.prefix)
    print("\nDone.")


if __name__ == '__main__':
    main()
