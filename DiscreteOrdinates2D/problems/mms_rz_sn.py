#!/usr/bin/env python3
import sys
from pathlib import Path

_REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(_REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPOSITORY_ROOT))

"""
Manufactured Solution (MMS) verification for the 2-D r-z cylindrical S_N solver.

Section 11.5 of the textbook.

Manufactured solution
---------------------
    I_exact(r, z, η, μ) = m_r · r · η  +  m_z · z · μ  +  b

where η = −ρ cos λ, ξ = ρ sin λ, ρ = sqrt(1−μ²).

Applying the cylindrical streaming operator
    L[I] = η ∂I/∂r  +  μ ∂I/∂z  +  (ξ/r) ∂I/∂λ

    η ∂I/∂r       = η · m_r η               = m_r η²
    μ ∂I/∂z       = μ · m_z μ               = m_z μ²
    (ξ/r) ∂I/∂λ  = (ξ/r) · m_r r ξ         = m_r ξ²
                    [since ∂(m_r r η)/∂λ = m_r r ρ sin λ = m_r r ξ]

    L[I_exact] = m_r(η² + ξ²) + m_z μ² = m_r ρ² + m_z μ² = m_r(1−μ²) + m_z μ²

Manufactured source (spatially and angle-dependent):
    Q(r, z, η, μ) = m_r(1−μ²) + m_z μ²  +  σ · I_exact(r, z, η, μ)

At r = 0: I_exact(0, z, η, μ) = m_z z μ + b (independent of η), which is
consistent with the cylindrical axis regularity condition (eq 11.5.7).

Exact scalar flux:
    φ_exact(r, z) = b    (the m_r r η and m_z z μ terms average to zero
                           by symmetry of the quadrature over the sphere)

No scattering (σ_s = 0), so one sweep suffices.

Expected convergence
--------------------
Spatial: O(h²) for the corner-balance scheme (second-order).
Angular: the angular coupling term involves a weighted DD reconstruction of
  the angular edge, which is O(Δλ²) accurate for smooth I(η).  For fixed N
  the angular error saturates as the mesh is refined; for fixed mesh it
  decreases as O(1/N²).

Run
---
    cd DiscreteOrdinates2D
    python problems/mms_rz_sn.py
    python problems/mms_rz_sn.py --Ir 8 16 32 64 --Iz 8 16 32 64 --N 4
    python problems/mms_rz_sn.py --no-convergence
"""

import sys
import os
import math
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_this_dir = os.path.dirname(os.path.abspath(__file__))
_solver_dir = os.path.dirname(_this_dir)   # DiscreteOrdinates2D/
sys.path.insert(0, _solver_dir)

from DiscreteOrdinates2D.src.sn_solver_2d_rz import (
    get_rz_quadrature,
    build_rz_mesh,
    _solve_starting_direction,
    _sweep_one_azimuthal,
)


# ---------------------------------------------------------------------------
# Manufactured solution and source
# ---------------------------------------------------------------------------

def I_exact(r, z, eta, mu, m_r, m_z, b):
    """Manufactured intensity I(r, z, η, μ) = m_r r η + m_z z μ + b."""
    return m_r * r * eta + m_z * z * mu + b


def Q_mms(r, z, eta, mu, sigma, m_r, m_z, b):
    """Manufactured source Q = L[I_exact] + σ I_exact.

    L[I_exact] = m_r(1−μ²) + m_z μ²  (derived in module docstring).
    """
    return (m_r * (1.0 - mu**2) + m_z * mu**2
            + sigma * I_exact(r, z, eta, mu, m_r, m_z, b))


# ---------------------------------------------------------------------------
# Per-angle source and BC builders
# ---------------------------------------------------------------------------

def _build_source(Ir, Iz, r_faces, z_faces, eta, mu, sigma, m_r, m_z, b):
    """Build (Ir, Iz, 4) source array Q at each corner for given (η, μ)."""
    src = np.empty((Ir, Iz, 4))
    for i in range(Ir):
        r_lo = r_faces[i]
        r_hi = r_faces[i + 1]
        for j in range(Iz):
            z_lo = z_faces[j]
            z_hi = z_faces[j + 1]
            src[i, j, 0] = Q_mms(r_hi, z_hi, eta, mu, sigma, m_r, m_z, b)  # NE
            src[i, j, 1] = Q_mms(r_lo, z_hi, eta, mu, sigma, m_r, m_z, b)  # NW
            src[i, j, 2] = Q_mms(r_lo, z_lo, eta, mu, sigma, m_r, m_z, b)  # SW
            src[i, j, 3] = Q_mms(r_hi, z_lo, eta, mu, sigma, m_r, m_z, b)  # SE
    return src


def _build_bcs(Ir, Iz, r_faces, z_faces, eta, mu, m_r, m_z, b):
    """Build boundary-condition arrays for a given angle (η, μ).

    Returns (bc_rlo, bc_rhi, bc_zlo, bc_zhi) each shaped (cells, 2).
    The (cells, 2) axis holds [corner_low, corner_high] along the face.
    """
    r_min = r_faces[0]
    r_max = r_faces[-1]
    z_min = z_faces[0]
    z_max = z_faces[-1]

    # r-faces: axis (j) is cell in z; index 0 = low-z corner, 1 = high-z corner
    bc_rlo = np.zeros((Iz, 2))
    bc_rhi = np.zeros((Iz, 2))
    for j in range(Iz):
        bc_rlo[j, 0] = I_exact(r_min, z_faces[j],     eta, mu, m_r, m_z, b)
        bc_rlo[j, 1] = I_exact(r_min, z_faces[j + 1], eta, mu, m_r, m_z, b)
        bc_rhi[j, 0] = I_exact(r_max, z_faces[j],     eta, mu, m_r, m_z, b)
        bc_rhi[j, 1] = I_exact(r_max, z_faces[j + 1], eta, mu, m_r, m_z, b)

    # z-faces: axis (i) is cell in r; index 0 = low-r corner, 1 = high-r corner
    bc_zlo = np.zeros((Ir, 2))
    bc_zhi = np.zeros((Ir, 2))
    for i in range(Ir):
        bc_zlo[i, 0] = I_exact(r_faces[i],     z_min, eta, mu, m_r, m_z, b)
        bc_zlo[i, 1] = I_exact(r_faces[i + 1], z_min, eta, mu, m_r, m_z, b)
        bc_zhi[i, 0] = I_exact(r_faces[i],     z_max, eta, mu, m_r, m_z, b)
        bc_zhi[i, 1] = I_exact(r_faces[i + 1], z_max, eta, mu, m_r, m_z, b)

    return bc_rlo, bc_rhi, bc_zlo, bc_zhi


def _exact_corners(Ir, Iz, r_faces, z_faces, eta, mu, m_r, m_z, b):
    """Return (Ir, Iz, 4) exact corner values for angle (η, μ)."""
    psi_ex = np.empty((Ir, Iz, 4))
    for i in range(Ir):
        r_lo = r_faces[i]
        r_hi = r_faces[i + 1]
        for j in range(Iz):
            z_lo = z_faces[j]
            z_hi = z_faces[j + 1]
            psi_ex[i, j, 0] = I_exact(r_hi, z_hi, eta, mu, m_r, m_z, b)
            psi_ex[i, j, 1] = I_exact(r_lo, z_hi, eta, mu, m_r, m_z, b)
            psi_ex[i, j, 2] = I_exact(r_lo, z_lo, eta, mu, m_r, m_z, b)
            psi_ex[i, j, 3] = I_exact(r_hi, z_lo, eta, mu, m_r, m_z, b)
    return psi_ex


# ---------------------------------------------------------------------------
# Core: single MMS sweep with per-angle sources and BCs
# ---------------------------------------------------------------------------

def run_mms(Ir, Iz, N_quad, quad_type, m_r, m_z, b, sigma, r_max, z_max):
    """Sweep the MMS problem once (no scattering) and return errors.

    Parameters
    ----------
    Ir, Iz : int   Number of cells in r and z.
    N_quad : int   Quadrature order.
    quad_type : str
    m_r, m_z, b : float  Manufactured-solution parameters.
    sigma : float  Absorption opacity (cm⁻¹).
    r_max, z_max : float  Domain extents.

    Returns
    -------
    phi_num    : (Ir, Iz, 4)  Numerical scalar flux.
    phi_exact  : float        Analytic scalar flux = b.
    all_errors : list of (Ir, Iz, 4) abs-error arrays per angle direction.
    levels     : list of level dicts (for diagnostic prints).
    r_faces, z_faces : face arrays.
    """
    # Mesh
    r_faces = np.linspace(0.0, r_max, Ir + 1)
    z_faces = np.linspace(0.0, z_max, Iz + 1)
    r_ctr, z_ctr, dr, dz = build_rz_mesh(r_faces, z_faces)
    r_lo = r_faces[:-1]
    r_hi = r_faces[1:]
    on_axis = (r_faces[0] < 1e-15)

    # Quadrature
    levels = get_rz_quadrature(quad_type, N_quad)

    # sigma_hat = sigma for steady state (no 1/(c Δt) term)
    sigma_hat = np.full((Ir, Iz, 4), sigma)

    phi_num = np.zeros((Ir, Iz, 4))
    all_errors = []

    for lv in levels:
        mu_l = lv['mu']
        rho_l = lv['rho']
        eta_arr = lv['eta']
        b_arr = lv['b']
        beta = lv['beta']
        w_lk = lv['w_lk']
        K_l = len(eta_arr)

        # --- Starting direction: η = −ρ_ℓ, no angular coupling ---
        eta_g = -rho_l
        src_g = _build_source(Ir, Iz, r_faces, z_faces, eta_g, mu_l,
                               sigma, m_r, m_z, b)
        bc_rlo_g, bc_rhi_g, bc_zlo_g, bc_zhi_g = _build_bcs(
            Ir, Iz, r_faces, z_faces, eta_g, mu_l, m_r, m_z, b)

        g_l = _solve_starting_direction(
            Ir, Iz, dr, dz, r_lo, r_hi, r_ctr,
            rho_l, mu_l, sigma_hat, src_g,
            bc_rlo_g, bc_rhi_g, bc_zlo_g, bc_zhi_g,
            1 if on_axis else 0)

        # --- Seed angular edge: I_{ℓ,1/2} = g_ℓ ---
        ang_edge = g_l.copy()

        # --- Azimuthal directions k = 0 … K_ℓ−1 ---
        for k in range(K_l):
            eta_k = eta_arr[k]
            beta_lo_k = beta[k]
            beta_hi_k = beta[k + 1]
            delta_lam_k = math.pi * b_arr[k]

            src_k = _build_source(Ir, Iz, r_faces, z_faces, eta_k, mu_l,
                                   sigma, m_r, m_z, b)
            bc_rlo_k, bc_rhi_k, bc_zlo_k, bc_zhi_k = _build_bcs(
                Ir, Iz, r_faces, z_faces, eta_k, mu_l, m_r, m_z, b)

            # Axis condition (eq 11.5.7): seed outward r-directions from g_ℓ
            if on_axis and eta_k > 0.0:
                inc_r_lo_k = np.zeros((Iz, 2))
                for jj in range(Iz):
                    inc_r_lo_k[jj, 0] = g_l[0, jj, 2]   # SW corner at r=0
                    inc_r_lo_k[jj, 1] = g_l[0, jj, 1]   # NW corner at r=0
            else:
                inc_r_lo_k = bc_rlo_k

            psi_k, ang_edge = _sweep_one_azimuthal(
                Ir, Iz, dr, dz, r_lo, r_hi, r_ctr,
                eta_k, mu_l, sigma_hat, src_k,
                beta_lo_k, beta_hi_k, delta_lam_k,
                ang_edge,
                inc_r_lo_k, bc_rhi_k,
                bc_zlo_k, bc_zhi_k,
                1 if on_axis else 0)

            phi_num += w_lk[k] * psi_k

            # Error vs exact for this angle
            psi_ex = _exact_corners(Ir, Iz, r_faces, z_faces,
                                     eta_k, mu_l, m_r, m_z, b)
            all_errors.append(np.abs(psi_k - psi_ex))

    return phi_num, b, all_errors, levels, r_faces, z_faces


# ---------------------------------------------------------------------------
# Convergence study
# ---------------------------------------------------------------------------

def convergence_study(Ir_vals, Iz_vals, N_quad, quad_type,
                       m_r, m_z, b_val, sigma, r_max, z_max):
    """Run over a range of mesh sizes; return relative L∞/L² errors."""
    results = {'Ir': [], 'Iz': [], 'h': [],
               'Linf_psi': [], 'L2_psi': [],
               'Linf_phi': [], 'L2_phi': []}
    print(f"\n  {'Ir':>5} {'Iz':>5}  {'h':>8}  "
          f"{'Linf(ψ)':>12}  {'L2(ψ)':>12}  {'Linf(φ)':>12}")
    print("  " + "-" * 62)

    for Ir, Iz in zip(Ir_vals, Iz_vals):
        phi_num, phi_ex_val, errs, levels, rf, zf = run_mms(
            Ir, Iz, N_quad, quad_type, m_r, m_z, b_val, sigma, r_max, z_max)

        # Combine all per-angle errors
        all_abs = np.concatenate([e.ravel() for e in errs])
        Linf_psi = float(all_abs.max())
        L2_psi = float(np.sqrt(np.mean(all_abs**2)))

        # Scalar flux error: compare cell-corner mean to exact constant b
        phi_cell = np.mean(phi_num, axis=2)           # (Ir, Iz)
        phi_err = np.abs(phi_cell - phi_ex_val)
        Linf_phi = float(phi_err.max())
        L2_phi = float(np.sqrt(np.mean(phi_err**2)))

        h = 0.5 * (r_max / Ir + z_max / Iz)   # characteristic mesh size
        results['Ir'].append(Ir)
        results['Iz'].append(Iz)
        results['h'].append(h)
        results['Linf_psi'].append(Linf_psi)
        results['L2_psi'].append(L2_psi)
        results['Linf_phi'].append(Linf_phi)
        results['L2_phi'].append(L2_phi)

        print(f"  {Ir:>5} {Iz:>5}  {h:>8.4f}  "
              f"{Linf_psi:>12.3e}  {L2_psi:>12.3e}  {Linf_phi:>12.3e}")

    return results


def N_convergence_study(N_vals, Ir, Iz, quad_type,
                         m_r, m_z, b_val, sigma, r_max, z_max):
    """Fix the spatial mesh, vary N; report angular discretization error."""
    results = {'N': [], 'Linf_psi': [], 'L2_psi': [], 'Linf_phi': []}
    print(f"\n  {'N':>5}  {'Linf(ψ)':>12}  {'L2(ψ)':>12}  {'Linf(φ)':>12}")
    print("  " + "-" * 50)

    for N in N_vals:
        phi_num, phi_ex_val, errs, levels, rf, zf = run_mms(
            Ir, Iz, N, quad_type, m_r, m_z, b_val, sigma, r_max, z_max)

        all_abs = np.concatenate([e.ravel() for e in errs])
        Linf_psi = float(all_abs.max())
        L2_psi = float(np.sqrt(np.mean(all_abs**2)))

        phi_cell = np.mean(phi_num, axis=2)
        Linf_phi = float(np.abs(phi_cell - phi_ex_val).max())

        results['N'].append(N)
        results['Linf_psi'].append(Linf_psi)
        results['L2_psi'].append(L2_psi)
        results['Linf_phi'].append(Linf_phi)
        print(f"  {N:>5}  {Linf_psi:>12.3e}  {L2_psi:>12.3e}  {Linf_phi:>12.3e}")

    return results


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def _setup_style():
    matplotlib.rcParams.update({
        'font.size': 12,
        'axes.spines.top': False,
        'axes.spines.right': False,
    })


def plot_phi_error(phi_num, phi_ex_val, r_faces, z_faces,
                   N_quad, Ir, Iz, save_prefix='mms_rz'):
    """2-D colour map of |φ − φ_exact|."""
    _setup_style()
    phi_cell = np.mean(phi_num, axis=2)
    err = np.abs(phi_cell - phi_ex_val)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Left: scalar flux
    ax = axes[0]
    vmax = max(phi_cell.max(), 1e-14)
    im = ax.pcolormesh(z_faces, r_faces, phi_cell, shading='flat', cmap='plasma',
                       vmin=0.0, vmax=vmax)
    plt.colorbar(im, ax=ax, label=r'$\varphi$ (numerical)')
    ax.set_xlabel('z (cm)'); ax.set_ylabel('r (cm)')
    ax.set_title(fr'Numerical $\varphi$ ($N={N_quad}$, {Ir}×{Iz})')
    ax.set_aspect('equal')

    # Right: error
    ax = axes[1]
    im2 = ax.pcolormesh(z_faces, r_faces, err, shading='flat', cmap='hot_r')
    plt.colorbar(im2, ax=ax, label=r'$|\varphi - \varphi_{ex}|$')
    ax.set_xlabel('z (cm)'); ax.set_ylabel('r (cm)')
    ax.set_title(f'Scalar flux error (max = {err.max():.2e})')
    ax.set_aspect('equal')

    fig.suptitle(
        fr'MMS r-z cylindrical S$_N$ — $I_{{ex}} = m_r r\eta + m_z z\mu + b$, '
        fr'$\varphi_{{ex}} = {phi_ex_val}$',
        fontsize=11)
    fig.tight_layout()

    fname = f'{save_prefix}_N{N_quad}_phi'
    fig.savefig(fname + '.png', dpi=160, bbox_inches='tight')
    fig.savefig(fname + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}.png / .pdf')


def plot_convergence(results, N_quad, save_prefix='mms_rz'):
    """Log-log convergence plot of errors vs h."""
    _setup_style()
    hs = np.array(results['h'])
    Linf_psi = np.array(results['Linf_psi'])
    L2_psi = np.array(results['L2_psi'])
    Linf_phi = np.array(results['Linf_phi'])

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.loglog(hs, Linf_psi, 'o-b', lw=1.5, ms=6, label=r'$L^\infty(\psi)$')
    ax.loglog(hs, L2_psi,   's--b', lw=1.5, ms=6, label=r'$L^2(\psi)$')
    ax.loglog(hs, Linf_phi, '^:r', lw=1.5, ms=6, label=r'$L^\infty(\varphi)$')

    # Reference slopes
    h_ref = np.array([hs[0], hs[-1]])
    ax.loglog(h_ref, 0.5 * h_ref,      '--k', lw=1, alpha=0.5, label=r'$\mathcal{O}(h)$')
    ax.loglog(h_ref, 0.5 * h_ref**2,   ':k',  lw=1, alpha=0.5, label=r'$\mathcal{O}(h^2)$')

    ax.set_xlabel(r'$h$  (characteristic cell size, cm)')
    ax.set_ylabel('Error')
    ax.set_title(fr'MMS convergence — r-z cylindrical S$_N$ ($N={N_quad}$)')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_linewidth(1.5)
    fig.tight_layout()

    fname = f'{save_prefix}_N{N_quad}_convergence'
    fig.savefig(fname + '.png', dpi=160, bbox_inches='tight')
    fig.savefig(fname + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}.png / .pdf')


def plot_N_convergence(results, Ir, Iz, save_prefix='mms_rz'):
    """Log-log error vs S_N order N (angular convergence)."""
    _setup_style()
    N_arr = np.array(results['N'], dtype=float)
    Linf_psi = np.array(results['Linf_psi'])
    L2_psi = np.array(results['L2_psi'])
    Linf_phi = np.array(results['Linf_phi'])

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.loglog(N_arr, Linf_psi, 'o-b', lw=1.5, ms=6, label=r'$L^\infty(\psi)$')
    ax.loglog(N_arr, L2_psi,   's--b', lw=1.5, ms=6, label=r'$L^2(\psi)$')
    ax.loglog(N_arr, Linf_phi, '^:r', lw=1.5, ms=6, label=r'$L^\infty(\varphi)$')

    # Reference slope O(1/N^2)
    N_ref = np.array([N_arr[0], N_arr[-1]])
    ax.loglog(N_ref, N_ref[0]**2 * Linf_psi[0] / N_ref**2,
              '--k', lw=1, alpha=0.5, label=r'$\mathcal{O}(1/N^2)$')

    ax.set_xlabel(r'$N$ (quadrature order)')
    ax.set_ylabel('Error')
    ax.set_title(fr'Angular convergence ({Ir}×{Iz} spatial mesh)')
    ax.legend(fontsize=10)
    ax.grid(True, which='both', alpha=0.25)
    for spine in ('bottom', 'left'):
        ax.spines[spine].set_linewidth(1.5)
    fig.tight_layout()

    fname = f'{save_prefix}_I{Ir}x{Iz}_N_convergence'
    fig.savefig(fname + '.png', dpi=160, bbox_inches='tight')
    fig.savefig(fname + '.pdf', bbox_inches='tight')
    plt.close(fig)
    print(f'  Saved {fname}.png / .pdf')


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='MMS verification for 2-D r-z cylindrical S_N solver (Section 11.5)')
    parser.add_argument('--Ir', type=int, nargs='+', default=[4, 8, 16, 32],
                        help='Radial cell counts for convergence study')
    parser.add_argument('--Iz', type=int, nargs='+', default=[4, 8, 16, 32],
                        help='Axial cell counts (must match --Ir in length)')
    parser.add_argument('--N', type=int, default=4,
                        help='Quadrature order (default 4)')
    parser.add_argument('--quad', default='product_square',
                        choices=['product_square', 'product_triangular'],
                        help='Quadrature type (default product_square)')
    parser.add_argument('--r-max', type=float, default=2.0,
                        help='Domain radial extent in cm (default 2.0)')
    parser.add_argument('--z-max', type=float, default=3.0,
                        help='Domain axial extent in cm (default 3.0)')
    parser.add_argument('--sigma', type=float, default=1.0,
                        help='Absorption opacity σ (cm⁻¹, default 1.0)')
    parser.add_argument('--mr', type=float, default=0.3,
                        help='m_r slope (default 0.3)')
    parser.add_argument('--mz', type=float, default=0.5,
                        help='m_z slope (default 0.5)')
    parser.add_argument('--b', type=float, default=3.0,
                        help='Offset b (default 3.0)')
    parser.add_argument('--N-vals', type=int, nargs='+', default=[2, 4, 6, 8, 10],
                        help='S_N orders for angular convergence study')
    parser.add_argument('--detail-Ir', type=int, default=16,
                        help='Ir for detail/angular-convergence plots')
    parser.add_argument('--detail-Iz', type=int, default=16,
                        help='Iz for detail/angular-convergence plots')
    parser.add_argument('--no-convergence', action='store_true',
                        help='Skip spatial convergence study')
    parser.add_argument('--no-N-convergence', action='store_true',
                        help='Skip angular convergence study')
    parser.add_argument('--prefix', default='mms_rz',
                        help='Output file prefix')
    args = parser.parse_args()

    m_r = args.mr
    m_z = args.mz
    b_val = args.b
    sigma = args.sigma
    r_max = args.r_max
    z_max = args.z_max
    N_quad = args.N
    quad_type = args.quad

    # Sanity check: I_exact ≥ 0 throughout domain
    rho_max = 1.0   # ρ ≤ 1
    I_min_check = b_val - abs(m_r) * r_max * rho_max - abs(m_z) * z_max
    if I_min_check < 0.0:
        print(f"WARNING: I_exact may be negative (I_min ≈ {I_min_check:.3f}). "
              f"The positivity fixup will alter the solution. "
              f"Increase b or reduce slopes.")

    print(f"\nMMS r-z cylindrical S_N  (Section 11.5)")
    print(f"  I_exact(r, z, η, μ) = {m_r}·r·η + {m_z}·z·μ + {b_val}")
    print(f"  Q(r,z,η,μ) = {m_r}(1−μ²) + {m_z}μ² + {sigma}·I_exact")
    print(f"  φ_exact = {b_val}  (exact scalar flux)")
    print(f"  Domain: r ∈ [0, {r_max}], z ∈ [0, {z_max}] cm")
    print(f"  σ = {sigma} cm⁻¹,  quadrature: {quad_type} N={N_quad}")

    # ── Detail run at the fine mesh ───────────────────────────────────────────
    Ir_d = args.detail_Ir
    Iz_d = args.detail_Iz
    print(f"\n--- Detail run: {Ir_d}×{Iz_d}, N={N_quad} ---")

    phi_num, phi_ex, errs, levels, rf, zf = run_mms(
        Ir_d, Iz_d, N_quad, quad_type, m_r, m_z, b_val, sigma, r_max, z_max)

    all_abs = np.concatenate([e.ravel() for e in errs])
    phi_cell = np.mean(phi_num, axis=2)

    print(f"  φ_exact = {phi_ex:.4f},  "
          f"φ_num mean = {phi_cell.mean():.6f},  "
          f"max|φ−φ_ex| = {np.abs(phi_cell - phi_ex).max():.3e}")
    print(f"  L∞|ψ−ψ_ex| = {all_abs.max():.3e},  "
          f"L²|ψ−ψ_ex| = {np.sqrt(np.mean(all_abs**2)):.3e}")

    plot_phi_error(phi_num, phi_ex, rf, zf, N_quad, Ir_d, Iz_d,
                   save_prefix=args.prefix)

    # ── Spatial convergence study ─────────────────────────────────────────────
    if not args.no_convergence:
        Ir_list = args.Ir
        Iz_list = args.Iz
        if len(Ir_list) != len(Iz_list):
            raise ValueError("--Ir and --Iz must have the same number of values")
        print(f"\n--- Spatial convergence: N={N_quad}, "
              f"Ir ∈ {Ir_list}, Iz ∈ {Iz_list} ---")
        res = convergence_study(Ir_list, Iz_list, N_quad, quad_type,
                                m_r, m_z, b_val, sigma, r_max, z_max)

        if len(Ir_list) >= 3:
            hs = res['h']
            Linf = res['Linf_psi']
            orders = [math.log(Linf[i-1]/Linf[i]) / math.log(hs[i-1]/hs[i])
                      for i in range(1, len(hs))]
            print(f"  Estimated spatial convergence orders (Linf ψ): "
                  f"{[f'{o:.2f}' for o in orders]}")

        plot_convergence(res, N_quad, save_prefix=args.prefix)

    # ── Angular convergence study ─────────────────────────────────────────────
    if not args.no_N_convergence:
        Ir_f = Ir_d
        Iz_f = Iz_d
        print(f"\n--- Angular convergence: {Ir_f}×{Iz_f} mesh, "
              f"N ∈ {args.N_vals} ---")
        res_N = N_convergence_study(args.N_vals, Ir_f, Iz_f, quad_type,
                                     m_r, m_z, b_val, sigma, r_max, z_max)
        plot_N_convergence(res_N, Ir_f, Iz_f, save_prefix=args.prefix)

    print("\nDone.")


if __name__ == '__main__':
    main()
