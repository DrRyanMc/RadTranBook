"""
2-D Discrete Ordinates (S_N) solver for axisymmetric cylindrical (r-z) geometry.

Implements the method described in Chapter 11.5 of the textbook.

Spatial discretization: Simple Corner Balance (SCB) on a rectangular r-z mesh.
Geometry: each cell (i,j) has radial coordinate r_i and axial coordinate z_j.
Angular variables:
  - μ   = axial direction cosine (constant on an angular level)
  - η   = -ρ cos λ = radial direction cosine  (ρ = sqrt(1-μ²))
  - ξ   = ρ sin λ  = azimuthal direction cosine
  - λ   = transverse angle, 0 ≤ λ ≤ π (0 = radially inward, π = radially outward)

Key features beyond 2-D Cartesian:
  - Angular levels of constant μ_ℓ, with K_ℓ azimuthal directions within each level.
  - Angular metric coefficients β_{ℓ,k+1/2} defined recursively (eq 11.57).
  - A starting-direction intensity g_ℓ(r,z,t) at λ=0 (η=-ρ_ℓ, ξ=0) per level.
  - Cylindrical corner volumes and face areas replacing Cartesian ones.
  - Axis condition at r=0: outgoing radial directions seeded by g_ℓ(0,z,t).
  - Sweep is nested: starting direction → k=1 → k=2 → … → k=K_ℓ per level.

Quadrature support:
  - Product quadrature in (μ, λ): explicit levels, easiest to use.
  - Level-symmetric and equal-weight can be used if grouped into levels.
  A helper `get_rz_quadrature` returns the required level structure.

Array conventions:
  - phi, T, sigma : (Ir, Iz, 4) — corner-averaged values
  - Corner numbering: 0=NE, 1=NW, 2=SW, 3=SE
    (NE = high-r, high-z corner; counterclockwise from NE)
  - g_level : (N_levels, Ir, Iz, 4) — starting-direction intensity per level
  - psi_level: (N_levels, K_max, Ir, Iz, 4) — angular intensity per (level, angle)

Physical constants in CGS with time in nanoseconds (same as Cartesian solver).
"""

import math
import numpy as np
from numba import jit, prange
import sys
import os

# Add DiscreteOrdinates2D/ (this file's directory) so that 'from quadratures import'
# works regardless of where the script is invoked from.
_this_file_dir = os.path.dirname(os.path.abspath(__file__))
if _this_file_dir not in sys.path:
    sys.path.insert(0, _this_file_dir)

sys.path.insert(0, os.path.join(_this_file_dir, '..', 'DiscreteOrdinates'))
from DiscreteOrdinates.src.sn_solver import solver_with_dmd_inc as _solver_with_dmd_inc_1d

# Physical constants (CGS, time in ns) — match sn_solver_2d.py
c = 29.98
a = 0.01372
ac = a * c


# ===========================================================================
# Quadrature: grouping into angular levels
# ===========================================================================

def get_rz_quadrature(quad_type='product_square', N=4):
    """Return quadrature organized into angular levels of constant μ.

    For r-z (axisymmetric cylindrical) problems the axial direction cosine
    μ is constant along a characteristic.  We therefore group directions
    into levels of constant μ and, within each level, order the transverse
    angle λ from 0 (radially inward, η=-ρ) to π (radially outward, η=+ρ).

    Product quadrature in (μ, λ) is the cleanest because levels are exact.
    For LS/EQ quadratures the grouping is approximate — directions are binned
    by their |μ| value (within a tolerance).

    Returns
    -------
    levels : list of dicts, each with
        'mu'    : float          — μ_ℓ (axial direction cosine, positive)
        'rho'   : float          — ρ_ℓ = sqrt(1-μ_ℓ²)
        'w_mu'  : float          — polar weight (so sum over levels of w_mu = 0.5)
        'lam'   : (K_ℓ,) float  — transverse angles λ_{ℓ,k}, 0 < λ < π
        'eta'   : (K_ℓ,) float  — η_{ℓ,k} = -ρ_ℓ cos λ_{ℓ,k}
        'xi'    : (K_ℓ,) float  — ξ_{ℓ,k} = ρ_ℓ sin λ_{ℓ,k}
        'b'     : (K_ℓ,) float  — within-level weights (sum = 1)
        'beta'  : (K_ℓ+1,) float— angular metric coefficients β_{ℓ,k+1/2}
        'w_lk'  : (K_ℓ,) float  — full 3-D weight w_{ℓ,k} = w_mu * b_{ℓ,k}
    """
    if quad_type in ('product_square', 'product_triangular'):
        return _product_rz_levels(N, triangular=(quad_type == 'product_triangular'))
    elif quad_type in ('level_symmetric', 'equal_weight'):
        return _group_rz_levels(quad_type, N)
    else:
        raise ValueError(f"Unknown quad_type '{quad_type}'. "
                         "Use 'product_square', 'product_triangular', "
                         "'level_symmetric', or 'equal_weight'.")


def _product_rz_levels(Np, triangular=False):
    """Product quadrature in (μ, λ) for r-z geometry.

    Both +μ and -μ levels are included because r-z problems are not
    generally symmetric in z.  Weights are normalised so that
    sum over all levels and azimuths of w_lk = 1.
    """
    mu_nodes_full, w_polar_full = np.polynomial.legendre.leggauss(Np)
    # Normalise polar weights to sum to 1
    w_polar_full = w_polar_full / w_polar_full.sum()

    levels = []
    for i, (mu_l, w_mu) in enumerate(zip(mu_nodes_full, w_polar_full)):
        rho_l = math.sqrt(max(0.0, 1.0 - mu_l**2))
        if triangular:
            Na = max(2, Np - 2 * abs(i - Np // 2))
        else:
            Na = Np

        # Transverse angles uniformly in (0, π): λ_k = (2k+1)π/(2Na)
        lam = np.array([(2*k + 1) * math.pi / (2 * Na) for k in range(Na)])
        eta = -rho_l * np.cos(lam)
        xi = rho_l * np.sin(lam)
        b = np.ones(Na) / Na  # within-level weights, sum=1

        beta = _compute_beta(eta, b, rho_l)

        levels.append({
            'mu': mu_l, 'rho': rho_l, 'w_mu': w_mu,
            'lam': lam, 'eta': eta, 'xi': xi,
            'b': b, 'beta': beta,
            'w_lk': w_mu * b,
        })
    return levels


def _group_rz_levels(quad_type, N):
    """Group LS/EW quadrature directions into levels by common |\u03bc_z|."""
    from quadratures import (
        level_symmetric_quadrature, equal_weight_quadrature,
    )
    if quad_type == 'level_symmetric':
        omegas, weights = level_symmetric_quadrature(N)
    else:
        omegas, weights = equal_weight_quadrature(N)

    # Keep upper hemisphere (μ_z > 0) — the physical z-axis.
    # In cylindrical, x->r, y->z, z->azimuthal (symmetry axis).
    # We exploit the full-sphere symmetry: use all μ_z > 0 directions.
    pos = omegas[:, 2] > 1e-12
    eta_all = omegas[pos, 0]   # r direction cosine
    mu_all = omegas[pos, 1]    # z direction cosine
    xi_sq = np.maximum(0.0, 1.0 - eta_all**2 - mu_all**2)
    xi_all = np.sqrt(xi_sq)
    w_all = weights[pos] * 2.0  # double for z-symmetry
    w_all /= w_all.sum()

    # Group by mu value (within tolerance)
    tol = 1e-8
    used = np.zeros(len(mu_all), dtype=bool)
    levels = []
    for i in range(len(mu_all)):
        if used[i]:
            continue
        mu_l = mu_all[i]
        same = np.abs(mu_all - mu_l) < tol
        used |= same

        rho_l = math.sqrt(max(0.0, 1.0 - mu_l**2))
        # Sort within level by λ = atan2(ξ, -η) increasing from 0 to π
        lam_i = np.arctan2(xi_all[same], -eta_all[same])
        order = np.argsort(lam_i)
        lam = lam_i[order]
        eta_l = eta_all[same][order]
        xi_l = xi_all[same][order]
        w_l = w_all[same][order]

        # Within-level weights normalised to 1; w_mu is the polar total
        w_mu = w_l.sum()
        b = w_l / w_mu

        beta = _compute_beta(eta_l, b, rho_l)

        levels.append({
            'mu': mu_l, 'rho': rho_l, 'w_mu': w_mu,
            'lam': lam, 'eta': eta_l, 'xi': xi_l,
            'b': b, 'beta': beta,
            'w_lk': w_mu * b,
        })

    # Sort levels by increasing μ
    levels.sort(key=lambda lv: lv['mu'])
    return levels


def _compute_beta(eta, b, rho):
    """Compute angular metric coefficients β_{k+1/2} from eq 11.57.

    β_{1/2} = 0
    β_{k+1/2} = β_{k-1/2} - η_k * Δλ_k,   Δλ_k = π * b_k.

    For a symmetric within-level quadrature β_{K+1/2} = 0 automatically.
    """
    K = len(eta)
    beta = np.zeros(K + 1)
    # β_{1/2} = 0 (starting edge)
    for k in range(K):
        delta_lam_k = math.pi * b[k]
        beta[k + 1] = beta[k] - eta[k] * delta_lam_k
    # Force exact closure for symmetric quadratures
    beta[K] = 0.0
    return beta


# ===========================================================================
# Cylindrical geometry helpers
# ===========================================================================

def build_rz_mesh(r_faces, z_faces):
    """Return cell-centre arrays and derived geometry from face arrays."""
    r_centers = 0.5 * (r_faces[:-1] + r_faces[1:])
    z_centers = 0.5 * (z_faces[:-1] + z_faces[1:])
    dr = np.diff(r_faces)
    dz = np.diff(z_faces)
    return r_centers, z_centers, dr, dz


# ===========================================================================
# JIT-compiled cylindrical corner-balance cell solver
# ===========================================================================

@jit(nopython=True, cache=True)
def _solve_cyl_cell_corners(eta, mu, dr, dz,
                             r_lo, r_hi, r_i,
                             sig0, sig1, sig2, sig3,
                             src0, src1, src2, src3,
                             inc_r0, inc_r1,   # incoming on r-face (lo=SW/NW, hi=SE/NE)
                             inc_z0, inc_z1,   # incoming on z-face (lo=SW/SE, hi=NW/NE)
                             beta_lo, beta_hi, delta_lam,
                             ang_edge0, ang_edge1, ang_edge2, ang_edge3,
                             include_axis_term):
    """Solve the 4×4 cylindrical SCB system for one cell.

    Corner numbering: 0=NE (high-r, high-z), 1=NW (low-r, high-z),
                      2=SW (low-r, low-z),   3=SE (high-r, low-z).

    Cylindrical geometric measures (eq 11.60–11.64):
      V_c     = ½[(r_hc)² - (r_lc)²] * Δz_c       (corner volume)
      A_r_ext = r_ext * Δz_c                        (exterior radial face)
      A_r_int = r_i   * Δz_c                        (interior radial face)
      A_z_c   = ½[(r_hc)² - (r_lc)²]               (axial face)
      A_rz_c  = Δr_c  * Δz_c                        (unweighted, for β-term)

    Parameters
    ----------
    eta, mu     : float — direction cosines (η radial, μ axial)
    dr, dz      : float — full cell widths
    r_lo, r_hi  : float — cell radial face positions r_{i-1/2}, r_{i+1/2}
    r_i         : float — cell centre radius
    sig0..3     : float — σ_hat per corner
    src0..3     : float — total source per corner
    inc_r0/1    : float — incoming r-face values (bottom/top of face)
    inc_z0/1    : float — incoming z-face values (left/right of face)
    beta_lo, beta_hi : float — β_{k-1/2}, β_{k+1/2} angular metric coefficients
    delta_lam   : float — Δλ_k = π * b_k within-level angular weight
    ang_edge0..3: float — I_{k-1/2} at each corner (incoming angular edge)
    include_axis_term : int — 1 if r_lo == 0 (axis), exterior r-face area = 0
    """
    # Half-cell dimensions
    dr_h = 0.5 * dr
    dz_h = 0.5 * dz

    # Corner radial and axial bounds
    # 0=NE: r in [r_i, r_hi], z in [z_j, z_hi]
    # 1=NW: r in [r_lo, r_i], z in [z_j, z_hi]
    # 2=SW: r in [r_lo, r_i], z in [z_lo, z_j]
    # 3=SE: r in [r_i, r_hi], z in [z_lo, z_j]
    # Each corner: r_lc, r_hc
    r_lc0 = r_i;   r_hc0 = r_hi   # NE
    r_lc1 = r_lo;  r_hc1 = r_i    # NW
    r_lc2 = r_lo;  r_hc2 = r_i    # SW
    r_lc3 = r_i;   r_hc3 = r_hi   # SE

    # Corner volumes: V = ½[(r_hc)²-(r_lc)²] * dz_h
    V0 = 0.5 * (r_hc0**2 - r_lc0**2) * dz_h
    V1 = 0.5 * (r_hc1**2 - r_lc1**2) * dz_h
    V2 = 0.5 * (r_hc2**2 - r_lc2**2) * dz_h
    V3 = 0.5 * (r_hc3**2 - r_lc3**2) * dz_h

    # Axial face areas: A_z = ½[(r_hc)²-(r_lc)²]
    Az0 = 0.5 * (r_hc0**2 - r_lc0**2)
    Az1 = 0.5 * (r_hc1**2 - r_lc1**2)
    Az2 = 0.5 * (r_hc2**2 - r_lc2**2)
    Az3 = 0.5 * (r_hc3**2 - r_lc3**2)

    # Exterior radial face areas: A_r_ext = r_ext * dz_h
    # NE, SE corners: exterior face at r=r_hi; NW, SW corners: exterior at r=r_lo
    # If at axis (r_lo=0 for NW,SW) exterior area = 0
    if include_axis_term:
        Ar_ext0 = r_hc0 * dz_h
        Ar_ext1 = 0.0         # axis: r_lo = 0
        Ar_ext2 = 0.0         # axis
        Ar_ext3 = r_hc3 * dz_h
    else:
        Ar_ext0 = r_hc0 * dz_h
        Ar_ext1 = r_lc1 * dz_h
        Ar_ext2 = r_lc2 * dz_h
        Ar_ext3 = r_hc3 * dz_h

    # Interior radial face areas: A_r_int = r_i * dz_h
    Ar_int = r_i * dz_h   # same for all corners

    # Unweighted corner areas for angular β-term: A_rz = dr_h * dz_h
    Arz0 = dr_h * dz_h
    Arz1 = dr_h * dz_h
    Arz2 = dr_h * dz_h
    Arz3 = dr_h * dz_h

    # Direction flow signs
    # For η: sx_c = +1 for NE/SE (outward-r corners), -1 for NW/SW (inward-r corners)
    # For μ: sz_c = +1 for NE/NW (high-z), -1 for SE/SW (low-z)
    alpha_r0 = eta   # NE: s_r = +1
    alpha_r1 = -eta  # NW: s_r = -1
    alpha_r2 = -eta  # SW: s_r = -1
    alpha_r3 = eta   # SE: s_r = +1
    alpha_z0 = mu    # NE: s_z = +1
    alpha_z1 = mu    # NW: s_z = +1
    alpha_z2 = -mu   # SW: s_z = -1
    alpha_z3 = -mu   # SE: s_z = -1

    # α+ = max(α,0), α- = max(-α,0)
    def aplus(x):  return x if x > 0.0 else 0.0
    def aminus(x): return (-x) if x < 0.0 else 0.0

    # Angular β-term per corner: (A_rz / Δλ) * (β_{k+1/2} I_{k+1/2} - β_{k-1/2} I_{k-1/2})
    # = (A_rz / Δλ) * (β_hi * I_hi - β_lo * ang_edge_c)
    # I_{k+1/2,c} is the outgoing angular edge, closed by weighted DD:
    #   I_c = s * I_{k+1/2} + (1-s) * I_{k-1/2}  =>  I_{k+1/2} = (I_c - (1-s)*ang_edge_c)/s
    # We defer computing I_{k+1/2} until after solving for I_c.
    # For the LHS matrix we treat the β-term as:
    #   β_hi * I_{k+1/2,c} = β_hi/s * (I_c - (1-s)*ang_edge_c)
    # which contributes β_hi/s to the diagonal and a known RHS term from ang_edge_c.

    # within-level weight s_k = (λ_k - λ_{k-1/2})/(λ_{k+1/2} - λ_{k-1/2})
    # For uniform spacing Δλ_k = π*b_k, λ_{k-1/2} is the lower edge.
    # The textbook uses s (eqs 11.55) for the weighted DD closure.
    # For uniform Chebyshev: s = 0.5 (symmetric).
    # We pass s = 0.5 for product quadrature (lambda midpoints).
    # In general s = (lam_k - lam_{k-1/2}) / delta_lam_k.
    # We use s_k = 0.5 (symmetric within-level spacing for product quad).
    s_k = 0.5

    if delta_lam > 1e-20:
        coeff_beta = beta_hi / (s_k * delta_lam)
    else:
        coeff_beta = 0.0

    # Diagonal entries:
    # d_c = sigma_c * V_c + A_r_ext_c * alpha_r_c^+ - ½ A_r_int * alpha_r_c
    #       + ½ Az_c * |alpha_z_c| + (A_rz_c / Δλ) * beta_hi / s_k
    d0 = sig0*V0 + Ar_ext0*aplus(alpha_r0) - 0.5*Ar_int*alpha_r0 + 0.5*Az0*abs(alpha_z0) + Arz0*coeff_beta
    d1 = sig1*V1 + Ar_ext1*aplus(alpha_r1) - 0.5*Ar_int*alpha_r1 + 0.5*Az1*abs(alpha_z1) + Arz1*coeff_beta
    d2 = sig2*V2 + Ar_ext2*aplus(alpha_r2) - 0.5*Ar_int*alpha_r2 + 0.5*Az2*abs(alpha_z2) + Arz2*coeff_beta
    d3 = sig3*V3 + Ar_ext3*aplus(alpha_r3) - 0.5*Ar_int*alpha_r3 + 0.5*Az3*abs(alpha_z3) + Arz3*coeff_beta

    # Off-diagonal entries from internal face coupling:
    # Radial internal face: -½ A_r_int * alpha_r couples radially adjacent corners
    #   c_r[0]= 3 (SE) for NE with shared face at r_i on z-high side
    #   c_r[1]= 2 (SW) for NW
    #   c_r[2]= 3 — wait, SW/SE share the r_i face differently
    # Coupling pattern (same as Cartesian but η replaces Ω_x):
    #   NE(0) ↔ NW(1) via r-face (s_r*η changes sign)
    #   SE(3) ↔ SW(2) via r-face
    #   NE(0) ↔ SE(3) via z-face (s_z*μ changes sign)
    #   NW(1) ↔ SW(2) via z-face

    # Off-diagonal entries follow the same rule as Cartesian:
    #   A[c, c'] = -0.5 * face_area * alpha_c
    # where alpha_c includes the sign of the flow direction for corner c.
    # For r-direction: face_area = Ar_int (corner's portion of internal r-face).
    # For z-direction: face_area = Az_c (corner's cylindrical axial face area).
    #
    # Row 0 (NE): r-coupling to NW(1), z-coupling to SE(3)
    a01 = -0.5 * Ar_int * alpha_r0   # alpha_r0 = +η  → negative for η>0
    a03 = -0.5 * Az0    * alpha_z0   # alpha_z0 = +μ  → negative for μ>0
    # Row 1 (NW): r-coupling to NE(0), z-coupling to SW(2)
    a10 = -0.5 * Ar_int * alpha_r1   # alpha_r1 = -η  → positive for η>0
    a12 = -0.5 * Az1    * alpha_z1   # alpha_z1 = +μ  → negative for μ>0
    # Row 2 (SW): z-coupling to NW(1), r-coupling to SE(3)
    a21 = -0.5 * Az2    * alpha_z2   # alpha_z2 = -μ  → positive for μ>0
    a23 = -0.5 * Ar_int * alpha_r2   # alpha_r2 = -η  → positive for η>0
    # Row 3 (SE): z-coupling to NE(0), r-coupling to SW(2)
    a30 = -0.5 * Az3    * alpha_z3   # alpha_z3 = -μ  → positive for μ>0
    a32 = -0.5 * Ar_int * alpha_r3   # alpha_r3 = +η  → negative for η>0

    # RHS: source + external boundary contributions + angular β-term (known part)
    b0 = src0 * V0
    b1 = src1 * V1
    b2 = src2 * V2
    b3 = src3 * V3

    # Angular β-term: after substituting the weighted-DD closure for the outgoing
    # angular edge, the known incoming angular edge I_{k-1/2} contributes:
    #   (A_rz / Δλ) * [β_hi*(1-s_k)/s_k + β_lo] * I_{k-1/2}
    # to the RHS (see derivation in Section 11.5.5; the sign is +β_hi*(1-s)/s + β_lo).
    if delta_lam > 1e-20:
        beta_rhs_coeff = (beta_hi * (1.0 - s_k) / s_k + beta_lo) / delta_lam
    else:
        beta_rhs_coeff = 0.0

    b0 += Arz0 * beta_rhs_coeff * ang_edge0
    b1 += Arz1 * beta_rhs_coeff * ang_edge1
    b2 += Arz2 * beta_rhs_coeff * ang_edge2
    b3 += Arz3 * beta_rhs_coeff * ang_edge3

    # External r-face: Ar_ext * α_r^- * inc_r
    # inc_r0 = incoming bottom (low-z side of r-face), inc_r1 = high-z side
    # Corner assignments:
    #   NE(0): exterior r-face at r_hi, inc = inc_r1 (high-z) if η<0 (entering from outside)
    #   SE(3): exterior r-face at r_hi, inc = inc_r0 (low-z)  if η<0
    #   NW(1): exterior r-face at r_lo, inc = inc_r1 (high-z) if η>0 (entering from inside)
    #   SW(2): exterior r-face at r_lo, inc = inc_r0 (low-z)  if η>0
    b0 += Ar_ext0 * aminus(alpha_r0) * inc_r1   # NE, high-z side
    b3 += Ar_ext3 * aminus(alpha_r3) * inc_r0   # SE, low-z side
    b1 += Ar_ext1 * aminus(alpha_r1) * inc_r1   # NW, high-z side
    b2 += Ar_ext2 * aminus(alpha_r2) * inc_r0   # SW, low-z side

    # External z-face: Az_c * |α_z_c| * inc_z
    # (Uses the full corner Az area, analogous to Cartesian which uses Ay not Ay/2.)
    # inc_z0 = low-r side of z-face, inc_z1 = high-r side
    # SW(2): z-face low-z,  inc_z0 (low-r)  if μ>0
    # SE(3): z-face low-z,  inc_z1 (high-r) if μ>0
    # NE(0): z-face high-z, inc_z1 (high-r) if μ<0
    # NW(1): z-face high-z, inc_z0 (low-r)  if μ<0
    if mu > 0.0:
        b2 += Az2 * mu * inc_z0
        b3 += Az3 * mu * inc_z1
    elif mu < 0.0:
        b0 += Az0 * (-mu) * inc_z1
        b1 += Az1 * (-mu) * inc_z0

    # Gaussian elimination (same sparse structure as Cartesian):
    # Row 0 has off-diag a01 (col 1) and a03 (col 3)
    # Row 1 has off-diag a10 (col 0) and a12 (col 2)
    # Row 2 has off-diag a21 (col 1) and a23 (col 3)
    # Row 3 has off-diag a30 (col 0) and a32 (col 2)
    inv_d0 = 1.0 / d0
    f1 = a10 * inv_d0
    f3 = a30 * inv_d0

    d1p = d1 - f1 * a01
    g13 = -f1 * a03
    b1p = b1 - f1 * b0

    g31 = -f3 * a01
    d3p = d3 - f3 * a03
    b3p = b3 - f3 * b0

    inv_d1p = 1.0 / d1p
    f2 = a21 * inv_d1p
    f3b = g31 * inv_d1p

    d2p = d2 - f2 * a12
    g23 = a23 - f2 * g13
    b2p = b2 - f2 * b1p

    g32 = a32 - f3b * a12
    d3pp = d3p - f3b * g13
    b3pp = b3p - f3b * b1p

    inv_d2p = 1.0 / d2p
    f3c = g32 * inv_d2p

    d3ppp = d3pp - f3c * g23
    b3ppp = b3pp - f3c * b2p

    I3 = b3ppp / d3ppp
    I2 = (b2p - g23 * I3) * inv_d2p
    I1 = (b1p - a12 * I2 - g13 * I3) * inv_d1p
    I0 = (b0 - a01 * I1 - a03 * I3) * inv_d0

    return I0, I1, I2, I3


# ===========================================================================
# Cartesian cell solver for the starting direction
# ===========================================================================
# The starting-direction equation (eq 11.59) is slab-like: it uses the
# NON-CONSERVATIVE radial operator -ρ ∂g/∂r, not (1/r)∂(rηg)/∂r.  This is
# equivalent to the Cartesian corner-balance equation with direction cosines
# Ox = -ρ_ℓ and Oy = μ_ℓ and uniform face areas (no cylindrical r-weighting).

@jit(nopython=True, cache=True)
def _solve_cart2d_corners(Ox, Oy, dx, dy, sig0, sig1, sig2, sig3,
                          src0, src1, src2, src3,
                          inc_x0, inc_x1, inc_y0, inc_y1):
    """2-D Cartesian SCB for one cell — identical physics to the Cartesian solver.

    Used for the starting direction whose equation is slab-like (eq 11.59).
    Corners: 0=NE, 1=NW, 2=SW, 3=SE.
    inc_x = [low-z, high-z] incoming on x(=r) face.
    inc_y = [low-r, high-r] incoming on y(=z) face.
    """
    Ax = dy * 0.5        # half-cell y-extent (= vertical face area per corner)
    Ay = dx * 0.5        # half-cell x-extent
    V = dx * dy * 0.25   # corner volume
    half_Ax = Ax * 0.5
    half_Ay = Ay * 0.5

    abs_Ox = Ox if Ox > 0.0 else -Ox
    abs_Oy = Oy if Oy > 0.0 else -Oy

    diag_geom = half_Ax * abs_Ox + half_Ay * abs_Oy
    d0 = sig0 * V + diag_geom
    d1 = sig1 * V + diag_geom
    d2 = sig2 * V + diag_geom
    d3 = sig3 * V + diag_geom

    half_Ax_Ox = half_Ax * Ox
    half_Ay_Oy = half_Ay * Oy

    a01 = -half_Ax_Ox;  a03 = -half_Ay_Oy
    a10 =  half_Ax_Ox;  a12 = -half_Ay_Oy
    a21 =  half_Ay_Oy;  a23 =  half_Ax_Ox
    a30 =  half_Ay_Oy;  a32 = -half_Ax_Ox

    b0 = V * src0;  b1 = V * src1
    b2 = V * src2;  b3 = V * src3

    if Ox > 0.0:
        Ax_Ox = Ax * Ox
        b1 += Ax_Ox * inc_x1
        b2 += Ax_Ox * inc_x0
    elif Ox < 0.0:
        Ax_nOx = Ax * (-Ox)
        b0 += Ax_nOx * inc_x1
        b3 += Ax_nOx * inc_x0

    if Oy > 0.0:
        Ay_Oy = Ay * Oy
        b2 += Ay_Oy * inc_y0
        b3 += Ay_Oy * inc_y1
    elif Oy < 0.0:
        Ay_nOy = Ay * (-Oy)
        b0 += Ay_nOy * inc_y1
        b1 += Ay_nOy * inc_y0

    inv_d0 = 1.0 / d0
    f1 = a10 * inv_d0;  f3 = a30 * inv_d0
    d1p = d1 - f1 * a01;  g13 = -f1 * a03;  b1p = b1 - f1 * b0
    g31 = -f3 * a01;      d3p = d3 - f3 * a03;  b3p = b3 - f3 * b0
    inv_d1p = 1.0 / d1p
    f2 = a21 * inv_d1p;   f3b = g31 * inv_d1p
    d2p = d2 - f2 * a12;  g23 = a23 - f2 * g13;  b2p = b2 - f2 * b1p
    g32 = a32 - f3b * a12; d3pp = d3p - f3b * g13; b3pp = b3p - f3b * b1p
    inv_d2p = 1.0 / d2p
    f3c = g32 * inv_d2p
    d3ppp = d3pp - f3c * g23;  b3ppp = b3pp - f3c * b2p
    I3 = b3ppp / d3ppp
    I2 = (b2p - g23 * I3) * inv_d2p
    I1 = (b1p - a12 * I2 - g13 * I3) * inv_d1p
    I0 = (b0  - a01 * I1 - a03 * I3) * inv_d0
    return I0, I1, I2, I3


@jit(nopython=True, cache=True)
def _solve_starting_direction(Ir, Iz, dr_arr, dz_arr, r_lo_arr, r_hi_arr, r_ctr,
                              rho_l, mu_l, sigma_hat, source_g,
                              inc_r_lo, inc_r_hi, inc_z_lo, inc_z_hi,
                              on_axis):
    """Sweep the starting-direction equation (eq 11.59) for one level.

    The starting direction has η = -ρ_ℓ, ξ = 0.  The equation is slab-like
    (no angular derivative), but uses the *non-conservative* radial operator
    (no 1/r factor) because the angular derivative term cancels it at λ=0.

    Uses the cylindrical SCB with β_lo = β_hi = 0 (no angular coupling) and
    η = -ρ_ℓ (purely inward in r).

    Returns g : (Ir, Iz, 4) starting-direction intensity.
    """
    g = np.zeros((Ir, Iz, 4))

    eta_g = -rho_l   # radially inward: η = -ρ_ℓ
    mu_g = mu_l

    # Sweep direction: η < 0 → r-sweep from high to low; μ > 0 → z-sweep low to high
    if eta_g < 0.0:
        r_start = Ir - 1; r_end = -1; r_step = -1
    else:
        r_start = 0; r_end = Ir; r_step = 1

    if mu_g >= 0.0:
        z_start = 0; z_end = Iz; z_step = 1
    else:
        z_start = Iz - 1; z_end = -1; z_step = -1

    # Face value arrays for upwind passing
    psi_r_face = np.zeros((Iz, 2))   # r-face: [bottom(low-z), top(high-z)]
    psi_z_face = np.zeros((Ir, 2))   # z-face: [low-r, high-r]

    # Initialise from boundary conditions
    if eta_g < 0.0:
        for j in range(Iz):
            psi_r_face[j, 0] = inc_r_hi[j, 0]
            psi_r_face[j, 1] = inc_r_hi[j, 1]
    else:
        for j in range(Iz):
            psi_r_face[j, 0] = inc_r_lo[j, 0]
            psi_r_face[j, 1] = inc_r_lo[j, 1]

    if mu_g >= 0.0:
        for i in range(Ir):
            psi_z_face[i, 0] = inc_z_lo[i, 0]
            psi_z_face[i, 1] = inc_z_lo[i, 1]
    else:
        for i in range(Ir):
            psi_z_face[i, 0] = inc_z_hi[i, 0]
            psi_z_face[i, 1] = inc_z_hi[i, 1]

    # Zero angular-edge inputs for starting direction (no β coupling)
    i = r_start
    while (r_step > 0 and i < r_end) or (r_step < 0 and i > r_end):
        j = z_start
        while (z_step > 0 and j < z_end) or (z_step < 0 and j > z_end):
            # Cartesian-geometry cell solver: the starting-direction equation
            # (eq 11.59) is slab-like, not cylindrical.  Uniform face areas
            # (no r-weighting), no angular β-coupling.
            # Ox = eta_g = -rho  (radial direction cosine)
            # Oy = mu_g          (axial direction cosine)
            # inc_x = r-face: [low-z, high-z]; inc_y = z-face: [low-r, high-r]
            I0, I1, I2, I3 = _solve_cart2d_corners(
                eta_g, mu_g, dr_arr[i], dz_arr[j],
                sigma_hat[i, j, 0], sigma_hat[i, j, 1],
                sigma_hat[i, j, 2], sigma_hat[i, j, 3],
                source_g[i, j, 0], source_g[i, j, 1],
                source_g[i, j, 2], source_g[i, j, 3],
                psi_r_face[j, 0], psi_r_face[j, 1],
                psi_z_face[i, 0], psi_z_face[i, 1])

            if I0 < 0.0: I0 = 0.0
            if I1 < 0.0: I1 = 0.0
            if I2 < 0.0: I2 = 0.0
            if I3 < 0.0: I3 = 0.0

            g[i, j, 0] = I0
            g[i, j, 1] = I1
            g[i, j, 2] = I2
            g[i, j, 3] = I3

            # Propagate upwind face values
            if eta_g < 0.0:
                psi_r_face[j, 0] = I2   # SW = low-z → goes left
                psi_r_face[j, 1] = I1   # NW = high-z
            else:
                psi_r_face[j, 0] = I3   # SE = low-z → goes right
                psi_r_face[j, 1] = I0   # NE = high-z

            if mu_g >= 0.0:
                psi_z_face[i, 0] = I1   # NW = low-r going up
                psi_z_face[i, 1] = I0   # NE = high-r going up
            else:
                psi_z_face[i, 0] = I2   # SW = low-r going down
                psi_z_face[i, 1] = I3   # SE = high-r going down

            j += z_step
        i += r_step

    return g


@jit(nopython=True, cache=True)
def _sweep_one_azimuthal(Ir, Iz, dr_arr, dz_arr, r_lo_arr, r_hi_arr, r_ctr,
                          eta, mu, sigma_hat, source,
                          beta_lo, beta_hi, delta_lam,
                          ang_edge_in,
                          inc_r_lo, inc_r_hi, inc_z_lo, inc_z_hi,
                          on_axis):
    """Sweep one (ℓ, k) direction in r-z and return intensity + outgoing angular edge.

    Parameters
    ----------
    ang_edge_in : (Ir, Iz, 4) — I_{k-1/2} incoming angular-edge values
    Returns: psi (Ir,Iz,4), ang_edge_out (Ir,Iz,4)
    """
    psi = np.zeros((Ir, Iz, 4))
    ang_edge_out = np.zeros((Ir, Iz, 4))

    if eta < 0.0:
        r_start = Ir - 1; r_end = -1; r_step = -1
    else:
        r_start = 0; r_end = Ir; r_step = 1

    if mu >= 0.0:
        z_start = 0; z_end = Iz; z_step = 1
    else:
        z_start = Iz - 1; z_end = -1; z_step = -1

    psi_r_face = np.zeros((Iz, 2))
    psi_z_face = np.zeros((Ir, 2))

    # Initialise from boundary conditions (inc_r_lo, inc_r_hi for r-faces;
    # inc_z_lo, inc_z_hi for z-faces)
    if eta < 0.0:
        for j in range(Iz):
            psi_r_face[j, 0] = inc_r_hi[j, 0]
            psi_r_face[j, 1] = inc_r_hi[j, 1]
    else:
        for j in range(Iz):
            psi_r_face[j, 0] = inc_r_lo[j, 0]
            psi_r_face[j, 1] = inc_r_lo[j, 1]

    if mu >= 0.0:
        for i in range(Ir):
            psi_z_face[i, 0] = inc_z_lo[i, 0]
            psi_z_face[i, 1] = inc_z_lo[i, 1]
    else:
        for i in range(Ir):
            psi_z_face[i, 0] = inc_z_hi[i, 0]
            psi_z_face[i, 1] = inc_z_hi[i, 1]

    # s_k = 0.5 (symmetric product quadrature default)
    s_k = 0.5

    i = r_start
    while (r_step > 0 and i < r_end) or (r_step < 0 and i > r_end):
        axis_cell = (on_axis and i == 0)
        j = z_start
        while (z_step > 0 and j < z_end) or (z_step < 0 and j > z_end):
            I0, I1, I2, I3 = _solve_cyl_cell_corners(
                eta, mu, dr_arr[i], dz_arr[j],
                r_lo_arr[i], r_hi_arr[i], r_ctr[i],
                sigma_hat[i, j, 0], sigma_hat[i, j, 1],
                sigma_hat[i, j, 2], sigma_hat[i, j, 3],
                source[i, j, 0], source[i, j, 1],
                source[i, j, 2], source[i, j, 3],
                psi_r_face[j, 0], psi_r_face[j, 1],
                psi_z_face[i, 0], psi_z_face[i, 1],
                beta_lo, beta_hi, delta_lam,
                ang_edge_in[i, j, 0], ang_edge_in[i, j, 1],
                ang_edge_in[i, j, 2], ang_edge_in[i, j, 3],
                1 if axis_cell else 0)

            if I0 < 0.0: I0 = 0.0
            if I1 < 0.0: I1 = 0.0
            if I2 < 0.0: I2 = 0.0
            if I3 < 0.0: I3 = 0.0

            psi[i, j, 0] = I0
            psi[i, j, 1] = I1
            psi[i, j, 2] = I2
            psi[i, j, 3] = I3

            # Reconstruct outgoing angular edge via weighted DD:
            #   I_{k+1/2} = (I_c - (1-s)*I_{k-1/2}) / s
            if delta_lam > 1e-20:
                inv_s = 1.0 / s_k
                ang_edge_out[i, j, 0] = (I0 - (1.0-s_k)*ang_edge_in[i, j, 0]) * inv_s
                ang_edge_out[i, j, 1] = (I1 - (1.0-s_k)*ang_edge_in[i, j, 1]) * inv_s
                ang_edge_out[i, j, 2] = (I2 - (1.0-s_k)*ang_edge_in[i, j, 2]) * inv_s
                ang_edge_out[i, j, 3] = (I3 - (1.0-s_k)*ang_edge_in[i, j, 3]) * inv_s
                # Positivity clamp on angular edge
                if ang_edge_out[i, j, 0] < 0.0: ang_edge_out[i, j, 0] = 0.0
                if ang_edge_out[i, j, 1] < 0.0: ang_edge_out[i, j, 1] = 0.0
                if ang_edge_out[i, j, 2] < 0.0: ang_edge_out[i, j, 2] = 0.0
                if ang_edge_out[i, j, 3] < 0.0: ang_edge_out[i, j, 3] = 0.0

            # Propagate face values for spatial sweep
            if eta < 0.0:
                psi_r_face[j, 0] = I2
                psi_r_face[j, 1] = I1
            else:
                psi_r_face[j, 0] = I3
                psi_r_face[j, 1] = I0

            if mu >= 0.0:
                psi_z_face[i, 0] = I1
                psi_z_face[i, 1] = I0
            else:
                psi_z_face[i, 0] = I2
                psi_z_face[i, 1] = I3

            j += z_step
        i += r_step

    return psi, ang_edge_out


# ===========================================================================
# Full cylindrical sweep: all levels, all azimuthal directions
# ===========================================================================

def sweep_rz(Ir, Iz, dr_arr, dz_arr, r_lo_arr, r_hi_arr, r_ctr,
             sigma_hat, source_iso, levels,
             BCs_rlo, BCs_rhi, BCs_zlo, BCs_zhi,
             on_axis, weights_total):
    """Full cylindrical S_N sweep over all angular levels and azimuths.

    For each level ℓ:
      1. Sweep the starting direction g_ℓ.
      2. Set ang_edge = g_ℓ as I_{ℓ,1/2}.
      3. For k = 1 … K_ℓ: sweep direction (ℓ,k), reconstruct I_{ℓ,k+1/2}.
      4. Accumulate phi += w_{ℓ,k} * I_{ℓ,k}.

    Parameters
    ----------
    source_iso : (Ir, Iz, 4) — isotropic source (same for all angles)
    BCs_rlo : (Iz, 2) — incoming on r=r_min face [j, (bottom,top)]
    BCs_rhi : (Iz, 2) — incoming on r=r_max face
    BCs_zlo : (Ir, 2) — incoming on z=z_min face [i, (low-r,high-r)]
    BCs_zhi : (Ir, 2) — incoming on z=z_max face
    weights_total : used for normalization check only

    Returns
    -------
    phi : (Ir, Iz, 4)
    psi_all : list of (Ir, Iz, 4) per total angle (flattened)
    g_all   : list of (Ir, Iz, 4) starting-direction intensities per level
    """
    phi = np.zeros((Ir, Iz, 4))
    psi_all = []
    g_all = []

    inc_r_lo = np.zeros((Iz, 2))   # vacuum default
    inc_r_hi = np.zeros((Iz, 2))
    inc_z_lo = np.zeros((Ir, 2))
    inc_z_hi = np.zeros((Ir, 2))

    if BCs_rlo is not None:
        inc_r_lo = BCs_rlo
    if BCs_rhi is not None:
        inc_r_hi = BCs_rhi
    if BCs_zlo is not None:
        inc_z_lo = BCs_zlo
    if BCs_zhi is not None:
        inc_z_hi = BCs_zhi

    for lv in levels:
        mu_l = lv['mu']
        rho_l = lv['rho']
        eta_arr = lv['eta']
        b_arr = lv['b']
        beta = lv['beta']
        w_lk = lv['w_lk']
        K_l = len(eta_arr)

        # 1. Starting-direction sweep (η=-ρ, ξ=0, no angular coupling).
        # Pass BCs in geometric order (lo=r_min, hi=r_max, zlo=z_min, zhi=z_max);
        # _solve_starting_direction picks the correct face based on eta/mu sign.
        g_l = _solve_starting_direction(
            Ir, Iz, dr_arr, dz_arr, r_lo_arr, r_hi_arr, r_ctr,
            rho_l, mu_l, sigma_hat, source_iso,
            inc_r_lo,   # r=r_min boundary
            inc_r_hi,   # r=r_max boundary
            inc_z_lo,   # z=z_min boundary
            inc_z_hi,   # z=z_max boundary
            1 if on_axis else 0)
        g_all.append(g_l)

        # 2. I_{ℓ,1/2} = g_ℓ (eq 11.59 → eq at end of 11.5.6)
        ang_edge = g_l.copy()

        # 3. Sweep each azimuthal direction within this level
        for k in range(K_l):
            eta_k = eta_arr[k]
            beta_lo_k = beta[k]
            beta_hi_k = beta[k + 1]
            delta_lam_k = math.pi * b_arr[k]

            # Axis condition (eq 11.5.7): if on_axis and η > 0 (outward),
            # seed incoming r-face at r=0 from g_ℓ (the starting direction value)
            if on_axis and eta_k > 0.0:
                # Incoming from axis for outward-r direction: use g_l[:, 0] (low-r face)
                inc_r_lo_k = np.zeros((Iz, 2))
                for jj in range(Iz):
                    inc_r_lo_k[jj, 0] = g_l[0, jj, 2]   # SW corner at i=0
                    inc_r_lo_k[jj, 1] = g_l[0, jj, 1]   # NW corner at i=0
            else:
                inc_r_lo_k = inc_r_lo

            # Pass z BCs in geometric order; _sweep_one_azimuthal picks the
            # correct face (inc_z_lo for mu>=0, inc_z_hi for mu<0) internally.
            psi_k, ang_edge_new = _sweep_one_azimuthal(
                Ir, Iz, dr_arr, dz_arr, r_lo_arr, r_hi_arr, r_ctr,
                eta_k, mu_l, sigma_hat, source_iso,
                beta_lo_k, beta_hi_k, delta_lam_k,
                ang_edge,
                inc_r_lo_k, inc_r_hi,
                inc_z_lo,   # z=z_min boundary
                inc_z_hi,   # z=z_max boundary
                1 if on_axis else 0)

            ang_edge = ang_edge_new

            # 4. Accumulate scalar flux: phi += w_lk * psi_k
            # Weights sum to 1 over all levels and azimuths (full sphere).
            phi += w_lk[k] * psi_k
            psi_all.append(psi_k)

    return phi, psi_all, g_all


# ===========================================================================
# Richardson fallback solver
# ===========================================================================

def _find_z_mirror_levels(levels, tol=1e-8):
    """Return list mapping level_idx to its z-mirror level_idx.

    For z-reflection, the mirror of level ℓ with axial cosine μ_ℓ is the
    level ℓ' with μ_ℓ' = −μ_ℓ.  Product quadrature always has exact pairs;
    LS/EW quadratures are symmetric by construction.

    Returns a list mirror[i] = j such that levels[j]['mu'] ≈ −levels[i]['mu'].
    If no mirror exists (e.g. μ = 0), mirror[i] = i.
    """
    N = len(levels)
    mirror = list(range(N))
    for i in range(N):
        mu_i = levels[i]['mu']
        for j in range(N):
            if j != i and abs(levels[j]['mu'] + mu_i) < tol:
                mirror[i] = j
                break
    return mirror


def _extract_z_boundary_fluxes(psi_k, mu, Ir, Iz):
    """Extract outgoing z-boundary fluxes from an azimuthal sweep result.

    For μ > 0 (upward sweep): outgoing at z=z_max is carried by the high-z
    corners of the last z-row: NW(1) = low-r, NE(0) = high-r.
    For μ < 0 (downward sweep): outgoing at z=z_min is carried by the low-z
    corners of the first z-row: SW(2) = low-r, SE(3) = high-r.

    Returns (Ir, 2) array: [:, 0] = low-r corner, [:, 1] = high-r corner.
    """
    out = np.zeros((Ir, 2))
    if mu >= 0.0:
        out[:, 0] = psi_k[:, Iz - 1, 1]   # NW, low-r, last z-row
        out[:, 1] = psi_k[:, Iz - 1, 0]   # NE, high-r, last z-row
    else:
        out[:, 0] = psi_k[:, 0, 2]         # SW, low-r, first z-row
        out[:, 1] = psi_k[:, 0, 3]         # SE, high-r, first z-row
    return out


def sweep_rz(Ir, Iz, dr_arr, dz_arr, r_lo_arr, r_hi_arr, r_ctr,
             sigma_hat, source_iso, levels,
             BCs_rlo, BCs_rhi, BCs_zlo, BCs_zhi,
             on_axis, weights_total,
             reflect_zlo=False, reflect_zhi=False,
             n_reflect_its=2):
    """Full cylindrical S_N sweep over all angular levels and azimuths.

    For each level ℓ:
      1. Sweep the starting direction g_ℓ.
      2. Set ang_edge = g_ℓ as I_{ℓ,1/2}.
      3. For k = 1 … K_ℓ: sweep direction (ℓ,k), reconstruct I_{ℓ,k+1/2}.
      4. Accumulate phi += w_{ℓ,k} * I_{ℓ,k}.

    When reflect_zlo or reflect_zhi is True, the function performs
    n_reflect_its iterations of z-boundary reflection, building the
    incoming z-flux for each level from the outgoing flux of its μ-mirror
    level.  For optically thick (diffusive) problems 2 iterations suffice.

    Parameters
    ----------
    source_iso : (Ir, Iz, 4) — isotropic source (same for all angles)
    BCs_rlo : (Iz, 2) or None — external incoming on r=r_min
    BCs_rhi : (Iz, 2) or None — external incoming on r=r_max
    BCs_zlo : (Ir, 2) or None — external incoming on z=z_min
    BCs_zhi : (Ir, 2) or None — external incoming on z=z_max
    reflect_zlo : bool — reflecting BC at z=z_min
    reflect_zhi : bool — reflecting BC at z=z_max
    n_reflect_its : int — number of z-reflection passes

    Returns
    -------
    phi : (Ir, Iz, 4)
    psi_all : list of (Ir, Iz, 4) per total direction (flattened over levels)
    g_all   : list of (Ir, Iz, 4) starting-direction intensities per level
    """
    # Baseline incoming z BCs (external, non-reflected)
    inc_r_lo = np.zeros((Iz, 2)) if BCs_rlo is None else np.asarray(BCs_rlo, dtype=float)
    inc_r_hi = np.zeros((Iz, 2)) if BCs_rhi is None else np.asarray(BCs_rhi, dtype=float)
    inc_z_lo_ext = np.zeros((Ir, 2)) if BCs_zlo is None else np.asarray(BCs_zlo, dtype=float)
    inc_z_hi_ext = np.zeros((Ir, 2)) if BCs_zhi is None else np.asarray(BCs_zhi, dtype=float)

    # Per-direction incoming z BCs (mutable, updated by reflection)
    # Shape: (N_levels, K_max, Ir, 2) — but levels have different K.
    # Use a list of per-direction BC arrays indexed the same way psi_all is.
    reflect = reflect_zlo or reflect_zhi
    if reflect:
        z_mirror = _find_z_mirror_levels(levels)

    n_passes = n_reflect_its if reflect else 1

    # Per-direction z incoming BCs — start from external BCs
    total_dirs = sum(len(lv['eta']) for lv in levels)
    # inc_z_lo_dir[flat_dir_idx] = (Ir,2) for each direction
    inc_z_lo_dir = [inc_z_lo_ext.copy() for _ in range(total_dirs)]
    inc_z_hi_dir = [inc_z_hi_ext.copy() for _ in range(total_dirs)]

    phi = np.zeros((Ir, Iz, 4))
    psi_all = [None] * total_dirs
    g_all = []

    for _pass in range(n_passes):
        phi = np.zeros((Ir, Iz, 4))
        psi_all = [None] * total_dirs
        g_all = []
        flat_dir = 0

        for li, lv in enumerate(levels):
            mu_l  = lv['mu']
            rho_l = lv['rho']
            eta_arr = lv['eta']
            b_arr   = lv['b']
            beta    = lv['beta']
            w_lk    = lv['w_lk']
            K_l     = len(eta_arr)

            # Determine correct z BCs for this level based on μ sign
            # μ > 0 (upward): uses inc_z_lo.  μ < 0 (downward): uses inc_z_hi.
            # For reflecting BCs we need per-direction control, but the
            # starting direction uses the same μ_l so we use a representative
            # incoming from the first direction in the level.
            # For the starting direction, use the level-averaged reflection.
            if reflect:
                mirror_li = z_mirror[li]
                first_dir_flat = flat_dir
            else:
                mirror_li = li

            # Starting-direction sweep
            g_l = _solve_starting_direction(
                Ir, Iz, dr_arr, dz_arr, r_lo_arr, r_hi_arr, r_ctr,
                rho_l, mu_l, sigma_hat, source_iso,
                inc_r_lo, inc_r_hi,
                inc_z_lo_dir[flat_dir] if flat_dir < total_dirs else inc_z_lo_ext,
                inc_z_hi_dir[flat_dir] if flat_dir < total_dirs else inc_z_hi_ext,
                1 if on_axis else 0)
            g_all.append(g_l)

            ang_edge = g_l.copy()

            for k in range(K_l):
                eta_k      = eta_arr[k]
                beta_lo_k  = beta[k]
                beta_hi_k  = beta[k + 1]
                delta_lam_k = math.pi * b_arr[k]
                d_idx      = flat_dir

                # Axis condition
                if on_axis and eta_k > 0.0:
                    inc_r_lo_k = np.zeros((Iz, 2))
                    for jj in range(Iz):
                        inc_r_lo_k[jj, 0] = g_l[0, jj, 2]
                        inc_r_lo_k[jj, 1] = g_l[0, jj, 1]
                else:
                    inc_r_lo_k = inc_r_lo

                psi_k, ang_edge = _sweep_one_azimuthal(
                    Ir, Iz, dr_arr, dz_arr, r_lo_arr, r_hi_arr, r_ctr,
                    eta_k, mu_l, sigma_hat, source_iso,
                    beta_lo_k, beta_hi_k, delta_lam_k,
                    ang_edge,
                    inc_r_lo_k, inc_r_hi,
                    inc_z_lo_dir[d_idx],
                    inc_z_hi_dir[d_idx],
                    1 if on_axis else 0)

                psi_all[d_idx] = psi_k
                phi += w_lk[k] * psi_k
                flat_dir += 1

        # Build reflecting incoming z BCs from this pass's outgoing fluxes
        if reflect and _pass < n_passes - 1:
            # Map each direction's outgoing z flux to its mirror direction's incoming
            dir_per_level = [len(lv['eta']) for lv in levels]
            level_start   = [sum(dir_per_level[:i]) for i in range(len(levels))]

            for li, lv in enumerate(levels):
                mu_l       = lv['mu']
                mirror_li  = z_mirror[li]
                mirror_mu  = levels[mirror_li]['mu']
                K_l        = len(lv['eta'])
                K_m        = len(levels[mirror_li]['eta'])

                for k in range(min(K_l, K_m)):
                    d_idx        = level_start[li] + k
                    mirror_d_idx = level_start[mirror_li] + k

                    if psi_all[d_idx] is None:
                        continue

                    z_bnd = _extract_z_boundary_fluxes(psi_all[d_idx], mu_l, Ir, Iz)

                    if reflect_zhi and mu_l > 0.0:
                        # Outgoing z_hi from μ>0 → incoming z_hi for μ<0 mirror
                        inc_z_hi_dir[mirror_d_idx] = np.maximum(
                            inc_z_hi_ext, z_bnd)
                    if reflect_zlo and mu_l < 0.0:
                        # Outgoing z_lo from μ<0 → incoming z_lo for μ>0 mirror
                        inc_z_lo_dir[mirror_d_idx] = np.maximum(
                            inc_z_lo_ext, z_bnd)

    return phi, psi_all, g_all


# ===========================================================================
# Richardson fallback solver
# ===========================================================================

def _richardson_solve_rz(matvec, b, x, max_its, L2_tol, Linf_tol, LOUD):
    total_its = 0
    for _ in range(max_its):
        x_new = matvec(x) + b
        L2err = np.sqrt(np.mean(((x - x_new) / (np.abs(x_new) + 1e-14))**2))
        Linferr = np.max(np.abs(x - x_new) / (np.max(np.abs(x_new)) + 1e-14))
        x = x_new
        total_its += 1
        if LOUD:
            print(f"  Richardson {total_its}: L2={L2err:.3e}  Linf={Linferr:.3e}")
        if L2err < L2_tol and Linferr < Linf_tol:
            break
    return x, total_its

def temp_solve_rz(
    Ir, Iz, dr_arr, dz_arr, r_faces, z_faces,
    q_ext, sigma_func, scat_func,
    quad_type, N_quad,
    BCs_func, EOS, invEOS,
    phi_init, T_init,
    dt_min=1e-5, dt_max=0.001, tfinal=1.0,
    Linf_tol=1e-5, tolerance=1e-8, maxits=100,
    LOUD=False, K=50, R=3,
    time_outputs=None,
    reflect_rlo=False, reflect_rhi=False,
    reflect_zlo=False, reflect_zhi=False,
    print_stride=0,
    use_dmd=True,
    W=0,
    tau_T=1e-6,
    omega_T=1.0,
    T_floor=1e-10,
    store_full_history=True,
    step_callback=None,
):
    r"""Time-dependent gray TRT in 2-D axisymmetric cylindrical (r-z) geometry.

    Uses the cylindrical simple corner-balance S_N discretization from
    Chapter 11.5 with Fleck-Cummings linearization and optional DMD
    acceleration.

    Parameters
    ----------
    Ir, Iz : int
        Number of cells in r and z.
    dr_arr : (Ir,) float64
        Cell widths in r.
    dz_arr : (Iz,) float64
        Cell widths in z.
    r_faces : (Ir+1,) float64
        Radial face positions (r_{i-1/2}).
    z_faces : (Iz+1,) float64
        Axial face positions (z_{j-1/2}).
    q_ext : (Ir, Iz, 4) or callable(t) → (Ir, Iz, 4)
        External isotropic volumetric source.
    sigma_func : callable(T) → (Ir, Iz, 4)
        Absorption opacity.
    scat_func : callable(T) → (Ir, Iz, 4)
        Scattering opacity.
    quad_type : str
        Quadrature type for `get_rz_quadrature`.
    N_quad : int
        Quadrature order.
    BCs_func : callable(t) → dict
        Keys 'rlo', 'rhi', 'zlo', 'zhi'. Each value is (cells, 2) or None.
    EOS, invEOS : callable
        Equation of state and its inverse.
    phi_init : (Ir, Iz, 4) float64
    T_init   : (Ir, Iz, 4) float64
    dt_min, dt_max : float
    tfinal : float
    tolerance, Linf_tol : float
    maxits : int
    K, R : int  DMD parameters.
    reflect_rlo/rhi/zlo/zhi : bool  Reflecting BC flags.
    W : int  T_star outer iterations (0 = single linearization).
    store_full_history : bool
    step_callback : callable(t, phi, T) or None

    Returns
    -------
    phis : list of (Ir, Iz, 4)
    Ts   : list of (Ir, Iz, 4)
    iterations : int
    ts   : ndarray
    its_per_step : list
    """
    # --- Quadrature ---
    levels = get_rz_quadrature(quad_type, N_quad)
    N_total_angles = sum(len(lv['eta']) for lv in levels)
    w_total = sum(lv['w_lk'].sum() for lv in levels)

    # --- Mesh geometry ---
    r_lo_arr = np.ascontiguousarray(r_faces[:-1], dtype=np.float64)
    r_hi_arr = np.ascontiguousarray(r_faces[1:], dtype=np.float64)
    r_ctr = 0.5 * (r_lo_arr + r_hi_arr)
    on_axis = (r_faces[0] < 1e-15)

    dr_arr = np.ascontiguousarray(dr_arr, dtype=np.float64)
    dz_arr = np.ascontiguousarray(dz_arr, dtype=np.float64)

    # --- State ---
    phi = phi_init.copy()
    T = T_init.copy()
    T_old = T.copy()
    T_old2 = T.copy()
    e_old = EOS(T)

    # --- Storage ---
    phis = [phi.copy()]
    Ts   = [T.copy()]
    ts   = [0.0]
    its_per_step = []
    iterations = 0

    t_current = 0.0
    step_num = 0
    dt = dt_min
    dt_old = dt_min
    deriv_val = 0.0
    delta_step = 1e-3
    curr_step = 0
    t_output_index = 0
    _T_max = 1e50

    if time_outputs is not None:
        time_outputs = np.asarray(time_outputs, dtype=float)

    if step_callback is not None:
        step_callback(0.0, phi, T)

    print(f"2D-S_N r-z Cylindrical: Ir={Ir}, Iz={Iz}, "
          f"quad={quad_type} N={N_quad}, angles={N_total_angles}, "
          f"levels={len(levels)}")
    print(f"  dt range: [{dt_min:.2e}, {dt_max:.2e}], tfinal={tfinal}")
    print(f"  tolerances: L2={tolerance:.1e}, Linf={Linf_tol:.1e}, maxits={maxits}")
    print(f"  on_axis={on_axis}, reflect: rlo={reflect_rlo} rhi={reflect_rhi} "
          f"zlo={reflect_zlo} zhi={reflect_zhi}")
    print("|", end="", flush=True)

    while t_current < tfinal:
        dt_old2 = dt_old
        dt_old = dt
        step_num += 1

        # Adaptive time step
        if step_num > 2:
            dt_prop = math.sqrt(delta_step * deriv_val) if deriv_val > 0 else dt_max
            dt_prop = max(dt_min, min(dt_max, dt_prop))
            if dt_prop > 2.0 * dt:
                dt_prop = dt * 1.5
            dt = dt_prop
        else:
            dt = dt_min

        # Snap to final / output times
        if (tfinal - t_current) < dt:
            snap = tfinal - t_current
            if snap > 1e-10 * dt_min:
                dt = snap
            else:
                break

        if time_outputs is not None:
            out_tol = max(1e-12, 1e-6 * max(abs(t_current), dt_min))
            while (t_output_index < len(time_outputs) and
                   time_outputs[t_output_index] <= t_current + out_tol):
                t_output_index += 1

        if time_outputs is not None and t_output_index < len(time_outputs):
            if t_current + dt > time_outputs[t_output_index]:
                snap_dt = time_outputs[t_output_index] - t_current
                if snap_dt > 1e-6 * dt_min:
                    dt = snap_dt
                    t_output_index += 1

        if math.isnan(dt):
            dt = dt_min

        t_current += dt
        if int(10 * t_current / tfinal) > curr_step:
            curr_step += 1
            print(curr_step, end="", flush=True)

        icdt = 1.0 / (c * dt)
        iterations_step = 0

        # BCs at mid-step time
        bc_data = BCs_func(t_current - dt / 2.0)
        BCs_rlo_use = None
        BCs_rhi_use = None
        BCs_zlo_use = None
        BCs_zhi_use = None
        if bc_data is not None:
            if bc_data.get('rlo') is not None:
                BCs_rlo_use = np.ascontiguousarray(bc_data['rlo'], dtype=np.float64)
            if bc_data.get('rhi') is not None:
                BCs_rhi_use = np.ascontiguousarray(bc_data['rhi'], dtype=np.float64)
            if bc_data.get('zlo') is not None:
                BCs_zlo_use = np.ascontiguousarray(bc_data['zlo'], dtype=np.float64)
            if bc_data.get('zhi') is not None:
                BCs_zhi_use = np.ascontiguousarray(bc_data['zhi'], dtype=np.float64)

        # External source
        if callable(q_ext):
            q_now = q_ext(t_current)
        else:
            q_now = q_ext

        # T_star outer loop
        T_star = np.minimum(T_old, _T_max)
        do_final = False
        k_outer = 0
        x_sol = phi.ravel()

        while True:
            h_cv = np.maximum(np.abs(T_star) * 1e-4, 1e-12)
            Cv = np.maximum(
                (EOS(T_star + h_cv) - EOS(np.maximum(T_star - h_cv, T_floor))) / (2.0 * h_cv),
                1e-30)
            beta_fc = 4.0 * a * T_star**3 / Cv
            sigma = sigma_func(T_star)
            scat = scat_func(T_star)
            f = 1.0 / (1.0 + beta_fc * c * dt * sigma)
            sigma_a = f * sigma
            sigma_s = (1.0 - f) * sigma + scat
            sigma_t = sigma + scat
            sigma_hat = sigma_t + icdt
            emission = sigma_a * ac * T_star**4
            delta_e_src = -(1.0 - f) * (EOS(T_star) - e_old) / dt

            # Isotropic source for the sweep
            # The time-derivative term (icdt * psi_old) is handled
            # implicitly by adding icdt to sigma_hat and modifying the RHS.
            # Since psi_old is initialised isotropically from phi_old,
            # we use source_iso = emission + delta_e + q + icdt * phi_old/4π
            # (the 1/4π factor gives the isotropic part; summing over 4π gives phi_old).
            # For the cylindrical solver we fold the time derivative into an
            # isotropic augmented source using the previous step's phi.
            source_iso = emission + delta_e_src + q_now + icdt * phi

            sigma_hat_arr = np.ascontiguousarray(sigma_hat)

            # --- Source iteration / DMD ---
            tau_phi = tolerance if (W == 0 or do_final) else (
                tolerance * (1e-3 / tolerance) ** (1.0 - k_outer / max(W, 1)))

            def _matvec_rz(phi_vec):
                phi_3d = phi_vec.reshape((Ir, Iz, 4))
                scatter = np.ascontiguousarray(sigma_s * phi_3d)
                phi_new, _, _ = sweep_rz(
                    Ir, Iz, dr_arr, dz_arr, r_lo_arr, r_hi_arr, r_ctr,
                    sigma_hat_arr, scatter, levels,
                    None, None, None, None,    # zero BCs for scattering matvec
                    on_axis, w_total)
                return phi_new.ravel()

            # Fixed source sweep (reflecting z BCs applied to fixed source only;
            # scattering matvec uses vacuum z so reflection is folded in through phi)
            phi_b, _, _ = sweep_rz(
                Ir, Iz, dr_arr, dz_arr, r_lo_arr, r_hi_arr, r_ctr,
                sigma_hat_arr,
                np.ascontiguousarray(source_iso), levels,
                BCs_rlo_use, BCs_rhi_use, BCs_zlo_use, BCs_zhi_use,
                on_axis, w_total,
                reflect_zlo=reflect_zlo, reflect_zhi=reflect_zhi)
            b_vec = phi_b.ravel()

            if use_dmd:
                try:
                    x_sol, total_its, *_ = _solver_with_dmd_inc_1d(
                        matvec=_matvec_rz, b=b_vec, K=K, max_its=maxits,
                        steady=1, x=x_sol, Rits=R, LOUD=LOUD,
                        L2_tol=tau_phi, Linf_tol=Linf_tol)
                    if not np.all(np.isfinite(x_sol)):
                        raise np.linalg.LinAlgError("non-finite")
                except np.linalg.LinAlgError:
                    x_sol, total_its = _richardson_solve_rz(
                        _matvec_rz, b_vec, x_sol, maxits, tau_phi, Linf_tol, LOUD)
            else:
                x_sol, total_its = _richardson_solve_rz(
                    _matvec_rz, b_vec, x_sol, maxits, tau_phi, Linf_tol, LOUD)

            iterations += total_its
            iterations_step += total_its
            phi = x_sol.reshape((Ir, Iz, 4))

            if W == 0 or do_final:
                break

            delta_e_star = EOS(T_star) - e_old
            e_cand = (e_old + sigma_a * dt * (phi - ac * T_star**4)
                      + (1.0 - f) * delta_e_star)
            T_cand = invEOS(e_cand)
            T_star_prev = T_star.copy()
            T_star = np.clip((1.0 - omega_T) * T_star + omega_T * T_cand,
                             T_floor, _T_max)

            eta_T = float(np.sqrt(np.mean(
                ((T_star - T_star_prev) / (np.abs(T_star_prev) + T_floor))**2)))
            k_outer += 1
            if eta_T < tau_T or k_outer >= W:
                do_final = True

        # Material energy update
        delta_e_star = EOS(T_star) - e_old
        e = (e_old + sigma_a * dt * (phi - ac * T_star**4)
             + (1.0 - f) * delta_e_star)
        T = invEOS(e)

        dT_max = float(np.max(np.abs(T - T_old)))

        if print_stride > 0 and step_num % print_stride == 0:
            print(f"  step {step_num:5d}  t={t_current:.4e} ns  dt={dt:.3e}  "
                  f"T_max={np.max(T):.6f} keV  dT_max={dT_max:.3e}  "
                  f"sweeps={iterations_step}  T_iters={k_outer}")
        elif step_num <= 3 or (step_num <= 20 and step_num % 5 == 0):
            print(f"  step {step_num:5d}  t={t_current:.4e} ns  dt={dt:.3e}  "
                  f"T_max={np.max(T):.6f} keV  dT_max={dT_max:.3e}  "
                  f"sweeps={iterations_step}  T_iters={k_outer}")

        # Adaptive dt
        if step_num >= 2:
            denom = np.mean(np.abs(2.0 * (
                T.ravel() / (dt * (dt + dt_old))
                - T_old.ravel() / (dt * dt_old)
                + T_old2.ravel() / (dt_old * (dt + dt_old)))))
            mean_T = float(np.mean(T))
            if denom > 0 and math.isfinite(mean_T) and math.isfinite(denom):
                deriv_val = mean_T / denom
            else:
                deriv_val = dt_max**2 / delta_step

        e_old = EOS(T).copy()
        T_old2 = T_old.copy()
        T_old = T.copy()
        its_per_step.append(iterations_step)

        if step_callback is not None:
            step_callback(t_current, phi, T)

        store_step = store_full_history
        if not store_step:
            store_tol = max(1e-12, 1e-10 * max(abs(t_current), dt_min))
            at_final = abs(t_current - tfinal) <= store_tol
            at_output = (time_outputs is not None and
                         np.any(np.abs(time_outputs - t_current) <= store_tol))
            store_step = at_final or at_output

        if store_step:
            ts.append(t_current)
            phis.append(phi.copy())
            Ts.append(T.copy())

    print(f"\n  Finished: {step_num} steps, {iterations} total sweeps, "
          f"avg {iterations/max(step_num,1):.1f} sweeps/step")
    return phis, Ts, iterations, np.array(ts), its_per_step
