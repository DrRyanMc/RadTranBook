"""
1-D Discrete Ordinates (S_N) solver for spherical geometry using Linear
Discontinuous (LD) Galerkin elements with mass lumping.

Implements the spherical LDG sweep described in Chapter 10, Sections 10.6–10.7,
of the textbook.  The angular derivative in spherical geometry couples
neighbouring ordinates through the metric coefficients α, so the sweep is
sequential in angle (not parallel), proceeding:

    starting direction (μ = −1)
    → negative ordinates n = 0 … N//2−1  (μ < 0, inward)
    → origin condition  I_n(r=0) = g(r=0)  for μ > 0
    → positive ordinates n = N//2 … N−1  (μ > 0, outward)

The spatial discretisation (per ordinate) is the same 2×2 LD cell solve as
the slab version, but with additional geometric-moment weights M_{j,l/r}
(eqs 10.189/10.190) and angular-edge contributions κ/η (eqs 10.199–10.201).

A separate starting-direction intensity g(r, −1, t) is tracked alongside ψ.

Array shapes
------------
- phi, T, e, sigma  : (I, 2)  — 0 = left edge, 1 = right edge per cell
- psi               : (I, N, 2)
- g                 : (I, 2)  — starting-direction intensity I(r, μ=−1, t)
- source_n          : (I, N, 2)
- source_g          : (I, 2)
- bc_outer          : (N, 2)  — [n, 0] inflow from outer boundary (μ < 0)
                                [n, 1] unused (inner = origin condition)

Radial mesh
-----------
The mesh is specified by two 1-D arrays of length I:
  r_left[j]  — left (inner) edge of cell j  (r_{j-1/2})
  dr[j]      — cell width  Δr_j

so  r_right[j] = r_left[j] + dr[j]  (= r_{j+1/2}).

Quadrature normalisation
------------------------
_get_quadrature returns W normalised to sum to 1, which equals the spherical
weight w_n.  The angular cell width is Δμ_n = 2 w_n.

Physical constants are in CGS units with time in nanoseconds.

DMD acceleration utilities are imported from sn_solver.
"""

import math
import numpy as np
from numba import jit

from sn_solver import (
    c, a, ac,
    _get_quadrature,
    solver_with_dmd_inc,
)


# ---------------------------------------------------------------------------
# Module-level caches
# ---------------------------------------------------------------------------

_sph_quad_cache = {}     # key: N  → (mu_edges, alpha, s)


# ---------------------------------------------------------------------------
# Phase 1 — Quadrature helpers
# ---------------------------------------------------------------------------

def _compute_sph_quad_data(N):
    """Compute and cache spherical S_N angular data for N ordinates.

    Uses the Gauss-Legendre quadrature from ``_get_quadrature`` (W sums to 1,
    equal to the spherical weight w_n).

    Returns
    -------
    mu_edges : (N+1,)  float64
        Angular cell edge ordinates μ_{n−1/2}.  ``mu_edges[0] = −1``,
        ``mu_edges[N] = +1``.
    alpha : (N+1,)  float64
        Angular metric coefficients α_{n+1/2}.  ``alpha[0] = 0``,
        ``alpha[N] = 0`` (for symmetric quadratures).
        Defined by the recursion (eq 10.163):
            α_{n+1/2} = α_{n−1/2} − 4 μ_n w_n
    s : (N,)  float64
        Reconstruction weights for each ordinate:
            s_n = (μ_n − μ_{n−1/2}) / (μ_{n+1/2} − μ_{n−1/2}) = (μ_n − μ_{n−1/2}) / (2 w_n)
        Note s_n ∈ (0, 1).
    """
    if N not in _sph_quad_cache:
        MU, W = _get_quadrature(N)  # W sums to 1
        mu_edges = np.empty(N + 1, dtype=np.float64)
        mu_edges[0] = -1.0
        for n in range(N):
            mu_edges[n + 1] = mu_edges[n] + 2.0 * W[n]  # Δμ_n = 2 w_n
        # Force exact closure (floating-point drift correction)
        mu_edges[N] = 1.0

        alpha = np.empty(N + 1, dtype=np.float64)
        alpha[0] = 0.0
        for n in range(N):
            alpha[n + 1] = alpha[n] - 4.0 * MU[n] * W[n]
        # Should be 0 for symmetric quadrature; enforce exactly
        alpha[N] = 0.0

        s = np.empty(N, dtype=np.float64)
        for n in range(N):
            dmu = mu_edges[n + 1] - mu_edges[n]   # = 2 w_n
            s[n] = (MU[n] - mu_edges[n]) / dmu

        _sph_quad_cache[N] = (
            np.ascontiguousarray(mu_edges),
            np.ascontiguousarray(alpha),
            np.ascontiguousarray(s),
        )
    return _sph_quad_cache[N]


def _compute_geometric_moments(r_left, dr):
    """Compute LD geometric moments M_{j,l} and M_{j,r} (eqs 10.189/10.190).

    For cell j with r_{j,L} = r_left[j] and Δr_j = dr[j]:

        M_{j,l} = r_{j,L}²/2 + r_{j,L}·Δr_j/3 + Δr_j²/12
        M_{j,r} = r_{j,L}²/2 + 2·r_{j,L}·Δr_j/3 + Δr_j²/4

    These are ∫₀¹ r(ξ)² b_1(ξ) dξ and ∫₀¹ r(ξ)² b_2(ξ) dξ respectively,
    with r(ξ) = r_{j,L} + Δr_j ξ.

    Parameters
    ----------
    r_left : (I,)  float64
        Left-edge radii of each cell.
    dr : (I,)  float64
        Cell widths.

    Returns
    -------
    M_l, M_r : (I,)  float64
    """
    rL = r_left
    dz = dr
    M_l = rL**2 / 2.0 + rL * dz / 3.0 + dz**2 / 12.0
    M_r = rL**2 / 2.0 + 2.0 * rL * dz / 3.0 + dz**2 / 4.0
    return (np.ascontiguousarray(M_l, dtype=np.float64),
            np.ascontiguousarray(M_r, dtype=np.float64))


# ---------------------------------------------------------------------------
# Phase 2 — JIT sweep kernel (sequential in angle)
# ---------------------------------------------------------------------------

@jit(nopython=True, cache=True)
def _sweep_all_sph_ld(
    I, r_left, dr, source_n, source_g, sigma_hat,
    MU, W, mu_edges, alpha, s, M_l, M_r,
    bc_outer, bc_g_outer, bc_inner, full_sphere, fix,
):
    """Full spherical LD sweep; return phi (I,2), g_out (I,2), psi (I,N,2).

    Source convention: ``source_n[j,n,lr]`` and ``source_g[j,lr]`` are the
    Q̂ values already containing the ½ factors where the textbook has them.
    The cell-solve matrix coefficients use Δr·r²·σ̂ (no extra ½).

    Parameters
    ----------
    I : int
        Number of radial cells.
    r_left : (I,)  float64
        Left-edge radii r_{j−1/2}.
    dr : (I,)  float64
        Cell widths Δr_j.
    source_n : (I, N, 2)  float64
        Source Q̂_n at each edge and ordinate.
    source_g : (I, 2)  float64
        Source Q̂_{−1} for the starting direction.
    sigma_hat : (I, 2)  float64
        Effective total opacity σ̂ = σ_t + 1/(cΔt).
    MU : (N,)  float64
        Gauss-Legendre ordinates (W sums to 1).
    W : (N,)  float64
        Quadrature weights (sum = 1).
    mu_edges : (N+1,)  float64
        Angular cell edges μ_{n±1/2}.
    alpha : (N+1,)  float64
        Angular metric coefficients α_{n+1/2}.
    s : (N,)  float64
        Angular reconstruction weights s_n.
    M_l, M_r : (I,)  float64
        Geometric moments M_{j,l} and M_{j,r}.
    bc_outer : (N, 2)  float64
        Outer-boundary inflow intensities.  ``bc_outer[n, 0]`` is the inflow
        for μ_n < 0 (inward-moving ordinates) from the outer wall.
    bc_g_outer : float64
        Starting-direction inflow at the outer wall (μ = −1 from outside).
        For vacuum: 0.
    bc_inner : (N, 2)  float64
        Inner-boundary inflow intensities for μ_n > 0 (outward-moving
        ordinates entering from the inner wall).  Only used when
        ``full_sphere == 0``.  For vacuum inner wall use zeros.
    full_sphere : int
        1 if r_left[0] == 0 (full sphere — origin regularity condition
        ``I_n(r=0) = g(r=0)`` is applied automatically); 0 for a hollow
        shell (``bc_inner`` is used instead).
    fix : int
        If > 0 apply the conservative positivity fixup.

    Returns
    -------
    phi : (I, 2)  float64
    g_out : (I, 2)  float64
        Updated starting-direction intensity at cell edges.
    psi : (I, N, 2)  float64
    """
    N = MU.size
    psi = np.zeros((I, N, 2))
    phi = np.zeros((I, 2))

    # -----------------------------------------------------------------
    # Step A: Starting-direction solve  g(r, μ = −1, t)
    #   Governed by:  −∂g/∂r + σ̂ g = Q̂_{−1}
    #   (No geometric r-factors because (1−μ²)/r = 0 at μ = −1.)
    #
    #   Use the slab-like LD solve (sweep right-to-left):
    #       a00 = 0.5 + dz*sh_l,  a01 = -0.5
    #       a10 = 0.5,             a11 =  0.5 + dz*sh_r
    #       rhs0 = dz*Q_l,         rhs1 = dz*Q_r + g_in
    #
    #   The resulting g_l[0] = g(0,t) is then used as the radial upwind
    #   trace for all μ_n > 0 ordinates at the origin (Step E), though
    #   since rL²=0 at the origin, this trace enters rhs0 only through
    #   mu*rL²*I_in which vanishes, so it has no direct matrix effect.
    # -----------------------------------------------------------------
    g_l = np.zeros(I)
    g_r = np.zeros(I)

    g_in = bc_g_outer  # inflow at r = R_outer (j = I−1, right face)
    for j in range(I - 1, -1, -1):
        dz = dr[j]
        sh_l = sigma_hat[j, 0]
        sh_r = sigma_hat[j, 1]
        Ql = source_g[j, 0]
        Qr = source_g[j, 1]

        # LDG solve for -dg/dr + sigma_hat*g = Q_g
        a00g = 0.5 + dz * sh_l
        a01g = -0.5
        a10g = 0.5
        a11g = 0.5 + dz * sh_r
        rhs0g = dz * Ql
        rhs1g = dz * Qr + g_in
        detg = a00g * a11g - a01g * a10g
        gl_raw = (rhs0g * a11g - a01g * rhs1g) / detg
        gr_raw = (a00g * rhs1g - rhs0g * a10g) / detg
        # Conservative positivity fixup (slab-style)
        if fix > 0:
            gbar = 0.5 * (gl_raw + gr_raw)
            gmin = min(gl_raw, gr_raw)
            if gmin >= 0.0:
                gl = gl_raw
                gr = gr_raw
            elif gbar >= 0.0:
                theta = gbar / (gbar - gmin)
                gl = gbar + theta * (gl_raw - gbar)
                gr = gbar + theta * (gr_raw - gbar)
            else:
                gl = 0.0
                gr = 0.0
        else:
            gl = gl_raw
            gr = gr_raw
        g_l[j] = gl
        g_r[j] = gr
        g_in = g_l[j]   # left face of cell j becomes inflow for cell j-1

    g_out = np.empty((I, 2), dtype=np.float64)
    for j in range(I):
        g_out[j, 0] = g_l[j]
        g_out[j, 1] = g_r[j]

    # -----------------------------------------------------------------
    # Step B: Angular edge I_{1/2} = g  (α_{1/2} = 0; first angular edge)
    # I_half[j, lr] holds the current lower angular-edge value I_{n−1/2}
    # for each cell and side.
    # -----------------------------------------------------------------
    I_half_l = np.empty(I, dtype=np.float64)
    I_half_r = np.empty(I, dtype=np.float64)
    for j in range(I):
        I_half_l[j] = g_l[j]
        I_half_r[j] = g_r[j]

    # -----------------------------------------------------------------
    # Steps C & E: Angular sweep (negative then positive ordinates)
    # -----------------------------------------------------------------
    for n in range(N):
        mu = MU[n]
        wn = W[n]
        sn = s[n]
        alpha_lo = alpha[n]      # α_{n−1/2}
        alpha_hi = alpha[n + 1]  # α_{n+1/2}
        rL_arr = r_left
        dz_arr = dr

        psi_l = np.zeros(I)
        psi_r = np.zeros(I)

        if mu < 0.0:
            # ── Step C: negative ordinate, inward sweep (j = I−1 … 0) ──
            abs_mu = -mu
            I_in = bc_outer[n, 0]  # inflow from outer wall at r = R

            for j in range(I - 1, -1, -1):
                rL = rL_arr[j]
                rR = rL + dz_arr[j]
                dz = dz_arr[j]
                Ml = M_l[j]
                Mr = M_r[j]
                sh_l = sigma_hat[j, 0]
                sh_r = sigma_hat[j, 1]

                # r-weighted angular moments: R_l = int r(xi) b_1 dxi, R_r = int r(xi) b_2 dxi
                Rl = 0.5 * rL + dz / 6.0
                Rr = 0.5 * rL + dz / 3.0

                # κ_{n,j,l/r} and η_{n,j,l/r} using integrated r-moments (not endpoint lumping)
                kappa_l = dz * Rl / (2.0 * wn) * alpha_hi / sn
                kappa_r = dz * Rr / (2.0 * wn) * alpha_hi / sn
                eta_coeff_l = dz * Rl / (2.0 * wn) * (alpha_hi * (1.0 - sn) / sn + alpha_lo)
                eta_coeff_r = dz * Rr / (2.0 * wn) * (alpha_hi * (1.0 - sn) / sn + alpha_lo)
                eta_l = eta_coeff_l * I_half_l[j]
                eta_r = eta_coeff_r * I_half_r[j]

                Ql = source_n[j, n, 0]
                Qr = source_n[j, n, 1]

                # 2×2 system for μ < 0 using integrated moments
                # rL2 still needed in a00 (streaming); rR2 in upwind boundary term
                rL2 = rL * rL
                rR2 = rR * rR
                a00 = mu * (Ml - rL2) + dz * Ml * sh_l + kappa_l
                a01 = mu * Mr
                a10 = -mu * Ml
                a11 = -mu * Mr + dz * Mr * sh_r + kappa_r
                rhs0 = dz * Ml * Ql + eta_l
                rhs1 = dz * Mr * Qr + eta_r - mu * rR2 * I_in

                det = a00 * a11 - a01 * a10
                Il_raw = (rhs0 * a11 - a01 * rhs1) / det
                Ir_raw = (a00 * rhs1 - rhs0 * a10) / det

                # r²-weighted conservative fixup (eqs 10.211–10.214)
                if fix > 0:
                    I_bar_r2 = (Ml * Il_raw + Mr * Ir_raw) / (Ml + Mr)
                    I_min = min(Il_raw, Ir_raw)
                    if I_min >= 0.0:
                        psi_l[j] = Il_raw
                        psi_r[j] = Ir_raw
                    elif I_bar_r2 >= 0.0:
                        theta = I_bar_r2 / (I_bar_r2 - I_min)
                        psi_l[j] = I_bar_r2 + theta * (Il_raw - I_bar_r2)
                        psi_r[j] = I_bar_r2 + theta * (Ir_raw - I_bar_r2)
                    else:
                        psi_l[j] = 0.0
                        psi_r[j] = 0.0
                else:
                    psi_l[j] = Il_raw
                    psi_r[j] = Ir_raw

                I_in = psi_l[j]   # fixed-up left face → inflow for cell j−1

                # Update angular edge I_{n+1/2} via reconstruction (eqs 10.197/10.198)
                # I_{n+1/2,j,l/r} = (I_{n,j,l/r} − (1−s_n)·I_{n−1/2,j,l/r}) / s_n
                if sn > 1e-14:
                    I_half_l[j] = (psi_l[j] - (1.0 - sn) * I_half_l[j]) / sn
                    I_half_r[j] = (psi_r[j] - (1.0 - sn) * I_half_r[j]) / sn
                else:
                    I_half_l[j] = psi_l[j]
                    I_half_r[j] = psi_r[j]

        else:
            # ── Step E: positive ordinate, outward sweep (j = 0 … I−1) ──
            # Full sphere: at the origin (rL=0, rL²=0) the boundary term
            # mu*rL²*I_in vanishes naturally; the 2×2 LDG system is
            # non-degenerate due to the integrated moments Ml, Mr, Rl, Rr.
            # Hollow shell: prescribed inner-wall inflow bc_inner[n, 1].
            if full_sphere:
                I_in = g_l[0]
            else:
                I_in = bc_inner[n, 1]

            for j in range(I):
                rL = rL_arr[j]
                rR = rL + dz_arr[j]
                dz = dz_arr[j]
                Ml = M_l[j]
                Mr = M_r[j]
                sh_l = sigma_hat[j, 0]
                sh_r = sigma_hat[j, 1]

                # r-weighted angular moments: R_l = int r(xi) b_1 dxi, R_r = int r(xi) b_2 dxi
                Rl = 0.5 * rL + dz / 6.0
                Rr = 0.5 * rL + dz / 3.0

                # κ_{n,j,l/r} and η_{n,j,l/r} using integrated r-moments (not endpoint lumping)
                kappa_l = dz * Rl / (2.0 * wn) * alpha_hi / sn
                kappa_r = dz * Rr / (2.0 * wn) * alpha_hi / sn
                eta_coeff_l = dz * Rl / (2.0 * wn) * (alpha_hi * (1.0 - sn) / sn + alpha_lo)
                eta_coeff_r = dz * Rr / (2.0 * wn) * (alpha_hi * (1.0 - sn) / sn + alpha_lo)
                eta_l = eta_coeff_l * I_half_l[j]
                eta_r = eta_coeff_r * I_half_r[j]

                Ql = source_n[j, n, 0]
                Qr = source_n[j, n, 1]

                # 2×2 system for μ > 0 using integrated moments.
                # Boundary term mu*rL2*I_in enters rhs0; at origin rL2=0 so it
                # vanishes naturally — no special-case override needed.
                rL2 = rL * rL
                rR2 = rR * rR
                a00 = mu * Ml + dz * Ml * sh_l + kappa_l
                a01 = mu * Mr
                a10 = -mu * Ml
                a11 = mu * (rR2 - Mr) + dz * Mr * sh_r + kappa_r
                rhs0 = mu * rL2 * I_in + dz * Ml * Ql + eta_l
                rhs1 = dz * Mr * Qr + eta_r

                det = a00 * a11 - a01 * a10
                Il_raw = (rhs0 * a11 - a01 * rhs1) / det
                Ir_raw = (a00 * rhs1 - rhs0 * a10) / det

                # r²-weighted conservative fixup
                if fix > 0:
                    I_bar_r2 = (Ml * Il_raw + Mr * Ir_raw) / (Ml + Mr)
                    I_min = min(Il_raw, Ir_raw)
                    if I_min >= 0.0:
                        psi_l[j] = Il_raw
                        psi_r[j] = Ir_raw
                    elif I_bar_r2 >= 0.0:
                        theta = I_bar_r2 / (I_bar_r2 - I_min)
                        psi_l[j] = I_bar_r2 + theta * (Il_raw - I_bar_r2)
                        psi_r[j] = I_bar_r2 + theta * (Ir_raw - I_bar_r2)
                    else:
                        psi_l[j] = 0.0
                        psi_r[j] = 0.0
                else:
                    psi_l[j] = Il_raw
                    psi_r[j] = Ir_raw

                I_in = psi_r[j]   # fixed-up right face → inflow for cell j+1

                # Update angular edge I_{n+1/2}
                if sn > 1e-14:
                    I_half_l[j] = (psi_l[j] - (1.0 - sn) * I_half_l[j]) / sn
                    I_half_r[j] = (psi_r[j] - (1.0 - sn) * I_half_r[j]) / sn
                else:
                    I_half_l[j] = psi_l[j]
                    I_half_r[j] = psi_r[j]

        # Store angular flux and accumulate scalar flux phi = int_{-1}^{1} I dmu
        # W sums to 1 (not 2), so the physical integral needs factor 2.
        for j in range(I):
            psi[j, n, 0] = psi_l[j]
            psi[j, n, 1] = psi_r[j]
            phi[j, 0] += 2.0 * W[n] * psi_l[j]
            phi[j, 1] += 2.0 * W[n] * psi_r[j]

    return phi, g_out, psi


# ---------------------------------------------------------------------------
# Phase 3 — Public sweep wrappers
# ---------------------------------------------------------------------------

def single_sweep_phi_sph_ld(
    I, r_left, dr, source_n, source_g, sigma_hat, N,
    bc_outer, bc_g_outer, bc_inner=None, fix=1,
):
    """One sequential spherical LD sweep; return scalar flux phi (I, 2).

    Parameters
    ----------
    I : int
        Number of radial cells.
    r_left : (I,)  float64
        Left-edge radii.
    dr : (I,)  float64
        Cell widths.
    source_n : (I, N, 2)  float64
        Source Q̂_n at each edge and ordinate.
    source_g : (I, 2)  float64
        Source Q̂_{−1} for the starting direction.
    sigma_hat : (I, 2)  float64
        Effective total opacity σ̂.
    N : int
        Number of discrete ordinates.
    bc_outer : (N, 2)  float64
        Outer-boundary inflow.  ``bc_outer[n, 0]`` is used for μ_n < 0.
    bc_g_outer : float
        Starting-direction inflow at the outer wall.
    bc_inner : (N, 2)  float64 or None
        Inner-boundary inflow for μ_n > 0.  Pass ``None`` (default) for a
        full sphere (r_left[0] == 0); the origin regularity condition is
        applied automatically.  For a hollow shell pass the inflow array.
    fix : int
        Positivity fixup flag (default 1 = on).

    Returns
    -------
    phi : (I, 2)  float64
    """
    MU, W = _get_quadrature(N)
    mu_edges, alpha, s = _compute_sph_quad_data(N)
    M_l, M_r = _compute_geometric_moments(r_left, dr)
    r_left_c = np.ascontiguousarray(r_left, dtype=np.float64)
    full_sphere = 1 if r_left_c[0] < 1e-15 else 0
    if bc_inner is None:
        bc_inner_c = np.zeros((N, 2), dtype=np.float64)
    else:
        bc_inner_c = np.ascontiguousarray(bc_inner, dtype=np.float64)
    phi, _, _ = _sweep_all_sph_ld(
        I,
        r_left_c,
        np.ascontiguousarray(dr, dtype=np.float64),
        np.ascontiguousarray(source_n, dtype=np.float64),
        np.ascontiguousarray(source_g, dtype=np.float64),
        np.ascontiguousarray(sigma_hat, dtype=np.float64),
        MU, W, mu_edges, alpha, s, M_l, M_r,
        np.ascontiguousarray(bc_outer, dtype=np.float64),
        float(bc_g_outer),
        bc_inner_c,
        full_sphere,
        fix,
    )
    return phi


def single_sweep_psi_sph_ld(
    I, r_left, dr, source_n, source_g, sigma_hat, N,
    bc_outer, bc_g_outer, bc_inner=None, fix=1,
):
    """One sequential spherical LD sweep; return (psi, phi, g).

    Same parameters as ``single_sweep_phi_sph_ld``.

    Returns
    -------
    psi : (I, N, 2)  float64
    phi : (I, 2)  float64
    g   : (I, 2)  float64  — updated starting-direction intensity
    """
    MU, W = _get_quadrature(N)
    mu_edges, alpha, s = _compute_sph_quad_data(N)
    M_l, M_r = _compute_geometric_moments(r_left, dr)
    r_left_c = np.ascontiguousarray(r_left, dtype=np.float64)
    full_sphere = 1 if r_left_c[0] < 1e-15 else 0
    if bc_inner is None:
        bc_inner_c = np.zeros((N, 2), dtype=np.float64)
    else:
        bc_inner_c = np.ascontiguousarray(bc_inner, dtype=np.float64)
    phi, g, psi = _sweep_all_sph_ld(
        I,
        r_left_c,
        np.ascontiguousarray(dr, dtype=np.float64),
        np.ascontiguousarray(source_n, dtype=np.float64),
        np.ascontiguousarray(source_g, dtype=np.float64),
        np.ascontiguousarray(sigma_hat, dtype=np.float64),
        MU, W, mu_edges, alpha, s, M_l, M_r,
        np.ascontiguousarray(bc_outer, dtype=np.float64),
        float(bc_g_outer),
        bc_inner_c,
        full_sphere,
        fix,
    )
    return psi, phi, g


def build_reflecting_BCs_sph_ld(bcs, psi_old, N):
    """Fill incoming outer-boundary intensities from outgoing angular flux.

    For a reflecting outer wall the inward-moving ordinate n (μ_n < 0)
    receives the intensity that left the domain from the mirror ordinate
    n_ref = N−1−n (μ_{n_ref} > 0, right-face of the outermost cell).

    Parameters
    ----------
    bcs : (N, 2)
        Boundary array to update in-place.  Only ``bcs[n, 0]`` (for μ_n < 0)
        is modified; other entries retain their current values.
    psi_old : (I, N, 2)
        Previous-step angular flux.
    N : int
        Number of ordinates.

    Returns
    -------
    bcs : (N, 2)  updated in-place
    """
    MU, _ = _get_quadrature(N)
    for n in range(N):
        if MU[n] < 0.0:
            n_ref = N - 1 - n
            # Inflow for left-moving ray n comes from right face of last cell
            # for the mirror right-moving ray n_ref.
            bcs[n, 0] = psi_old[-1, n_ref, 1]
    return bcs


def build_reflecting_inner_BCs_sph_ld(bc_inner, psi_old, N):
    """Fill incoming inner-boundary intensities from outgoing angular flux.

    For a reflecting inner wall (hollow sphere) the outward-moving ordinate n
    (μ_n > 0) receives the intensity that exits the domain from the mirror
    inward-moving ordinate n_ref = N−1−n (μ_{n_ref} < 0, left-face of
    the innermost cell).

    Parameters
    ----------
    bc_inner : (N, 2)
        Inner boundary array to update in-place.  Only ``bc_inner[n, 1]``
        (for μ_n > 0) is modified.
    psi_old : (I, N, 2)
        Previous-step angular flux.
    N : int
        Number of ordinates.

    Returns
    -------
    bc_inner : (N, 2)  updated in-place
    """
    MU, _ = _get_quadrature(N)
    for n in range(N):
        if MU[n] > 0.0:
            n_ref = N - 1 - n
            # Inflow for right-moving ray n at inner wall comes from left face
            # of cell 0 for the mirror left-moving ray n_ref.
            bc_inner[n, 1] = psi_old[0, n_ref, 0]
    return bc_inner


# ---------------------------------------------------------------------------
# Phase 4 — Gray time-dependent TRT solver
# ---------------------------------------------------------------------------

def _richardson_solve(matvec, b, x0, max_its, L2_tol, Linf_tol, LOUD):
    """Pure Richardson (source) iteration; same return signature as solver_with_dmd_inc."""
    x = x0.copy()
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
    return x, total_its, np.array([]), np.array([]), [], [], []


def temp_solve_sph_ld(
    I, r_left, dr, q_n, q_g, sigma_func, scat_func, N, BCs, EOS, invEOS,
    phi, psi, T, g_init,
    dt_min=1e-5, dt_max=0.001, tfinal=1.0,
    Linf_tol=1.0e-5, tolerance=1.0e-8, maxits=100,
    LOUD=False, fix=1, K=100, R=3,
    time_outputs=None,
    reflect_outer=False,
    reflect_inner=False,
    BCs_inner=None,
    print_stride=0,
    use_dmd=True,
    W=0,
    tau_phi_max=0.1,
    tau_T=1e-6,
    C_T=1.0,
    omega_T=1.0,
    T_floor=1e-10,
):
    r"""Time-dependent gray TRT with spherical LD-S_N and DMD-accelerated source iteration.

    Implements the implicit linearised Fleck-Cummings scheme in spherical
    geometry using the LD spatial discretisation of Sections 10.6–10.7.

    The angular sweep is sequential, coupling ordinates through the metric
    coefficients α.  An additional starting-direction intensity
    ``g(r, −1, t)`` is tracked alongside ``psi``.

    Parameters
    ----------
    I : int
        Number of radial cells.
    r_left : (I,)  float64
        Left-edge radii of each cell.
    dr : (I,)  float64
        Cell widths Δr_j.
    q_n : (I, N, 2)  float64
        Fixed external source for ordinates (usually zero).
    q_g : (I, 2)  float64
        Fixed external source for the starting direction (usually zero).
    sigma_func : callable T → (I, 2)
        Absorption opacity as a function of temperature.
    scat_func : callable T → (I, 2)
        Scattering cross-section as a function of temperature.
    N : int
        Number of discrete ordinates.
    BCs : callable t → (bc_outer (N,2), bc_g_outer float)
        Returns outer-boundary inflows at time t.  ``bc_outer[n, 0]`` is the
        inflow for μ_n < 0; ``bc_g_outer`` is the starting-direction inflow.
        For a full sphere (r_left[0] == 0) this is all that is needed.
    EOS : callable T → (I, 2)
        Internal energy from temperature.
    invEOS : callable e → (I, 2)
        Temperature from internal energy.
    phi : (I, 2)
        Initial scalar flux.
    psi : (I, N, 2)
        Initial angular flux.
    T : (I, 2)
        Initial temperature.
    g_init : (I, 2)
        Initial starting-direction intensity g(r, −1, t=0).
        A good choice for near-equilibrium starts is ``ac * T_0**4 / 2``.
    dt_min, dt_max : float
        Adaptive time-step bounds.
    tfinal : float
        Final simulation time.
    Linf_tol, tolerance : float
        L∞ and L² convergence tolerances.
    maxits : int
        Maximum inner iterations per time step.
    LOUD : bool
        Verbosity flag.
    fix : int
        Positivity fixup flag (default 1 = on).
    K : int
        DMD history length.
    R : int
        Richardson iterations between DMD steps.
    time_outputs : ndarray or None
        Extra output times to hit exactly.
    reflect_outer : bool
        Reflecting wall at r = R_max.
    reflect_inner : bool
        Reflecting wall at the inner boundary r = r_left[0].  Only used when
        r_left[0] > 0 (hollow shell).  Ignored for full spheres.
    BCs_inner : callable t → (N, 2) or None
        Returns inner-boundary inflows for μ_n > 0 at time t.  Only used
        when r_left[0] > 0 and ``reflect_inner`` is False.  Pass ``None``
        for a vacuum inner wall (default) or a full sphere.
    print_stride : int
        Print diagnostics every this many steps (0 = off).
    use_dmd : bool
        Use DMD acceleration (True) or pure Richardson (False).
    W : int
        Number of T_star outer iterations (0 = single linearisation).
    tau_phi_max : float
        Loose inner tolerance for early T_star iterations.
    tau_T : float
        Convergence tolerance for T_star outer iteration.
    C_T : float
        Inexact-solve forcing factor for T_star iteration.
    omega_T : float
        Damping factor for T_star update (0 < ω ≤ 1).
    T_floor : float
        Temperature floor for EOS and T_star convergence checks.

    Returns
    -------
    phis : list of (I, 2)
        Scalar flux at each saved time.
    Ts : list of (I, 2)
        Temperature at each saved time.
    gs : list of (I, 2)
        Starting-direction intensity at each saved time.
    iterations : int
        Total transport sweeps.
    ts : ndarray
        Saved times.
    its_per_step : list of int
        Sweeps per time step.
    """
    r_left = np.ascontiguousarray(r_left, dtype=np.float64)
    dr_arr = np.ascontiguousarray(dr, dtype=np.float64)
    M_l, M_r = _compute_geometric_moments(r_left, dr_arr)
    MU, W_quad = _get_quadrature(N)
    mu_edges, alpha_arr, s_arr = _compute_sph_quad_data(N)
    _full_sphere = 1 if r_left[0] < 1e-15 else 0

    t_current = 0.0
    phis = [phi.copy()]
    Ts = [T.copy()]
    gs = [g_init.copy()]
    its_per_step = []

    psi_old = psi.copy()
    g_old = g_init.copy()
    T_old = T.copy()
    T_old2 = T.copy()
    e_old = EOS(T)

    ts = [t_current]
    step_num = 0
    dt_old = dt_min
    dt = dt_min
    deriv_val = 0.0
    delta_step = 1e-3
    curr_step = 0
    t_output_index = 0
    iterations = 0

    print(f"Spherical LD-S_N: I={I}, N={N}")
    print("|", end="")

    while t_current < tfinal:
        dt_old2 = dt_old
        dt_old = dt
        step_num += 1

        # ── adaptive time step ────────────────────────────────────────────
        if step_num > 2:
            dt_prop = np.sqrt(delta_step * deriv_val)
            if dt_prop > dt_max:
                dt_prop = dt_max
            if dt_prop < dt_min:
                dt_prop = dt_min
            if dt_prop > 2.0 * dt:
                dt_prop = dt * 1.5
            dt = dt_prop
        else:
            dt = dt_min
        if (tfinal - t_current) < dt:
            snap_to_final = tfinal - t_current
            if snap_to_final > 1e-10 * dt_min:
                dt = snap_to_final
            else:
                break
        try:
            if (time_outputs is not None and
                    t_current + dt > time_outputs[t_output_index] and
                    t_output_index < time_outputs.size):
                snap_dt = time_outputs[t_output_index] - t_current
                t_output_index += 1
                if snap_dt > 1e-10 * dt_min:
                    dt = snap_dt
        except Exception:
            pass
        if math.isnan(dt):
            dt = dt_min

        if LOUD:
            print("t = %0.4e, dt = %0.4e" % (t_current, dt))
        t_current += dt
        ts.append(t_current)
        if int(10 * t_current / tfinal) > curr_step:
            curr_step += 1
            print(curr_step, end="")

        icdt = 1.0 / (c * dt)
        zero_bc_outer = np.zeros((N, 2))
        zero_bc_inner = np.zeros((N, 2))
        iterations_step = 0

        # Preserve start-of-step ψ and g for icdt·ψ^n / icdt·g^n source terms
        psi_step_start = psi_old.copy()
        g_step_start = g_old.copy()

        # ── T_star nonlinear outer loop (identical logic to slab solver) ──
        _T_max = 1e50
        T_star = np.minimum(T_old.copy(), _T_max)
        T_star_prev = T_star.copy()
        psi_bc_state = psi_old.copy()
        max_reflect_its = 20 if (reflect_outer or reflect_inner) else 1
        reflect_tol = 1e-12
        x_sol = phi.ravel()
        sigma_a = None
        total_its = 0
        do_final = False
        k_outer = 0

        while True:
            # ── Inner tolerance ────────────────────────────────────────────
            if W == 0 or do_final:
                tau_phi = tolerance
            elif k_outer == 0:
                tau_phi = tau_phi_max
            else:
                eta_T_k = float(np.sqrt(np.mean(
                    np.clip((T_star - T_star_prev) / (np.abs(T_star_prev) + T_floor),
                            -1e30, 1e30)**2)))
                tau_phi = float(np.clip(C_T * eta_T_k, tolerance, tau_phi_max))

            # ── Linearise at T_star ────────────────────────────────────────
            h_cv = np.maximum(np.abs(T_star) * 1e-4, 1e-12)
            Cv = np.maximum(
                (EOS(T_star + h_cv) - EOS(np.maximum(T_star - h_cv, T_floor))) / (2.0 * h_cv),
                1e-30)
            beta_val = 4.0 * a * T_star**3 / Cv
            sigma = sigma_func(T_star)
            scat = scat_func(T_star)
            f = 1.0 / (1.0 + beta_val * c * dt * sigma)
            sigma_a = f * sigma
            sigma_s = (1.0 - f) * sigma + scat
            sigma_t = sigma + scat
            sigma_hat = sigma_t + icdt
            emission = sigma_a * ac * T_star**4
            delta_e_src = -(1.0 - f) * (EOS(T_star) - e_old) / dt   # (I, 2)

            # Fixed part of source.
            # Emission and delta_e_src are isotropic: each ordinate receives 1/2
            # of the scalar source (since phi = int_{-1}^{1} I dmu = 2*sum w_n I_n).
            # The icdt term carries the previous-step angular flux directly.
            source_n_fixed = (
                0.5 * (emission + delta_e_src)[:, np.newaxis, :]
                + icdt * psi_step_start
            )   # (I, N, 2)
            source_n_fixed = source_n_fixed + q_n

            # source_g[j, lr] = 0.5*(emission + delta_e_src) + icdt*g^n
            source_g_fixed = (
                0.5 * (emission + delta_e_src)
                + icdt * g_step_start
                + q_g
            )   # (I, 2)

            # matvec: scattering source only (sigma_s * phi), same for all n and g
            def _make_mv_sph(ss, sh, nI, nN, r_l, d_r, ml, mr,
                             mu_e, alp, sv, zbc_o, zbc_i, fs):
                _ss = ss
                _sh = sh

                def mv(phi_vec):
                    phi_2d = phi_vec.reshape((nI, 2))
                    # Isotropic scattering: each ordinate receives 1/2 of sigma_s*phi
                    src_s_n = (0.5 * _ss * phi_2d)[:, np.newaxis, :] * np.ones((1, nN, 1))
                    src_s_g = 0.5 * _ss * phi_2d      # (I, 2)
                    phi_out, _, _ = _sweep_all_sph_ld(
                        nI, r_l, d_r,
                        np.ascontiguousarray(src_s_n),
                        np.ascontiguousarray(src_s_g),
                        _sh, MU, W_quad, mu_e, alp, sv, ml, mr,
                        zbc_o, 0.0, zbc_i, fs, 1,
                    )
                    return phi_out.ravel()
                return mv

            mv = _make_mv_sph(
                sigma_s, sigma_hat, I, N, r_left, dr_arr, M_l, M_r,
                mu_edges, alpha_arr, s_arr, zero_bc_outer, zero_bc_inner,
                _full_sphere,
            )

            # ── Reflecting BC convergence loop ─────────────────────────────
            for _ in range(max_reflect_its):
                bc_outer_use, bc_g_outer_use = BCs(t_current - dt / 2.0)
                bc_outer_use = np.asarray(bc_outer_use, dtype=np.float64).copy()
                if reflect_outer:
                    bc_outer_use = build_reflecting_BCs_sph_ld(
                        bc_outer_use, psi_bc_state, N)
                    # Also reflect the starting direction g (μ = −1) back at
                    # the outer wall: for a reflecting wall, the inward g at
                    # r = R equals the outward g value at the outer face of
                    # the last cell from the previous iteration.
                    bc_g_outer_use = float(g_old[-1, 1])

                # Inner boundary
                if _full_sphere:
                    bc_inner_use = zero_bc_inner  # origin condition applied in kernel
                elif reflect_inner:
                    bc_inner_use = build_reflecting_inner_BCs_sph_ld(
                        np.zeros((N, 2)), psi_bc_state, N)
                elif BCs_inner is not None:
                    bc_inner_use = np.asarray(BCs_inner(t_current - dt / 2.0),
                                              dtype=np.float64).copy()
                else:
                    bc_inner_use = zero_bc_inner  # vacuum inner wall

                b_vec = single_sweep_phi_sph_ld(
                    I, r_left, dr_arr, source_n_fixed, source_g_fixed,
                    sigma_hat, N, bc_outer_use, bc_g_outer_use,
                    bc_inner=bc_inner_use, fix=fix,
                ).ravel()

                try:
                    x_sol_cand, total_its, _chg, _chgL, _At, _Yp, _Ym = \
                        solver_with_dmd_inc(
                            matvec=mv, b=b_vec, K=K, max_its=maxits, steady=1,
                            x=x_sol, Rits=R, LOUD=LOUD,
                            L2_tol=tau_phi, Linf_tol=Linf_tol,
                        ) if use_dmd else \
                        _richardson_solve(mv, b_vec, x_sol, maxits, tau_phi, Linf_tol, LOUD)
                    if np.any(~np.isfinite(x_sol_cand)):
                        raise np.linalg.LinAlgError("DMD produced non-finite solution")
                    x_sol = x_sol_cand
                except np.linalg.LinAlgError:
                    x_sol, total_its, _chg, _chgL, _At, _Yp, _Ym = \
                        _richardson_solve(mv, b_vec, x_sol, maxits, tau_phi, Linf_tol, LOUD)
                iterations += total_its
                iterations_step += total_its
                phi = x_sol.reshape((I, 2))

                # Reconstruct full angular flux ψ and g at end-of-step
                bc_outer_psi, bc_g_outer_psi = BCs(t_current)
                bc_outer_psi = np.asarray(bc_outer_psi, dtype=np.float64).copy()
                if reflect_outer:
                    bc_outer_psi = build_reflecting_BCs_sph_ld(
                        bc_outer_psi, psi_bc_state, N)
                    # Reflect g at outer wall using current g state
                    bc_g_outer_psi = float(g_old[-1, 1])
                if _full_sphere:
                    bc_inner_psi = zero_bc_inner
                elif reflect_inner:
                    bc_inner_psi = build_reflecting_inner_BCs_sph_ld(
                        np.zeros((N, 2)), psi_bc_state, N)
                elif BCs_inner is not None:
                    bc_inner_psi = np.asarray(BCs_inner(t_current),
                                              dtype=np.float64).copy()
                else:
                    bc_inner_psi = zero_bc_inner
                full_src_n = source_n_fixed + (0.5 * sigma_s * phi)[:, np.newaxis, :] * np.ones((1, N, 1))
                full_src_g = source_g_fixed + 0.5 * sigma_s * phi
                psi_candidate, _, g_candidate = single_sweep_psi_sph_ld(
                    I, r_left, dr_arr, full_src_n, full_src_g,
                    sigma_hat, N, bc_outer_psi, bc_g_outer_psi,
                    bc_inner=bc_inner_psi, fix=fix,
                )

                if not (reflect_outer or reflect_inner):
                    psi_old = psi_candidate
                    g_old = g_candidate
                    break

                # Check boundary fixed-point convergence
                bc_old_o = build_reflecting_BCs_sph_ld(
                    np.zeros((N, 2)), psi_bc_state, N)
                bc_new_o = build_reflecting_BCs_sph_ld(
                    np.zeros((N, 2)), psi_candidate, N)
                bc_change = np.max(np.abs(bc_new_o - bc_old_o))
                if reflect_inner and not _full_sphere:
                    bc_old_i = build_reflecting_inner_BCs_sph_ld(
                        np.zeros((N, 2)), psi_bc_state, N)
                    bc_new_i = build_reflecting_inner_BCs_sph_ld(
                        np.zeros((N, 2)), psi_candidate, N)
                    bc_change = max(bc_change, np.max(np.abs(bc_new_i - bc_old_i)))
                psi_bc_state = psi_candidate.copy()
                psi_old = psi_candidate
                g_old = g_candidate
                if bc_change < reflect_tol:
                    break

            # ── Exit condition ─────────────────────────────────────────────
            if W == 0 or do_final:
                break

            # ── T_cand and damped T_star update ───────────────────────────
            delta_e_star = EOS(T_star) - e_old
            e_cand = (e_old
                      + sigma_a * dt * (phi - ac * T_star**4)
                      + (1.0 - f) * delta_e_star)
            T_cand = invEOS(e_cand)
            T_star_prev = T_star.copy()
            T_star = np.clip(
                (1.0 - omega_T) * T_star + omega_T * T_cand,
                T_floor, _T_max)

            diff = T_star - T_star_prev
            eta_T_new = float(
                np.sqrt(np.mean(np.clip(diff / (np.abs(T_star_prev) + T_floor),
                                        -1e30, 1e30)**2)))
            k_outer += 1
            if eta_T_new < tau_T or k_outer >= W:
                do_final = True

        # ── Final material energy update ──────────────────────────────────
        delta_e_star = EOS(T_star) - e_old
        e = (e_old
             + sigma_a * dt * (phi - ac * T_star**4)
             + (1.0 - f) * delta_e_star)
        T = invEOS(e)

        # ── Per-step diagnostics ─────────────────────────────────────────
        if print_stride > 0 and step_num % print_stride == 0:
            print(f"  step {step_num:5d}  t={t_current:.4e} ns  dt={dt:.3e}  "
                  f"T_max={np.max(T):.4f} keV  its={total_its}")

        # ── Adaptive dt: second time-derivative of T ─────────────────────
        if step_num >= 2:
            denom = np.mean(np.abs(
                T / (dt**2)
                - (dt + dt_old) / (dt**2 * dt_old) * T_old
                + T_old2 / (dt_old * dt)))
            mean_T = np.mean(T)
            if denom > 0.0 and np.isfinite(mean_T) and np.isfinite(denom):
                deriv_val = mean_T / denom
            else:
                deriv_val = dt_max**2 / delta_step

        e_old = e.copy()
        T_old2 = T_old.copy()
        T_old = T.copy()
        its_per_step.append(iterations_step)
        phis.append(phi.copy())
        Ts.append(T.copy())
        gs.append(g_old.copy())

    print()
    return phis, Ts, gs, iterations, np.array(ts), its_per_step
