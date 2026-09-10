"""
2-D Cartesian Spherical Harmonics (P_N) solver using Simple Corner Balance.

Reference: Chapter 12 of the textbook.

Equations
---------
  (1/c) ∂I/∂t + A_x ∂I/∂x + A_z ∂I/∂z + σ_a I = Q̂

  I     – (N+1)(N+2)/2 vector of 2D real spherical-harmonics moments
  A_x, A_z – symmetric coupling matrices (loaded from CSV)
  Q̂_00  = source_iso / √(4π)  (isotropic source in φ-units)
  Q̂_lm  = 0  for lm > 0

Scalar flux: φ = √(4π) I_00.

SCB discretisation (Section 12.3.1)
------------------------------------
Each cell (i,j) is divided into 4 corners: NE=0, NW=1, SW=2, SE=3.
Face conventions (outward normal, face "length" in 2-D):
  NE corner:
    right  (+x̂, length Lz = dz/2): upwind flux  A_x⁺ I_NE + A_x⁻ I_right
    top    (+ẑ, length Lx = dx/2): upwind flux   A_z⁺ I_NE + A_z⁻ I_top
    left   (−x̂, length Lz):       centred  −A_x/2 (I_NE + I_NW)
    bottom (−ẑ, length Lx):       centred  −A_z/2 (I_NE + I_SE)
  (other corners follow by symmetry)

This gives diagonal block D = σ̂ V Id + (Lz/2)|A_x| + (Lx/2)|A_z|
for every corner (V = Lx·Lz is the corner volume), and a 4S×4S block system

  ┌ D       −Lz Ax/2   0        −Lx Az/2 ┐ ┌ I_NE ┐   ┌ rhs_NE ┐
  │ Lz Ax/2  D        −Lx Az/2   0       │ │ I_NW │ = │ rhs_NW │
  │ 0        Lx Az/2   D         Lz Ax/2 │ │ I_SW │   │ rhs_SW │
  └ Lx Az/2  0        −Lz Ax/2   D       ┘ └ I_SE ┘   └ rhs_SE ┘

where the RHS carries boundary incoming fluxes from neighbour cells / ghost cells.

Reflecting ghost-cell BCs (Section 12.3.2)
--------------------------------------------
  x-boundary: ghost = ±I_interior with −1 for odd-m moments, +1 otherwise.
  z-boundary: ghost = ±I_interior with −1 for odd-(l−m) moments, +1 otherwise.
  Vacuum:     ghost = 0.

Iteration
----------
Gauss-Seidel passes (forward j→, i→ then backward j←, i←) serve as the
transport solve.  Fleck-Cummings isotropic scatter is handled by Richardson
source iteration on the full moment vector I.

Physical constants: CGS, time in nanoseconds.

Array shapes
------------
  I         : (Ix, Iy, 4, S)  – moment vector per corner
  phi       : (Ix, Iy, 4)     – scalar flux √(4π) I[…, 0]
  sigma_hat : (Ix, Iy, 4)     – σ_t + 1/(c dt) per corner
  src_full  : (Ix, Iy, 4, S)  – full Q̂ per corner
"""

import math
import os

import numpy as np

try:
    from numba import njit, prange
    _HAVE_NUMBA = True
except Exception:
    njit = None
    prange = range
    _HAVE_NUMBA = False

try:
    from scipy.linalg import lu_factor, lu_solve
    _HAVE_SCIPY_LU = True
except Exception:
    lu_factor = None
    lu_solve = None
    _HAVE_SCIPY_LU = False

# ---------------------------------------------------------------------------
# Physical constants (CGS, time in ns)
# ---------------------------------------------------------------------------
c   = 29.98       # speed of light  (cm / ns)
a   = 0.01372     # radiation constant  (GJ / cm³ / keV⁴)
ac  = a * c

_SQRT4PI = math.sqrt(4.0 * math.pi)


# ===========================================================================
# Moment index helpers
# ===========================================================================

def _n_moments_2d(N):
    """Number of 2-D P_N moments (N+1)(N+2)/2."""
    return (N + 1) * (N + 2) // 2


def _moment_list_2d(N):
    """Ordered list of (l, m) pairs, m ≥ 0."""
    return [(l, m) for l in range(N + 1) for m in range(l + 1)]


def reflect_mask_x(N):
    """Ghost-cell sign mask for x-reflecting BC: −1 for odd m, +1 for even m."""
    return np.array([(-1.0) ** m for l, m in _moment_list_2d(N)])


def reflect_mask_z(N):
    """Ghost-cell sign mask for z-reflecting BC: −1 for odd (l−m), +1 otherwise."""
    return np.array([(-1.0) ** (l - m) for l, m in _moment_list_2d(N)])


def build_filter_moments_2d(N, filter_type='lanczos', exp_order=4):
    """Build f(l,N) = -log(rho(l/(N+1))) for each (l,m) moment entry."""
    zeta = np.array([l / float(N + 1) for l, _ in _moment_list_2d(N)], dtype=np.float64)

    ftype = str(filter_type).strip().lower()
    if ftype == 'lanczos':
        rho = np.sinc(zeta)
    elif ftype in ('spline', 'sspline'):
        rho = 1.0 / (1.0 + zeta ** 4)
    elif ftype in ('exp', 'exponential'):
        alpha = int(exp_order)
        if alpha <= 0:
            raise ValueError("exp_order must be positive for exponential filter")
        c0 = math.log(np.finfo(np.float64).eps)
        rho = np.exp(c0 * zeta ** alpha)
    else:
        raise ValueError(
            f"Unknown filter_type '{filter_type}'. Use 'lanczos', 'sspline', or 'exponential'.")

    rho = np.clip(rho, 1e-300, 1.0)
    f_mom = -np.log(rho)
    f_mom[0] = 0.0  # preserve scalar flux / energy moment exactly
    return f_mom


def _evaluate_filter_strength(filter_strength, T_state, Ix, Iy):
    """Return filter strength as an (Ix,Iy,4) array from scalar/array/callable input."""
    if filter_strength is None:
        return np.zeros((Ix, Iy, 4), dtype=np.float64)

    raw = filter_strength(T_state) if callable(filter_strength) else filter_strength
    arr = np.asarray(raw, dtype=np.float64)

    if arr.ndim == 0:
        return np.full((Ix, Iy, 4), float(arr), dtype=np.float64)
    if arr.shape == (Ix, Iy):
        return np.repeat(arr[:, :, np.newaxis], 4, axis=2)
    if arr.shape == (Ix, Iy, 1):
        return np.repeat(arr, 4, axis=2)
    if arr.shape == (Ix, Iy, 4):
        return arr
    raise ValueError(
        "filter_strength must be scalar, callable->scalar/array, or shaped (Ix,Iy) / (Ix,Iy,4)")


# ===========================================================================
# Jacobian loading
# ===========================================================================

def load_pn_jacobians(N, jacobian_dir):
    """Load P_N coupling matrices and pre-compute derived forms.

    Returns dict with keys:
      N, S, Ax, Az, abs_Ax, abs_Az, Ax_plus, Ax_minus, Az_plus, Az_minus
    """
    S = _n_moments_2d(N)
    Ax = np.loadtxt(os.path.join(jacobian_dir, f'P{N}_x.csv'),
                    delimiter=',')[:S, :S]
    Az = np.loadtxt(os.path.join(jacobian_dir, f'P{N}_z.csv'),
                    delimiter=',')[:S, :S]

    # Symmetric: |A| = V |Λ| Vᵀ
    ax_vals, ax_vecs = np.linalg.eigh(Ax)
    az_vals, az_vecs = np.linalg.eigh(Az)
    abs_Ax = ax_vecs @ np.diag(np.abs(ax_vals)) @ ax_vecs.T
    abs_Az = az_vecs @ np.diag(np.abs(az_vals)) @ az_vecs.T

    return dict(
        N=N, S=S,
        Ax=Ax,       Az=Az,
        abs_Ax=abs_Ax, abs_Az=abs_Az,
        Ax_plus =0.5 * (Ax + abs_Ax),
        Ax_minus=0.5 * (Ax - abs_Ax),
        Az_plus =0.5 * (Az + abs_Az),
        Az_minus=0.5 * (Az - abs_Az),
    )


# ===========================================================================
# Streaming-block cache (per unique dx, dz pair)
# ===========================================================================

def _streaming_block(S, dx, dz, jac):
    """Build the sigma-independent part of the 4S×4S cell matrix.

    Geometry:
      Lx = dx/2  (half-width in x)  — used for z-face coefficients
      Lz = dz/2  (half-width in z)  — used for x-face coefficients
      V  = Lx·Lz

    Diagonal (streaming):  D_stream = (Lz/2)|Ax| + (Lx/2)|Az|
    Off-diagonals:
      M[NE,NW] = −(Lz/2)Ax    M[NW,NE] = +(Lz/2)Ax
      M[NE,SE] = −(Lx/2)Az    M[SE,NE] = +(Lx/2)Az
      M[NW,SW] = −(Lx/2)Az    M[SW,NW] = +(Lx/2)Az
      M[SW,SE] = +(Lz/2)Ax    M[SE,SW] = −(Lz/2)Ax
    """
    Lx = 0.5 * dx      # half x-width (z-face coefficient)
    Lz = 0.5 * dz      # half z-width (x-face coefficient)

    D_s = 0.5 * Lz * jac['abs_Ax'] + 0.5 * Lx * jac['abs_Az']  # (S,S)
    hLz_Ax = 0.5 * Lz * jac['Ax']                                # (S,S)
    hLx_Az = 0.5 * Lx * jac['Az']                                # (S,S)

    M = np.zeros((4 * S, 4 * S))
    for c in range(4):
        sl = slice(c * S, (c + 1) * S)
        M[sl, sl] = D_s

    M[0*S:1*S, 1*S:2*S] = -hLz_Ax   # NE←NW
    M[1*S:2*S, 0*S:1*S] = +hLz_Ax   # NW←NE
    M[0*S:1*S, 3*S:4*S] = -hLx_Az   # NE←SE
    M[3*S:4*S, 0*S:1*S] = +hLx_Az   # SE←NE
    M[1*S:2*S, 2*S:3*S] = -hLx_Az   # NW←SW
    M[2*S:3*S, 1*S:2*S] = +hLx_Az   # SW←NW
    M[2*S:3*S, 3*S:4*S] = +hLz_Ax   # SW←SE
    M[3*S:4*S, 2*S:3*S] = -hLz_Ax   # SE←SW

    return M


def _build_cell_lu_cache(sigma_hat, sigma_s,
                         dx_arr, dy_arr, Ix, Iy, S,
                         jac, stream_cache,
                         sigma_filter=None,
                         filter_diag_mat=None,
                         deduplicate=True):
    """Pre-factor each cell's fixed 4Sx4S matrix for repeated GS solves.

    Returns
    -------
    dict[(int, int)] -> (lu, piv)
        LU factors from scipy.linalg.lu_factor, one per cell.
    """
    if not _HAVE_SCIPY_LU:
        return None

    eye_S = np.eye(S)
    cell_lu_cache = {}
    # Second-pass optimization: many problems have identical local
    # coefficients in many cells (uniform mesh/materials). Reuse LU factors.
    lu_by_signature = {} if deduplicate else None

    for j in range(Iy):
        dz = dy_arr[j]
        for i in range(Ix):
            dx = dx_arr[i]
            Lx = 0.5 * dx
            Lz = 0.5 * dz
            V = Lx * Lz

            if deduplicate:
                sig = (
                    dx, dz,
                    sigma_hat[i, j, 0], sigma_hat[i, j, 1],
                    sigma_hat[i, j, 2], sigma_hat[i, j, 3],
                    sigma_s[i, j, 0], sigma_s[i, j, 1],
                    sigma_s[i, j, 2], sigma_s[i, j, 3],
                    0.0 if sigma_filter is None else sigma_filter[i, j, 0],
                    0.0 if sigma_filter is None else sigma_filter[i, j, 1],
                    0.0 if sigma_filter is None else sigma_filter[i, j, 2],
                    0.0 if sigma_filter is None else sigma_filter[i, j, 3],
                )
                lu_piv = lu_by_signature.get(sig)
                if lu_piv is not None:
                    cell_lu_cache[(i, j)] = lu_piv
                    continue

            key = (dx, dz)
            if key not in stream_cache:
                stream_cache[key] = _streaming_block(S, dx, dz, jac)
            M = stream_cache[key].copy()

            for c in range(4):
                sl = slice(c * S, (c + 1) * S)
                M[sl, sl] += sigma_hat[i, j, c] * V * eye_S
                if sigma_filter is not None and filter_diag_mat is not None:
                    M[sl, sl] += sigma_filter[i, j, c] * V * filter_diag_mat
            for c in range(4):
                M[c * S, c * S] -= sigma_s[i, j, c] * V

            lu_piv = lu_factor(M, overwrite_a=True, check_finite=False)
            cell_lu_cache[(i, j)] = lu_piv
            if deduplicate:
                lu_by_signature[sig] = lu_piv

    return cell_lu_cache


# ===========================================================================
# Single Gauss-Seidel pass
# ===========================================================================

def _gs_pass(I_new, sigma_hat, sigma_s, src_full,
             dx_arr, dy_arr, Ix, Iy, S,
             jac, stream_cache,
             ref_xlo, ref_xhi, ref_ylo, ref_yhi,
             xmask, zmask,
             i_seq, j_seq,
             cell_lu_cache=None,
             boundary_moments=None,
             sigma_filter=None,
             filter_diag_mat=None):
    """
    One Gauss-Seidel pass (in-place update of I_new).

    Parameters
    ----------
    I_new     : (Ix, Iy, 4, S)  updated in place.
    sigma_hat : (Ix, Iy, 4)     σ_t + icdt per corner.
    sigma_s   : (Ix, Iy, 4)     isotropic scatter cross section per corner.
                For isotropic scattering the scatter gain σ_s I_00 exactly
                cancels the σ_s I_00 on the absorption side of the (0,0)
                equation, so the effective σ̂ for moment (0,0) is
                σ_hat − σ_s = σ_a + icdt.  This is applied here directly;
                no outer source-iteration loop is required.
    src_full  : (Ix, Iy, 4, S)  Q̂ per corner (all moments).
                The scatter source must NOT be included – it is already
                accounted for by the diagonal correction above.
    stream_cache : dict, maps (dx, dz) → precomputed streaming block.
    i_seq, j_seq : cell index sequences (define sweep order).
    """
    Ax_m = jac['Ax_minus']
    Ax_p = jac['Ax_plus']
    Az_m = jac['Az_minus']
    Az_p = jac['Az_plus']
    eye_S = np.eye(S)
    zero_S = np.zeros(S)

    xlo_bc = None if boundary_moments is None else boundary_moments.get('xlo')
    xhi_bc = None if boundary_moments is None else boundary_moments.get('xhi')
    ylo_bc = None if boundary_moments is None else boundary_moments.get('ylo')
    yhi_bc = None if boundary_moments is None else boundary_moments.get('yhi')

    for j in j_seq:
        dz = dy_arr[j]
        Lx = 0.5 * dx_arr[0]   # placeholder; overridden per i below
        for i in i_seq:
            dx = dx_arr[i]
            dz_ = dz            # local alias
            Lx  = 0.5 * dx     # half x-width
            Lz  = 0.5 * dz_    # half z-width
            V   = Lx * Lz      # corner volume

            if cell_lu_cache is None:
                # Retrieve or build the streaming block (sigma-independent)
                key = (dx, dz_)
                if key not in stream_cache:
                    stream_cache[key] = _streaming_block(S, dx, dz_, jac)
                M = stream_cache[key].copy()

                # Add sigma*V to diagonal blocks.
                # For the (0,0) moment the isotropic scatter gain cancels the
                # absorption loss, so the effective sigma is sigma_hat - sigma_s.
                for c in range(4):
                    sl = slice(c * S, (c + 1) * S)
                    M[sl, sl] += sigma_hat[i, j, c] * V * eye_S
                    if sigma_filter is not None and filter_diag_mat is not None:
                        M[sl, sl] += sigma_filter[i, j, c] * V * filter_diag_mat
                for c in range(4):
                    M[c * S, c * S] -= sigma_s[i, j, c] * V

            # -------------------------------------------------------
            # Gather incoming boundary data (always use I_new = latest)
            # -------------------------------------------------------
            # NE(0): right(+x) neighbour NW,  top(+z) neighbour SE
            inc_xhi_NE = (I_new[i+1, j, 1, :] if i < Ix-1
                      else (xhi_bc[j] if xhi_bc is not None
                          else (xmask * I_new[i, j, 0, :] if ref_xhi else zero_S)))
            inc_zhi_NE = (I_new[i, j+1, 3, :] if j < Iy-1
                      else (yhi_bc[i] if yhi_bc is not None
                          else (zmask * I_new[i, j, 0, :] if ref_yhi else zero_S)))

            # NW(1): left(-x) neighbour NE,  top(+z) neighbour SW
            inc_xlo_NW = (I_new[i-1, j, 0, :] if i > 0
                      else (xlo_bc[j] if xlo_bc is not None
                          else (xmask * I_new[i, j, 1, :] if ref_xlo else zero_S)))
            inc_zhi_NW = (I_new[i, j+1, 2, :] if j < Iy-1
                      else (yhi_bc[i] if yhi_bc is not None
                          else (zmask * I_new[i, j, 1, :] if ref_yhi else zero_S)))

            # SW(2): left(-x) neighbour SE,  bottom(-z) neighbour NW
            inc_xlo_SW = (I_new[i-1, j, 3, :] if i > 0
                      else (xlo_bc[j] if xlo_bc is not None
                          else (xmask * I_new[i, j, 2, :] if ref_xlo else zero_S)))
            inc_zlo_SW = (I_new[i, j-1, 1, :] if j > 0
                      else (ylo_bc[i] if ylo_bc is not None
                          else (zmask * I_new[i, j, 2, :] if ref_ylo else zero_S)))

            # SE(3): right(+x) neighbour SW,  bottom(-z) neighbour NE
            inc_xhi_SE = (I_new[i+1, j, 2, :] if i < Ix-1
                      else (xhi_bc[j] if xhi_bc is not None
                          else (xmask * I_new[i, j, 3, :] if ref_xhi else zero_S)))
            inc_zlo_SE = (I_new[i, j-1, 0, :] if j > 0
                      else (ylo_bc[i] if ylo_bc is not None
                          else (zmask * I_new[i, j, 3, :] if ref_ylo else zero_S)))

            # -------------------------------------------------------
            # Build RHS  (Lz = dz/2 is the x-face coefficient,
            #              Lx = dx/2 is the z-face coefficient)
            # -------------------------------------------------------
            rhs = np.empty(4 * S)
            rhs[0*S:1*S] = (V * src_full[i, j, 0]
                            - Lz * (Ax_m @ inc_xhi_NE)
                            - Lx * (Az_m @ inc_zhi_NE))
            rhs[1*S:2*S] = (V * src_full[i, j, 1]
                            + Lz * (Ax_p @ inc_xlo_NW)
                            - Lx * (Az_m @ inc_zhi_NW))
            rhs[2*S:3*S] = (V * src_full[i, j, 2]
                            + Lz * (Ax_p @ inc_xlo_SW)
                            + Lx * (Az_p @ inc_zlo_SW))
            rhs[3*S:4*S] = (V * src_full[i, j, 3]
                            - Lz * (Ax_m @ inc_xhi_SE)
                            + Lx * (Az_p @ inc_zlo_SE))

            # Solve 4S×4S system
            if cell_lu_cache is None:
                sol = np.linalg.solve(M, rhs)
            else:
                lu_piv = cell_lu_cache[(i, j)]
                sol = lu_solve(lu_piv, rhs, check_finite=False)
            I_new[i, j] = sol.reshape(4, S)


# ===========================================================================
# Public sweep function
# ===========================================================================

def sweep_pn(I_in, sigma_hat, sigma_s, src_full,
             dx_arr, dy_arr, Ix, Iy, S,
             jac, stream_cache,
             ref_xlo, ref_xhi, ref_ylo, ref_yhi,
             xmask, zmask,
             n_gs=1,
             cell_lu_cache=None,
             boundary_moments=None,
             sigma_filter=None,
             filter_diag_mat=None):
    """
    Perform n_gs forward+backward Gauss-Seidel passes.

    Each pass = one forward sweep (i↑, j↑) + one backward sweep (i↓, j↓).
    sigma_s is folded into the (0,0) diagonal (no scatter lag needed).
    Returns (I_new, phi) where phi = √(4π) I_new[…, 0].
    """
    I_new = I_in.copy()
    fwd_i = range(Ix)
    fwd_j = range(Iy)
    bwd_i = range(Ix - 1, -1, -1)
    bwd_j = range(Iy - 1, -1, -1)

    for _ in range(n_gs):
        _gs_pass(I_new, sigma_hat, sigma_s, src_full,
                 dx_arr, dy_arr, Ix, Iy, S,
                 jac, stream_cache,
                 ref_xlo, ref_xhi, ref_ylo, ref_yhi,
                 xmask, zmask, fwd_i, fwd_j,
                 cell_lu_cache=cell_lu_cache,
                 boundary_moments=boundary_moments,
                 sigma_filter=sigma_filter,
                 filter_diag_mat=filter_diag_mat)
        _gs_pass(I_new, sigma_hat, sigma_s, src_full,
                 dx_arr, dy_arr, Ix, Iy, S,
                 jac, stream_cache,
                 ref_xlo, ref_xhi, ref_ylo, ref_yhi,
                 xmask, zmask, bwd_i, bwd_j,
                 cell_lu_cache=cell_lu_cache,
                 boundary_moments=boundary_moments,
                 sigma_filter=sigma_filter,
                 filter_diag_mat=filter_diag_mat)

    phi = _SQRT4PI * I_new[:, :, :, 0]
    return I_new, phi


if _HAVE_NUMBA:
    @njit(cache=True, fastmath=True, parallel=True)
    def _apply_pn_op_numba_core(
        I, sigma_hat, sigma_s, dx_arr, dy_arr,
        Ax, Az, abs_Ax, abs_Az, Ax_m, Ax_p, Az_m, Az_p,
        ref_xlo, ref_xhi, ref_ylo, ref_yhi,
        xmask, zmask,
        xlo_bc, xhi_bc, ylo_bc, yhi_bc,
        has_xlo_bc, has_xhi_bc, has_ylo_bc, has_yhi_bc,
    ):
        Ix = I.shape[0]
        Iy = I.shape[1]
        S = I.shape[3]
        result = np.empty_like(I)

        for j in prange(Iy):
            dz = dy_arr[j]
            for i in range(Ix):
                dx = dx_arr[i]
                Lx = 0.5 * dx
                Lz = 0.5 * dz
                V = Lx * Lz

                I_ne = I[i, j, 0, :]
                I_nw = I[i, j, 1, :]
                I_sw = I[i, j, 2, :]
                I_se = I[i, j, 3, :]

                out_ne = (0.5 * Lz * (abs_Ax @ I_ne)
                          + 0.5 * Lx * (abs_Az @ I_ne)
                          + sigma_hat[i, j, 0] * V * I_ne)
                out_nw = (0.5 * Lz * (abs_Ax @ I_nw)
                          + 0.5 * Lx * (abs_Az @ I_nw)
                          + sigma_hat[i, j, 1] * V * I_nw)
                out_sw = (0.5 * Lz * (abs_Ax @ I_sw)
                          + 0.5 * Lx * (abs_Az @ I_sw)
                          + sigma_hat[i, j, 2] * V * I_sw)
                out_se = (0.5 * Lz * (abs_Ax @ I_se)
                          + 0.5 * Lx * (abs_Az @ I_se)
                          + sigma_hat[i, j, 3] * V * I_se)

                # Isotropic scattering correction only on the 00 moment.
                out_ne[0] -= sigma_s[i, j, 0] * V * I_ne[0]
                out_nw[0] -= sigma_s[i, j, 1] * V * I_nw[0]
                out_sw[0] -= sigma_s[i, j, 2] * V * I_sw[0]
                out_se[0] -= sigma_s[i, j, 3] * V * I_se[0]

                # Internal corner couplings from the local 4S x 4S block.
                out_ne += -0.5 * Lz * (Ax @ I_nw) - 0.5 * Lx * (Az @ I_se)
                out_nw += +0.5 * Lz * (Ax @ I_ne) - 0.5 * Lx * (Az @ I_sw)
                out_sw += +0.5 * Lx * (Az @ I_nw) + 0.5 * Lz * (Ax @ I_se)
                out_se += +0.5 * Lx * (Az @ I_ne) - 0.5 * Lz * (Ax @ I_sw)

                # External face states (neighbor, ghost BC, reflecting, vacuum)
                if i < Ix - 1:
                    inc_xhi_NE = I[i + 1, j, 1, :]
                    inc_xhi_SE = I[i + 1, j, 2, :]
                elif has_xhi_bc:
                    inc_xhi_NE = xhi_bc[j, :]
                    inc_xhi_SE = xhi_bc[j, :]
                elif ref_xhi:
                    inc_xhi_NE = xmask * I_ne
                    inc_xhi_SE = xmask * I_se
                else:
                    inc_xhi_NE = np.zeros(S)
                    inc_xhi_SE = np.zeros(S)

                if i > 0:
                    inc_xlo_NW = I[i - 1, j, 0, :]
                    inc_xlo_SW = I[i - 1, j, 3, :]
                elif has_xlo_bc:
                    inc_xlo_NW = xlo_bc[j, :]
                    inc_xlo_SW = xlo_bc[j, :]
                elif ref_xlo:
                    inc_xlo_NW = xmask * I_nw
                    inc_xlo_SW = xmask * I_sw
                else:
                    inc_xlo_NW = np.zeros(S)
                    inc_xlo_SW = np.zeros(S)

                if j < Iy - 1:
                    inc_zhi_NE = I[i, j + 1, 3, :]
                    inc_zhi_NW = I[i, j + 1, 2, :]
                elif has_yhi_bc:
                    inc_zhi_NE = yhi_bc[i, :]
                    inc_zhi_NW = yhi_bc[i, :]
                elif ref_yhi:
                    inc_zhi_NE = zmask * I_ne
                    inc_zhi_NW = zmask * I_nw
                else:
                    inc_zhi_NE = np.zeros(S)
                    inc_zhi_NW = np.zeros(S)

                if j > 0:
                    inc_zlo_SW = I[i, j - 1, 1, :]
                    inc_zlo_SE = I[i, j - 1, 0, :]
                elif has_ylo_bc:
                    inc_zlo_SW = ylo_bc[i, :]
                    inc_zlo_SE = ylo_bc[i, :]
                elif ref_ylo:
                    inc_zlo_SW = zmask * I_sw
                    inc_zlo_SE = zmask * I_se
                else:
                    inc_zlo_SW = np.zeros(S)
                    inc_zlo_SE = np.zeros(S)

                # External couplings (same signs as the Python operator)
                out_ne += Lz * (Ax_m @ inc_xhi_NE) + Lx * (Az_m @ inc_zhi_NE)
                out_nw -= Lz * (Ax_p @ inc_xlo_NW)
                out_nw += Lx * (Az_m @ inc_zhi_NW)
                out_sw -= Lz * (Ax_p @ inc_xlo_SW) + Lx * (Az_p @ inc_zlo_SW)
                out_se += Lz * (Ax_m @ inc_xhi_SE)
                out_se -= Lx * (Az_p @ inc_zlo_SE)

                result[i, j, 0, :] = out_ne
                result[i, j, 1, :] = out_nw
                result[i, j, 2, :] = out_sw
                result[i, j, 3, :] = out_se

        return result


def _apply_pn_op_numba(I, sigma_hat, sigma_s, dx_arr, dy_arr, Ix, Iy, S,
                       jac,
                       ref_xlo, ref_xhi, ref_ylo, ref_yhi, xmask, zmask,
                       boundary_moments=None):
    """Numba-accelerated operator path used primarily by GMRES matvec."""
    has_xlo_bc = boundary_moments is not None and boundary_moments.get('xlo') is not None
    has_xhi_bc = boundary_moments is not None and boundary_moments.get('xhi') is not None
    has_ylo_bc = boundary_moments is not None and boundary_moments.get('ylo') is not None
    has_yhi_bc = boundary_moments is not None and boundary_moments.get('yhi') is not None

    xlo_bc = np.ascontiguousarray(boundary_moments['xlo']) if has_xlo_bc else np.zeros((Iy, S))
    xhi_bc = np.ascontiguousarray(boundary_moments['xhi']) if has_xhi_bc else np.zeros((Iy, S))
    ylo_bc = np.ascontiguousarray(boundary_moments['ylo']) if has_ylo_bc else np.zeros((Ix, S))
    yhi_bc = np.ascontiguousarray(boundary_moments['yhi']) if has_yhi_bc else np.zeros((Ix, S))

    return _apply_pn_op_numba_core(
        I, sigma_hat, sigma_s, dx_arr, dy_arr,
        jac['Ax'], jac['Az'], jac['abs_Ax'], jac['abs_Az'],
        jac['Ax_minus'], jac['Ax_plus'], jac['Az_minus'], jac['Az_plus'],
        ref_xlo, ref_xhi, ref_ylo, ref_yhi,
        xmask, zmask,
        xlo_bc, xhi_bc, ylo_bc, yhi_bc,
        has_xlo_bc, has_xhi_bc, has_ylo_bc, has_yhi_bc,
    )


# ===========================================================================
# Matrix-free operator  A(I)  (used by GMRES)
# ===========================================================================

def _apply_pn_op(I, sigma_hat, sigma_s, dx_arr, dy_arr, Ix, Iy, S,
                 jac, stream_cache,
                 ref_xlo, ref_xhi, ref_ylo, ref_yhi, xmask, zmask,
                 boundary_moments=None,
                 sigma_filter=None,
                 filter_diag_mat=None):
    """
    Apply the global P_N transport operator A to I (matrix-free).

    The system is  A I = b  where b = V * src_full (per cell/corner).

    For each cell (i,j) and corner c::

      (A I)_{ij,c}  =  (M_local @ I_{ij})[c-block]
                    +  external-face coupling contributions from I_neighbours

    where M_local is the 4S×4S cell matrix (absorption + internal streaming)
    with scatter folded into the (0,0) diagonal exactly as in _gs_pass.

    The external coupling terms appeared as rhs boundary-incoming terms in
    the GS formulation; here they are moved to the LHS (sign flip):

      NE: + Lz Ax⁻ I[i+1,j,NW]  + Lx Az⁻ I[i,j+1,SE]
      NW: − Lz Ax⁺ I[i−1,j,NE]  + Lx Az⁻ I[i,j+1,SW]
      SW: − Lz Ax⁺ I[i−1,j,SE]  − Lx Az⁺ I[i,j−1,NW]
      SE: + Lz Ax⁻ I[i+1,j,SW]  − Lx Az⁺ I[i,j−1,NE]

    Ghost-cell values for reflecting/vacuum boundaries are substituted
    consistently with _gs_pass.
    """
    if _HAVE_NUMBA and sigma_filter is None and filter_diag_mat is None:
        return _apply_pn_op_numba(
            I, sigma_hat, sigma_s, dx_arr, dy_arr, Ix, Iy, S,
            jac,
            ref_xlo, ref_xhi, ref_ylo, ref_yhi, xmask, zmask,
            boundary_moments=boundary_moments)

    Ax_m = jac['Ax_minus']
    Ax_p = jac['Ax_plus']
    Az_m = jac['Az_minus']
    Az_p = jac['Az_plus']
    eye_S  = np.eye(S)
    zero_S = np.zeros(S)

    xlo_bc = None if boundary_moments is None else boundary_moments.get('xlo')
    xhi_bc = None if boundary_moments is None else boundary_moments.get('xhi')
    ylo_bc = None if boundary_moments is None else boundary_moments.get('ylo')
    yhi_bc = None if boundary_moments is None else boundary_moments.get('yhi')

    result = np.empty_like(I)

    for j in range(Iy):
        dz = dy_arr[j]
        for i in range(Ix):
            dx = dx_arr[i]
            Lx = 0.5 * dx
            Lz = 0.5 * dz
            V  = Lx * Lz

            key = (dx, dz)
            if key not in stream_cache:
                stream_cache[key] = _streaming_block(S, dx, dz, jac)
            M = stream_cache[key].copy()

            for c in range(4):
                sl = slice(c * S, (c + 1) * S)
                M[sl, sl] += sigma_hat[i, j, c] * V * eye_S
                if sigma_filter is not None and filter_diag_mat is not None:
                    M[sl, sl] += sigma_filter[i, j, c] * V * filter_diag_mat
            for c in range(4):
                M[c * S, c * S] -= sigma_s[i, j, c] * V

            out = M @ I[i, j].ravel()       # (4S,) — local part

            # --- external face coupling (sign-flipped rhs terms) ----------
            # NE(0): +x and +z external faces
            inc_xhi_NE = (I[i+1, j, 1, :] if i < Ix-1
                      else (xhi_bc[j] if xhi_bc is not None
                          else (xmask * I[i, j, 0, :] if ref_xhi else zero_S)))
            inc_zhi_NE = (I[i, j+1, 3, :] if j < Iy-1
                      else (yhi_bc[i] if yhi_bc is not None
                          else (zmask * I[i, j, 0, :] if ref_yhi else zero_S)))
            # NW(1): −x and +z external faces
            inc_xlo_NW = (I[i-1, j, 0, :] if i > 0
                      else (xlo_bc[j] if xlo_bc is not None
                          else (xmask * I[i, j, 1, :] if ref_xlo else zero_S)))
            inc_zhi_NW = (I[i, j+1, 2, :] if j < Iy-1
                      else (yhi_bc[i] if yhi_bc is not None
                          else (zmask * I[i, j, 1, :] if ref_yhi else zero_S)))
            # SW(2): −x and −z external faces
            inc_xlo_SW = (I[i-1, j, 3, :] if i > 0
                      else (xlo_bc[j] if xlo_bc is not None
                          else (xmask * I[i, j, 2, :] if ref_xlo else zero_S)))
            inc_zlo_SW = (I[i, j-1, 1, :] if j > 0
                      else (ylo_bc[i] if ylo_bc is not None
                          else (zmask * I[i, j, 2, :] if ref_ylo else zero_S)))
            # SE(3): +x and −z external faces
            inc_xhi_SE = (I[i+1, j, 2, :] if i < Ix-1
                      else (xhi_bc[j] if xhi_bc is not None
                          else (xmask * I[i, j, 3, :] if ref_xhi else zero_S)))
            inc_zlo_SE = (I[i, j-1, 0, :] if j > 0
                      else (ylo_bc[i] if ylo_bc is not None
                          else (zmask * I[i, j, 3, :] if ref_ylo else zero_S)))

            out[0*S:1*S] += Lz*(Ax_m @ inc_xhi_NE) + Lx*(Az_m @ inc_zhi_NE)
            out[1*S:2*S] -= Lz*(Ax_p @ inc_xlo_NW)
            out[1*S:2*S] += Lx*(Az_m @ inc_zhi_NW)
            out[2*S:3*S] -= Lz*(Ax_p @ inc_xlo_SW) + Lx*(Az_p @ inc_zlo_SW)
            out[3*S:4*S] += Lz*(Ax_m @ inc_xhi_SE)
            out[3*S:4*S] -= Lx*(Az_p @ inc_zlo_SE)

            result[i, j] = out.reshape(4, S)

    return result


# ===========================================================================
# Step solver: GS iteration or matrix-free GMRES
# ===========================================================================

def solve_pn_step(sigma_hat, sigma_s, src_full, I_init,
                  dx_arr, dy_arr, Ix, Iy, S,
                  jac, stream_cache,
                  ref_xlo, ref_xhi, ref_ylo, ref_yhi,
                  xmask, zmask,
                  use_gmres=False,
                  n_gs=1, tol=1e-8, maxits=300, loud=False,
                  use_cached_lu=True,
                  deduplicate_lu=True,
                  gmres_precond_gs=1,
                  boundary_moments=None,
                  sigma_filter=None,
                  filter_moments=None):
    """
    Solve one linearised P_N transport step:  A I = V * src_full.

    Isotropic scatter is folded into the (0,0) diagonal of A (no lagging):
      effective σ̂₀₀ = σ_hat − σ_s  (= σ_a + icdt).

    Parameters
    ----------
    sigma_hat : (Ix, Iy, 4)   σ_t + icdt.
    sigma_s   : (Ix, Iy, 4)   isotropic scatter (Fleck-Cummings effective).
    src_full  : (Ix, Iy, 4, S) full moment source Q̂ (no scatter term).
    use_gmres : bool  If True, use matrix-free GMRES with one GS pass as
                      preconditioner.  If False, use GS iteration (n_gs
                      forward+backward passes per Picard step).
    n_gs      : int   GS passes per Picard step (ignored when use_gmres=True).
    tol       : convergence tolerance on ‖Δφ‖_∞ / ‖φ‖_∞.
    maxits    : maximum iterations.

    Returns
    -------
    I_new : (Ix, Iy, 4, S)
    phi   : (Ix, Iy, 4)
    n_its : int
    """
    if use_gmres:
        return _solve_pn_gmres(
            sigma_hat, sigma_s, src_full, I_init,
            dx_arr, dy_arr, Ix, Iy, S,
            jac, stream_cache,
            ref_xlo, ref_xhi, ref_ylo, ref_yhi,
            xmask, zmask, tol=tol, maxits=maxits, loud=loud,
            boundary_moments=boundary_moments,
            use_cached_lu=use_cached_lu,
            deduplicate_lu=deduplicate_lu,
            gmres_precond_gs=gmres_precond_gs,
            sigma_filter=sigma_filter,
            filter_moments=filter_moments)
    else:
        return _solve_pn_gs(
            sigma_hat, sigma_s, src_full, I_init,
            dx_arr, dy_arr, Ix, Iy, S,
            jac, stream_cache,
            ref_xlo, ref_xhi, ref_ylo, ref_yhi,
            xmask, zmask, n_gs=n_gs, tol=tol, maxits=maxits, loud=loud,
            boundary_moments=boundary_moments,
            use_cached_lu=use_cached_lu,
            deduplicate_lu=deduplicate_lu,
            sigma_filter=sigma_filter,
            filter_moments=filter_moments)


def _solve_pn_gs(sigma_hat, sigma_s, src_full, I_init,
                 dx_arr, dy_arr, Ix, Iy, S,
                 jac, stream_cache,
                 ref_xlo, ref_xhi, ref_ylo, ref_yhi,
                 xmask, zmask,
                 n_gs=1, tol=1e-8, maxits=300, loud=False,
                 use_cached_lu=True,
                 deduplicate_lu=True,
                 boundary_moments=None,
                 sigma_filter=None,
                 filter_moments=None):
    """Picard (GS) iteration: repeat n_gs fwd+bwd passes until phi converges."""
    filter_diag_mat = (np.diag(filter_moments)
                       if (filter_moments is not None and sigma_filter is not None)
                       else None)

    cell_lu_cache = None
    if use_cached_lu and _HAVE_SCIPY_LU:
        cell_lu_cache = _build_cell_lu_cache(
            sigma_hat, sigma_s,
            dx_arr, dy_arr, Ix, Iy, S,
            jac, stream_cache,
            sigma_filter=sigma_filter,
            filter_diag_mat=filter_diag_mat,
            deduplicate=deduplicate_lu)

    I_curr = I_init.copy()
    for k in range(maxits):
        I_new, phi_new = sweep_pn(
            I_curr, sigma_hat, sigma_s, src_full,
            dx_arr, dy_arr, Ix, Iy, S,
            jac, stream_cache,
            ref_xlo, ref_xhi, ref_ylo, ref_yhi,
            xmask, zmask, n_gs=n_gs,
            cell_lu_cache=cell_lu_cache,
            boundary_moments=boundary_moments,
            sigma_filter=sigma_filter,
            filter_diag_mat=filter_diag_mat)
        phi_curr = _SQRT4PI * I_curr[:, :, :, 0]
        denom = float(np.max(np.abs(phi_new))) + 1e-30
        res   = float(np.max(np.abs(phi_new - phi_curr))) / denom
        if loud:
            print(f"    GS it {k+1:3d}: rel residual = {res:.3e}")
        I_curr = I_new
        if res < tol:
            return I_curr, phi_new, k + 1
    return I_curr, _SQRT4PI * I_curr[:, :, :, 0], maxits


def _solve_pn_gmres(sigma_hat, sigma_s, src_full, I_init,
                    dx_arr, dy_arr, Ix, Iy, S,
                    jac, stream_cache,
                    ref_xlo, ref_xhi, ref_ylo, ref_yhi,
                    xmask, zmask,
                    tol=1e-8, maxits=300, loud=False,
                    use_cached_lu=True,
                    deduplicate_lu=True,
                    gmres_precond_gs=1,
                    boundary_moments=None,
                    sigma_filter=None,
                    filter_moments=None):
    """
    Matrix-free GMRES for one P_N transport step.

    Operator A is applied via _apply_pn_op.  One forward+backward GS pass
    is used as a left preconditioner M ≈ A.
    """
    from scipy.sparse.linalg import gmres, LinearOperator

    n = Ix * Iy * 4 * S

    filter_diag_mat = (np.diag(filter_moments)
                       if (filter_moments is not None and sigma_filter is not None)
                       else None)

    cell_lu_cache = None
    if use_cached_lu and _HAVE_SCIPY_LU:
        cell_lu_cache = _build_cell_lu_cache(
            sigma_hat, sigma_s,
            dx_arr, dy_arr, Ix, Iy, S,
            jac, stream_cache,
            sigma_filter=sigma_filter,
            filter_diag_mat=filter_diag_mat,
            deduplicate=deduplicate_lu)

    # Precompute V per cell for rhs scaling
    V_arr = np.empty((Ix, Iy))
    for j in range(Iy):
        for i in range(Ix):
            V_arr[i, j] = 0.5 * dx_arr[i] * 0.5 * dy_arr[j]

    # RHS: b = V * src_full  (scalar volume weight per cell)
    b = src_full * V_arr[:, :, np.newaxis, np.newaxis]
    b_flat = b.ravel()

    def matvec(x_flat):
        I_x = x_flat.reshape(Ix, Iy, 4, S)
        return _apply_pn_op(
            I_x, sigma_hat, sigma_s,
            dx_arr, dy_arr, Ix, Iy, S,
            jac, stream_cache,
            ref_xlo, ref_xhi, ref_ylo, ref_yhi, xmask, zmask,
            boundary_moments=boundary_moments,
            sigma_filter=sigma_filter,
            filter_diag_mat=filter_diag_mat).ravel()

    def precond(r_flat):
        """One GS pass starting from I=0 with rhs r (left preconditioner)."""
        r = r_flat.reshape(Ix, Iy, 4, S)
        src_r = r / V_arr[:, :, np.newaxis, np.newaxis]
        I_zero = np.zeros((Ix, Iy, 4, S))
        I_out, _ = sweep_pn(
            I_zero, sigma_hat, sigma_s, src_r,
            dx_arr, dy_arr, Ix, Iy, S,
            jac, stream_cache,
            ref_xlo, ref_xhi, ref_ylo, ref_yhi,
            xmask, zmask, n_gs=gmres_precond_gs,
            cell_lu_cache=cell_lu_cache,
            boundary_moments=boundary_moments,
            sigma_filter=sigma_filter,
            filter_diag_mat=filter_diag_mat)
        return I_out.ravel()

    A_op = LinearOperator((n, n), matvec=matvec, dtype=np.float64)
    M_op = LinearOperator((n, n), matvec=precond, dtype=np.float64)
    x0   = I_init.ravel()

    iters = [0]
    def callback(_):
        iters[0] += 1

    x_sol, info = gmres(A_op, b_flat, x0=x0, M=M_op,
                        rtol=tol, maxiter=maxits, callback=callback,
                        callback_type='pr_norm')
    if info != 0 and loud:
        print(f"  GMRES did not converge: info={info}, iters={iters[0]}")
    elif loud:
        print(f"  GMRES converged in {iters[0]} iterations")

    I_new = x_sol.reshape(Ix, Iy, 4, S)
    return I_new, _SQRT4PI * I_new[:, :, :, 0], iters[0]


# ===========================================================================
# Fleck-Cummings TRT solver
# ===========================================================================

def temp_solve_pn_2d(
    Ix, Iy, dx_arr, dy_arr,
    q_ext,
    sigma_func,
    scat_func,
    N_pn,
    jacobian_dir,
    EOS, invEOS,
    phi_init, T_init,
    *,
    I_init=None,
    dt_start=1e-4,
    t_end=1.0,
    reflect_xlo=False, reflect_xhi=False,
    reflect_ylo=False, reflect_yhi=False,
    tolerance=1e-8,
    maxits=300,
    W=0,
    n_gs=1,
    use_gmres=False,
    loud=False,
    print_stride=10,
    dt_max=None,
    T_floor=1e-6,
    omega_T=1.0,
    tau_T=1e-4,
    step_callback=None,
    use_cached_lu=True,
    deduplicate_lu=True,
    gmres_precond_gs=1,
    boundary_moments=None,
    filter_type='lanczos',
    filter_strength=None,
    filter_exp_order=4,
):
    """
    Gray TRT time-step loop using the P_N SCB solver.

    Parameters
    ----------
    Ix, Iy       : mesh dimensions.
    dx_arr       : (Ix,)  x-cell widths  (cm).
    dy_arr       : (Iy,)  z-cell widths  (cm).
    q_ext        : (Ix, Iy, 4) or callable(t) – volumetric source (φ units, GJ/cm³/ns).
    sigma_func   : callable T → (Ix, Iy, 4)  absorption opacity  (cm⁻¹).
    scat_func    : callable T → (Ix, Iy, 4)  scattering opacity  (cm⁻¹).
    N_pn         : P_N order (odd positive integer).
    jacobian_dir : directory containing P{N}_x.csv, P{N}_z.csv.
    EOS          : callable T → e  (material energy density, GJ/cm³).
    invEOS       : callable e → T  (keV).
    phi_init     : (Ix, Iy, 4) initial scalar flux.
    T_init       : (Ix, Iy, 4) initial material temperature (keV).
    W            : T_star outer iterations (0 = single Fleck-Cummings step).
    n_gs         : Gauss-Seidel passes per Picard step (used when use_gmres=False).
    use_gmres    : if True use matrix-free GMRES with GS preconditioner instead.

    Returns
    -------
    phi   : (Ix, Iy, 4)
    T     : (Ix, Iy, 4)
    I     : (Ix, Iy, 4, S)  full moment array at t_end
    t_now : float
    history : list of per-step dicts

    step_callback : callable or None
        If provided, called after every accepted time step as
            step_callback(t_now, phi, T)
        where phi and T are corner arrays at the updated state.
    """
    dx_arr = np.asarray(dx_arr, dtype=np.float64)
    dy_arr = np.asarray(dy_arr, dtype=np.float64)
    if dt_max is None:
        dt_max = t_end

    jac  = load_pn_jacobians(N_pn, jacobian_dir)
    S    = jac['S']
    xmask = reflect_mask_x(N_pn)
    zmask = reflect_mask_z(N_pn)
    stream_cache: dict = {}
    filter_moments = build_filter_moments_2d(
        N_pn, filter_type=filter_type, exp_order=filter_exp_order)

    # Initialise state
    phi = phi_init.astype(np.float64).copy()
    T   = T_init.astype(np.float64).copy()
    e   = EOS(T)

    # Full moment array
    if I_init is not None:
        I   = np.asarray(I_init, dtype=np.float64).copy()
        phi = _SQRT4PI * I[:, :, :, 0]     # keep phi consistent
    else:
        I = np.zeros((Ix, Iy, 4, S), dtype=np.float64)
        I[:, :, :, 0] = phi / _SQRT4PI

    t_now      = 0.0
    dt         = dt_start
    dt_old     = dt_start
    T_old2     = T.copy()
    deriv_val  = 0.0
    delta_step = 1e-3
    history    = []
    total_its  = 0
    step_num   = 0
    curr_tick  = 0
    _T_max     = 1e10

    print(f"P{N_pn} 2-D SCB solver:  S={S}  mesh={Ix}×{Iy}")

    while t_now < t_end * (1.0 - 1e-12):
        dt_old = dt
        if step_num > 2:
            dt_prop = np.sqrt(delta_step * deriv_val) if deriv_val > 0 else dt_max
            dt_prop = np.clip(dt_prop, dt_start, dt_max)
            if dt_prop > 2.0 * dt:
                dt_prop = dt * 1.5
            dt = dt_prop
        else:
            dt = dt_start

        dt = min(dt, t_end - t_now, dt_max)
        step_num += 1

        if loud:
            print(f"t = {t_now:.4e}  dt = {dt:.4e}")

        tick = int(10 * (t_now + dt) / t_end)
        if tick > curr_tick:
            curr_tick = tick
            print(curr_tick, end="", flush=True)

        icdt = 1.0 / (c * dt)
        its_step = 0

        q_now = q_ext(t_now) if callable(q_ext) else q_ext

        T_old = T.copy()
        e_old = e.copy()
        I_old = I.copy()

        # -------- T_star outer loop ----------------------------------------
        T_star   = np.minimum(T_old, _T_max)
        do_final = False
        k_outer  = 0

        while True:
            # Linearise Fleck-Cummings at T_star
            h_cv   = np.maximum(np.abs(T_star) * 1e-4, 1e-12)
            Cv     = np.maximum(
                (EOS(T_star + h_cv) -
                 EOS(np.maximum(T_star - h_cv, T_floor))) / (2.0 * h_cv),
                1e-30)
            beta_val  = 4.0 * a * T_star ** 3 / Cv
            sigma_abs = sigma_func(T_star)          # (Ix, Iy, 4)
            scat      = scat_func(T_star)
            sigma_filter = _evaluate_filter_strength(filter_strength, T_star, Ix, Iy)
            f         = 1.0 / (1.0 + beta_val * c * dt * sigma_abs)
            sigma_a   = f * sigma_abs
            sigma_s   = (1.0 - f) * sigma_abs + scat
            sigma_hat = sigma_abs + scat + icdt     # (Ix, Iy, 4)
            emission  = sigma_a * ac * T_star ** 4  # (Ix, Iy, 4), φ units

            delta_e_src = -(1.0 - f) * (EOS(T_star) - e_old) / dt

            # Fixed isotropic source in φ-units
            source_iso_fixed = emission + delta_e_src + q_now   # (Ix, Iy, 4)

            # Full moment source  (NO scatter term — scatter is folded into
            # the (0,0) diagonal of A via the sigma_hat - sigma_s correction):
            #   Q̂_00 = source_iso_fixed / √(4π) + icdt * I_old_00
            #   Q̂_lm = icdt * I_old_lm  for lm > 0
            src_fixed = icdt * I_old.copy()
            src_fixed[:, :, :, 0] += source_iso_fixed / _SQRT4PI

            # tolerance for this T_star iteration
            tau_phi = (tolerance if (W == 0 or do_final)
                       else tolerance * (1e-3 / tolerance) **
                            (1.0 - k_outer / max(W, 1)))

            I, phi, n_its = solve_pn_step(
                sigma_hat, sigma_s, src_fixed, I_old,
                dx_arr, dy_arr, Ix, Iy, S,
                jac, stream_cache,
                reflect_xlo, reflect_xhi, reflect_ylo, reflect_yhi,
                xmask, zmask,
                use_gmres=use_gmres,
                n_gs=n_gs, tol=tau_phi, maxits=maxits, loud=loud,
                use_cached_lu=use_cached_lu,
                deduplicate_lu=deduplicate_lu,
                gmres_precond_gs=gmres_precond_gs,
                boundary_moments=boundary_moments,
                sigma_filter=sigma_filter,
                filter_moments=filter_moments)

            total_its += n_its
            its_step  += n_its

            # T_star exit / update
            if W == 0 or do_final:
                break

            delta_e_star = EOS(T_star) - e_old
            e_cand = (e_old + sigma_a * dt * (phi - ac * T_star ** 4)
                      + (1.0 - f) * delta_e_star)
            T_cand = invEOS(e_cand)
            T_star_prev = T_star.copy()
            T_star = np.clip(
                (1.0 - omega_T) * T_star + omega_T * T_cand,
                T_floor, _T_max)

            eta_T = float(np.sqrt(np.mean(
                ((T_star - T_star_prev) / (np.abs(T_star_prev) + T_floor)) ** 2)))
            k_outer += 1
            if eta_T < tau_T or k_outer >= W:
                do_final = True

        # -------- Material energy update -----------------------------------
        delta_e_star = EOS(T_star) - e_old
        e = (e_old + sigma_a * dt * (phi - ac * T_star ** 4)
             + (1.0 - f) * delta_e_star)
        T = invEOS(e)
        t_now += dt

        dT_max = float(np.max(np.abs(T - T_old)))
        history.append(dict(t=t_now, dt=dt,
                            T_max=float(np.max(T)), dT_max=dT_max,
                            sweeps=its_step, T_iters=k_outer))

        if print_stride > 0 and step_num % print_stride == 0:
            print(f"  step {step_num:5d}  t={t_now:.4e} ns  dt={dt:.3e}  "
                  f"T_max={np.max(T):.6f} keV  dT_max={dT_max:.3e}  "
                  f"SI_its={its_step}  T_iters={k_outer}")
        elif step_num <= 3 or (step_num <= 20 and step_num % 5 == 0):
            print(f"  step {step_num:5d}  t={t_now:.4e} ns  dt={dt:.3e}  "
                  f"T_max={np.max(T):.6f} keV  dT_max={dT_max:.3e}  "
                  f"SI_its={its_step}  T_iters={k_outer}")

        if step_callback is not None:
            step_callback(t_now, phi, T)

        if step_num >= 2:
            T_flat = T.ravel()
            T_old_flat = T_old.ravel()
            T_old2_flat = T_old2.ravel()
            denom = np.mean(np.abs(
                T_flat / (dt**2)
                - (dt + dt_old) / (dt**2 * dt_old) * T_old_flat
                + T_old2_flat / (dt_old * dt)))
            mean_T = np.mean(T_flat)
            if denom > 0 and np.isfinite(mean_T) and np.isfinite(denom):
                deriv_val = mean_T / denom
            else:
                deriv_val = dt_max**2 / delta_step

        T_old2 = T_old.copy()
        T_old = T.copy()

    print()
    print(f"Total source iterations: {total_its}")
    return phi, T, I, t_now, history
