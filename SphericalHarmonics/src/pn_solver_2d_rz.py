"""
2-D axisymmetric cylindrical (r-z) P_N solver using simple corner balance.

Implements the cylindrical equations from Sec. 12.6:

  (1/c) dI/dt + (A_r / r) d(r I)/dr + A_z dI/dz - (G / r) I + sigma_a I = Q_hat

Discretization per corner (eq. 12.116) keeps the same Jacobians A_r=A_x and A_z
as the Cartesian solver, but replaces Cartesian face/volume factors by
cylindrical measures and adds the local reaction-like G term.

The angular-coupling matrix G is loaded from the Jacobians directory when
available, with numerical construction retained as a fallback.
"""

from __future__ import annotations

import math
import os
from typing import Callable

import numpy as np

try:
    from scipy import special as _sp_special
except Exception:
    _sp_special = None

try:
    from scipy.linalg import lu_factor, lu_solve
    _HAVE_SCIPY_LU = True
except Exception:
    lu_factor = None
    lu_solve = None
    _HAVE_SCIPY_LU = False

# Physical constants (CGS, time in ns)
c = 29.98
a = 0.01372
ac = a * c

_SQRT4PI = math.sqrt(4.0 * math.pi)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_JACOBIAN_DIR = os.path.join(_THIS_DIR, "Jacobians")

_G_CACHE: dict[int, np.ndarray] = {}


# ===========================================================================
# Moment helpers
# ===========================================================================

def _n_moments_2d(N: int) -> int:
    return (N + 1) * (N + 2) // 2


def _moment_list_2d(N: int) -> list[tuple[int, int]]:
    return [(l, m) for l in range(N + 1) for m in range(l + 1)]


def reflect_mask_r(N: int) -> np.ndarray:
    # Radial reflection behaves like x-reflection in Cartesian geometry.
    return np.array([(-1.0) ** m for _, m in _moment_list_2d(N)], dtype=float)


def reflect_mask_z(N: int) -> np.ndarray:
    return np.array([(-1.0) ** (l - m) for l, m in _moment_list_2d(N)], dtype=float)


def build_filter_moments_2d(N: int, filter_type: str = 'lanczos', exp_order: int = 4) -> np.ndarray:
    """Build f(l,N) = -log(rho(l/(N+1))) for each (l,m) entry in moment order."""
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
    f_mom[0] = 0.0
    return f_mom


def _evaluate_filter_strength(filter_strength, T_state: np.ndarray, Ir: int, Iz: int) -> np.ndarray:
    """Return filter strength as an (Ir,Iz,4) array from scalar/array/callable input."""
    if filter_strength is None:
        return np.zeros((Ir, Iz, 4), dtype=np.float64)

    raw = filter_strength(T_state) if callable(filter_strength) else filter_strength
    arr = np.asarray(raw, dtype=np.float64)

    if arr.ndim == 0:
        return np.full((Ir, Iz, 4), float(arr), dtype=np.float64)
    if arr.shape == (Ir, Iz):
        return np.repeat(arr[:, :, np.newaxis], 4, axis=2)
    if arr.shape == (Ir, Iz, 1):
        return np.repeat(arr, 4, axis=2)
    if arr.shape == (Ir, Iz, 4):
        return arr
    raise ValueError(
        "filter_strength must be scalar, callable->scalar/array, or shaped (Ir,Iz) / (Ir,Iz,4)")


# ===========================================================================
# Jacobians and cylindrical G matrix
# ===========================================================================

def load_pn_jacobians(N: int, jacobian_dir: str) -> dict[str, np.ndarray | int]:
    S = _n_moments_2d(N)
    Ax = np.loadtxt(os.path.join(jacobian_dir, f"P{N}_x.csv"), delimiter=",")[:S, :S]
    Az = np.loadtxt(os.path.join(jacobian_dir, f"P{N}_z.csv"), delimiter=",")[:S, :S]

    ax_vals, ax_vecs = np.linalg.eigh(Ax)
    az_vals, az_vecs = np.linalg.eigh(Az)
    abs_Ax = ax_vecs @ np.diag(np.abs(ax_vals)) @ ax_vecs.T
    abs_Az = az_vecs @ np.diag(np.abs(az_vals)) @ az_vecs.T

    return {
        "N": N,
        "S": S,
        "Ax": Ax,
        "Az": Az,
        "abs_Ax": abs_Ax,
        "abs_Az": abs_Az,
        "Ax_plus": 0.5 * (Ax + abs_Ax),
        "Ax_minus": 0.5 * (Ax - abs_Ax),
        "Az_plus": 0.5 * (Az + abs_Az),
        "Az_minus": 0.5 * (Az - abs_Az),
    }


def load_cylindrical_G(N: int, jacobian_dir: str | None = None) -> np.ndarray:
    """Load precomputed cylindrical G from CSV when present.

    Falls back to the numerical construction if the CSV is missing or does not
    contain the full SxS matrix for this PN order.
    """
    cached = _G_CACHE.get(N)
    if cached is not None:
        return cached

    if jacobian_dir is None:
        jacobian_dir = _DEFAULT_JACOBIAN_DIR

    S = _n_moments_2d(N)
    g_path = os.path.join(jacobian_dir, f"G_N{N}.csv")
    if os.path.exists(g_path):
        G = np.loadtxt(g_path, delimiter=",")
        G = np.atleast_2d(G)
        if G.shape[0] >= S and G.shape[1] >= S:
            G = np.asarray(G[:S, :S], dtype=float)
            _G_CACHE[N] = G
            return G

    G = build_cylindrical_G(N)
    _G_CACHE[N] = G
    return G


def _real_sh(l: int, m: int, theta: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """Real orthonormal spherical harmonics built from SciPy's complex basis."""
    if _sp_special is None:
        raise ImportError("SciPy special functions are required to build the cylindrical G matrix")

    if hasattr(_sp_special, "sph_harm"):
        # Legacy API: sph_harm(m, l, phi, theta)
        _ylm = _sp_special.sph_harm

        def Y(mm: int):
            return _ylm(mm, l, phi, theta)
    elif hasattr(_sp_special, "sph_harm_y"):
        # SciPy >= 1.18 API: sph_harm_y(l, m, theta, phi)
        _ylm = _sp_special.sph_harm_y

        def Y(mm: int):
            return _ylm(l, mm, theta, phi)
    else:
        raise ImportError("No spherical-harmonic function (sph_harm or sph_harm_y) found in SciPy")

    if m > 0:
        return np.sqrt(2.0) * ((-1.0) ** m) * np.real(Y(m))
    if m == 0:
        return np.real(Y(0))

    mm = -m
    return np.sqrt(2.0) * ((-1.0) ** mm) * np.imag(Y(mm))


def build_cylindrical_G(N: int, n_mu: int = 64, n_phi: int = 128) -> np.ndarray:
    """Compute cylindrical angular-coupling matrix G once for this P_N order."""
    moments = _moment_list_2d(N)
    S = len(moments)

    mu_nodes, w_mu = np.polynomial.legendre.leggauss(n_mu)
    phi_nodes = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    w_phi = (2.0 * math.pi) / n_phi

    MU, PHI = np.meshgrid(mu_nodes, phi_nodes, indexing="ij")
    TH = np.arccos(np.clip(MU, -1.0, 1.0))

    mu_flat = MU.ravel()
    phi_flat = PHI.ravel()
    th_flat = TH.ravel()
    w_flat = np.repeat(w_mu, n_phi) * w_phi

    Y_col = np.empty((S, mu_flat.size), dtype=float)
    Y_neg = np.empty((S, mu_flat.size), dtype=float)
    m_arr = np.empty(S, dtype=float)

    for idx, (l, m) in enumerate(moments):
        Y_col[idx, :] = _real_sh(l, m, th_flat, phi_flat)
        Y_neg[idx, :] = _real_sh(l, -m, th_flat, phi_flat)
        m_arr[idx] = float(m)

    pref = np.sqrt(np.maximum(0.0, 1.0 - mu_flat * mu_flat)) * np.sin(phi_flat)
    A = (m_arr[:, None] * Y_neg) * pref[None, :]

    G = (A * w_flat[None, :]) @ Y_col.T
    G = 0.5 * (G + G.T)

    return G


# ===========================================================================
# One cylindrical GS pass
# ===========================================================================

def _signed_radial_ops(Ax: np.ndarray, Ax_plus: np.ndarray, Ax_minus: np.ndarray, s_r: int):
    if s_r > 0:
        return Ax, Ax_plus, Ax_minus
    return -Ax, -Ax_minus, -Ax_plus


def _signed_axial_ops(Az: np.ndarray, Az_plus: np.ndarray, Az_minus: np.ndarray, s_z: int):
    if s_z > 0:
        return Az, Az_plus, Az_minus
    return -Az, -Az_minus, -Az_plus


def _build_cell_lu_cache_rz(
    sigma_hat: np.ndarray,
    sigma_s: np.ndarray,
    r_faces: np.ndarray,
    z_faces: np.ndarray,
    Ir: int,
    Iz: int,
    S: int,
    jac: dict[str, np.ndarray | int],
    G_mat: np.ndarray,
    sigma_filter: np.ndarray | None = None,
    filter_diag_mat: np.ndarray | None = None,
    deduplicate: bool = True,
):
    """Pre-factor per-cell cylindrical 4Sx4S matrices for repeated GS solves."""
    if not _HAVE_SCIPY_LU:
        return None

    Ax = jac["Ax"]
    Az = jac["Az"]
    Ax_p = jac["Ax_plus"]
    Ax_m = jac["Ax_minus"]
    Az_p = jac["Az_plus"]
    Az_m = jac["Az_minus"]

    eye_S = np.eye(S)
    on_axis = bool(r_faces[0] < 1e-15)

    cell_lu_cache = {}
    lu_by_signature = {} if deduplicate else None

    meta = [
        (+1, +1, 1, 3),
        (-1, +1, 0, 2),
        (-1, -1, 3, 1),
        (+1, -1, 2, 0),
    ]

    for j in range(Iz):
        dz = z_faces[j + 1] - z_faces[j]
        dz_h = 0.5 * dz
        for i in range(Ir):
            r_lo = r_faces[i]
            r_hi = r_faces[i + 1]
            r_ctr = 0.5 * (r_lo + r_hi)
            dr = r_hi - r_lo

            if deduplicate:
                sig = (
                    r_lo,
                    r_hi,
                    dz,
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

            V_e = 0.5 * (r_hi * r_hi - r_ctr * r_ctr) * dz_h
            V_w = 0.5 * (r_ctr * r_ctr - r_lo * r_lo) * dz_h
            Vc = [V_e, V_w, V_w, V_e]

            A_r_hi = r_hi * dz_h
            A_r_lo = 0.0 if (on_axis and i == 0) else r_lo * dz_h
            A_r_ext = [A_r_hi, A_r_lo, A_r_lo, A_r_hi]
            A_r_int = r_ctr * dz_h

            A_z_e = 0.5 * (r_hi * r_hi - r_ctr * r_ctr)
            A_z_w = 0.5 * (r_ctr * r_ctr - r_lo * r_lo)
            A_z_ext = [A_z_e, A_z_w, A_z_w, A_z_e]

            A_g = 0.25 * dr * dz

            M = np.zeros((4 * S, 4 * S), dtype=float)

            for c_idx, (s_r, s_z, pair_r, pair_z) in enumerate(meta):
                Ar, Ar_p, _ = _signed_radial_ops(Ax, Ax_p, Ax_m, s_r)
                Azs, Az_p_s, _ = _signed_axial_ops(Az, Az_p, Az_m, s_z)

                sl = slice(c_idx * S, (c_idx + 1) * S)
                sl_pr = slice(pair_r * S, (pair_r + 1) * S)
                sl_pz = slice(pair_z * S, (pair_z + 1) * S)

                Vloc = Vc[c_idx]
                A_re = A_r_ext[c_idx]
                A_ze = A_z_ext[c_idx]

                M[sl, sl] += (
                    A_re * Ar_p
                    - 0.5 * A_r_int * Ar
                    + A_ze * Az_p_s
                    - 0.5 * A_ze * Azs
                    + sigma_hat[i, j, c_idx] * Vloc * eye_S
                    - A_g * G_mat
                )
                if sigma_filter is not None and filter_diag_mat is not None:
                    M[sl, sl] += sigma_filter[i, j, c_idx] * Vloc * filter_diag_mat
                M[sl, sl_pr] += -0.5 * A_r_int * Ar
                M[sl, sl_pz] += -0.5 * A_ze * Azs
                M[c_idx * S, c_idx * S] -= sigma_s[i, j, c_idx] * Vloc

            lu_piv = lu_factor(M, overwrite_a=True, check_finite=False)
            cell_lu_cache[(i, j)] = lu_piv
            if deduplicate:
                lu_by_signature[sig] = lu_piv

    return cell_lu_cache


def _gs_pass_rz(
    I_new: np.ndarray,
    sigma_hat: np.ndarray,
    sigma_s: np.ndarray,
    src_full: np.ndarray,
    r_faces: np.ndarray,
    z_faces: np.ndarray,
    Ir: int,
    Iz: int,
    S: int,
    jac: dict[str, np.ndarray | int],
    G_mat: np.ndarray,
    reflect_rlo: bool,
    reflect_rhi: bool,
    reflect_zlo: bool,
    reflect_zhi: bool,
    rmask: np.ndarray,
    zmask: np.ndarray,
    i_seq,
    j_seq,
    boundary_moments: dict[str, np.ndarray] | None,
    cell_lu_cache=None,
    sigma_filter: np.ndarray | None = None,
    filter_diag_mat: np.ndarray | None = None,
):
    Ax = jac["Ax"]
    Az = jac["Az"]
    Ax_p = jac["Ax_plus"]
    Ax_m = jac["Ax_minus"]
    Az_p = jac["Az_plus"]
    Az_m = jac["Az_minus"]

    eye_S = np.eye(S)
    zero_S = np.zeros(S)

    rlo_bc = None if boundary_moments is None else boundary_moments.get("rlo")
    rhi_bc = None if boundary_moments is None else boundary_moments.get("rhi")
    zlo_bc = None if boundary_moments is None else boundary_moments.get("zlo")
    zhi_bc = None if boundary_moments is None else boundary_moments.get("zhi")

    on_axis = bool(r_faces[0] < 1e-15)

    for j in j_seq:
        dz = z_faces[j + 1] - z_faces[j]
        dz_h = 0.5 * dz
        for i in i_seq:
            r_lo = r_faces[i]
            r_hi = r_faces[i + 1]
            r_ctr = 0.5 * (r_lo + r_hi)
            dr = r_hi - r_lo

            V_e = 0.5 * (r_hi * r_hi - r_ctr * r_ctr) * dz_h
            V_w = 0.5 * (r_ctr * r_ctr - r_lo * r_lo) * dz_h
            Vc = [V_e, V_w, V_w, V_e]

            A_r_hi = r_hi * dz_h
            A_r_lo = 0.0 if (on_axis and i == 0) else r_lo * dz_h
            A_r_ext = [A_r_hi, A_r_lo, A_r_lo, A_r_hi]
            A_r_int = r_ctr * dz_h

            A_z_e = 0.5 * (r_hi * r_hi - r_ctr * r_ctr)
            A_z_w = 0.5 * (r_ctr * r_ctr - r_lo * r_lo)
            A_z_ext = [A_z_e, A_z_w, A_z_w, A_z_e]

            A_g = 0.25 * dr * dz

            # Corner metadata: (s_r, s_z, radial_pair, axial_pair)
            meta = [
                (+1, +1, 1, 3),  # NE
                (-1, +1, 0, 2),  # NW
                (-1, -1, 3, 1),  # SW
                (+1, -1, 2, 0),  # SE
            ]

            if cell_lu_cache is None:
                M = np.zeros((4 * S, 4 * S), dtype=float)
                for c_idx, (s_r, s_z, pair_r, pair_z) in enumerate(meta):
                    Ar, Ar_p, _ = _signed_radial_ops(Ax, Ax_p, Ax_m, s_r)
                    Azs, Az_p_s, _ = _signed_axial_ops(Az, Az_p, Az_m, s_z)

                    sl = slice(c_idx * S, (c_idx + 1) * S)
                    sl_pr = slice(pair_r * S, (pair_r + 1) * S)
                    sl_pz = slice(pair_z * S, (pair_z + 1) * S)

                    Vloc = Vc[c_idx]
                    A_re = A_r_ext[c_idx]
                    A_ze = A_z_ext[c_idx]

                    M[sl, sl] += (
                        A_re * Ar_p
                        - 0.5 * A_r_int * Ar
                        + A_ze * Az_p_s
                        - 0.5 * A_ze * Azs
                        + sigma_hat[i, j, c_idx] * Vloc * eye_S
                        - A_g * G_mat
                    )
                    if sigma_filter is not None and filter_diag_mat is not None:
                        M[sl, sl] += sigma_filter[i, j, c_idx] * Vloc * filter_diag_mat
                    M[sl, sl_pr] += -0.5 * A_r_int * Ar
                    M[sl, sl_pz] += -0.5 * A_ze * Azs

                    # Isotropic scattering correction on the (0,0) moment.
                    M[c_idx * S, c_idx * S] -= sigma_s[i, j, c_idx] * Vloc

            # External incoming states, same corner pairing as Cartesian.
            inc_rhi_NE = (
                I_new[i + 1, j, 1, :] if i < Ir - 1
                else (rhi_bc[j] if rhi_bc is not None
                      else (rmask * I_new[i, j, 0, :] if reflect_rhi else zero_S))
            )
            inc_zhi_NE = (
                I_new[i, j + 1, 3, :] if j < Iz - 1
                else (zhi_bc[i] if zhi_bc is not None
                      else (zmask * I_new[i, j, 0, :] if reflect_zhi else zero_S))
            )

            inc_rlo_NW = (
                I_new[i - 1, j, 0, :] if i > 0
                else (rlo_bc[j] if rlo_bc is not None
                      else (rmask * I_new[i, j, 1, :] if reflect_rlo else zero_S))
            )
            inc_zhi_NW = (
                I_new[i, j + 1, 2, :] if j < Iz - 1
                else (zhi_bc[i] if zhi_bc is not None
                      else (zmask * I_new[i, j, 1, :] if reflect_zhi else zero_S))
            )

            inc_rlo_SW = (
                I_new[i - 1, j, 3, :] if i > 0
                else (rlo_bc[j] if rlo_bc is not None
                      else (rmask * I_new[i, j, 2, :] if reflect_rlo else zero_S))
            )
            inc_zlo_SW = (
                I_new[i, j - 1, 1, :] if j > 0
                else (zlo_bc[i] if zlo_bc is not None
                      else (zmask * I_new[i, j, 2, :] if reflect_zlo else zero_S))
            )

            inc_rhi_SE = (
                I_new[i + 1, j, 2, :] if i < Ir - 1
                else (rhi_bc[j] if rhi_bc is not None
                      else (rmask * I_new[i, j, 3, :] if reflect_rhi else zero_S))
            )
            inc_zlo_SE = (
                I_new[i, j - 1, 0, :] if j > 0
                else (zlo_bc[i] if zlo_bc is not None
                      else (zmask * I_new[i, j, 3, :] if reflect_zlo else zero_S))
            )

            _, _, Ar_m_NE = _signed_radial_ops(Ax, Ax_p, Ax_m, +1)
            _, _, Az_m_NE = _signed_axial_ops(Az, Az_p, Az_m, +1)
            _, _, Ar_m_NW = _signed_radial_ops(Ax, Ax_p, Ax_m, -1)
            _, _, Az_m_NW = _signed_axial_ops(Az, Az_p, Az_m, +1)
            _, _, Ar_m_SW = _signed_radial_ops(Ax, Ax_p, Ax_m, -1)
            _, _, Az_m_SW = _signed_axial_ops(Az, Az_p, Az_m, -1)
            _, _, Ar_m_SE = _signed_radial_ops(Ax, Ax_p, Ax_m, +1)
            _, _, Az_m_SE = _signed_axial_ops(Az, Az_p, Az_m, -1)

            rhs = np.empty(4 * S, dtype=float)
            rhs[0 * S:1 * S] = (
                Vc[0] * src_full[i, j, 0]
                - A_r_ext[0] * (Ar_m_NE @ inc_rhi_NE)
                - A_z_ext[0] * (Az_m_NE @ inc_zhi_NE)
            )
            rhs[1 * S:2 * S] = (
                Vc[1] * src_full[i, j, 1]
                - A_r_ext[1] * (Ar_m_NW @ inc_rlo_NW)
                - A_z_ext[1] * (Az_m_NW @ inc_zhi_NW)
            )
            rhs[2 * S:3 * S] = (
                Vc[2] * src_full[i, j, 2]
                - A_r_ext[2] * (Ar_m_SW @ inc_rlo_SW)
                - A_z_ext[2] * (Az_m_SW @ inc_zlo_SW)
            )
            rhs[3 * S:4 * S] = (
                Vc[3] * src_full[i, j, 3]
                - A_r_ext[3] * (Ar_m_SE @ inc_rhi_SE)
                - A_z_ext[3] * (Az_m_SE @ inc_zlo_SE)
            )

            if cell_lu_cache is None:
                sol = np.linalg.solve(M, rhs)
            else:
                sol = lu_solve(cell_lu_cache[(i, j)], rhs, check_finite=False)
            I_new[i, j] = sol.reshape(4, S)


# ===========================================================================
# Vectorized batch sweep  (loop over j only; all i processed simultaneously)
# ===========================================================================

def _sweep_batch_rz(
    I_new: np.ndarray,
    sigma_hat: np.ndarray,
    sigma_s: np.ndarray,
    src_full: np.ndarray,
    r_faces: np.ndarray,
    z_faces: np.ndarray,
    Ir: int,
    Iz: int,
    S: int,
    jac: dict[str, np.ndarray | int],
    G_mat: np.ndarray,
    reflect_rlo: bool,
    reflect_rhi: bool,
    reflect_zlo: bool,
    reflect_zhi: bool,
    rmask: np.ndarray,
    zmask: np.ndarray,
    forward_j: bool = True,
    boundary_moments: dict[str, np.ndarray] | None = None,
    sigma_filter: np.ndarray | None = None,
    filter_diag_mat: np.ndarray | None = None,
):
    """Vectorized GS sweep: sequential over z-rows, batch over all r-cells.

    Replaces the double Python loop in ``_gs_pass_rz`` with a single loop
    over j (z-rows) and numpy batch operations over all i (r-cells).  The
    z-direction retains GS ordering (already-updated values from j-1 or j+1
    are used).  The r-direction is Jacobi within each j-row (old values from
    the same pass).  Convergence is slightly slower per sweep, but each sweep
    is ~50–200x faster for small S (e.g. SP3/SP7) because the inner Python
    loop over Ir cells is eliminated.
    """
    Ax   = jac["Ax"]
    Az   = jac["Az"]
    Ax_p = jac["Ax_plus"];  Ax_m = jac["Ax_minus"]
    Az_p = jac["Az_plus"];  Az_m = jac["Az_minus"]

    rlo_bc = None if boundary_moments is None else boundary_moments.get("rlo")
    rhi_bc = None if boundary_moments is None else boundary_moments.get("rhi")
    zlo_bc = None if boundary_moments is None else boundary_moments.get("zlo")
    zhi_bc = None if boundary_moments is None else boundary_moments.get("zhi")

    # Precompute r-geometry vectors (shape Ir).
    r_lo  = r_faces[:-1]
    r_hi  = r_faces[1:]
    r_ctr = 0.5 * (r_lo + r_hi)
    dr    = r_hi - r_lo

    # Signed Jacobians for the four corners:
    # (NE, NW, SW, SE)  → (s_r, s_z, pair_r, pair_z)
    # pair_r index: NE↔NW (1↔0), SW↔SE (3↔2)
    # pair_z index: NE↔SE (0↔3), NW↔SW (1↔2)
    _, Ar_p_NE, Ar_m_NE = _signed_radial_ops(Ax, Ax_p, Ax_m, +1)
    _, Az_p_NE, Az_m_NE = _signed_axial_ops(Az, Az_p, Az_m, +1)
    Ar_NW, Ar_p_NW, Ar_m_NW = _signed_radial_ops(Ax, Ax_p, Ax_m, -1)
    _, Az_p_NW, Az_m_NW = _signed_axial_ops(Az, Az_p, Az_m, +1)
    Ar_SW, Ar_p_SW, Ar_m_SW = _signed_radial_ops(Ax, Ax_p, Ax_m, -1)
    Az_SW, Az_p_SW, Az_m_SW = _signed_axial_ops(Az, Az_p, Az_m, -1)
    Ar_SE, Ar_p_SE, Ar_m_SE = _signed_radial_ops(Ax, Ax_p, Ax_m, +1)
    Az_SE, Az_p_SE, Az_m_SE = _signed_axial_ops(Az, Az_p, Az_m, -1)

    eye_S = np.eye(S)
    j_seq = range(Iz) if forward_j else range(Iz - 1, -1, -1)

    for j in j_seq:
        dz   = float(z_faces[j + 1] - z_faces[j])
        dz_h = 0.5 * dz

        # --- Geometry vectors for all i at this j ---  (shape: Ir)
        A_rhi  = r_hi  * dz_h                          # outer r-face area
        A_rlo  = r_lo  * dz_h                          # inner r-face area (0 at axis)
        A_rint = r_ctr * dz_h                          # half-cell r-interface
        A_ze   = 0.5 * (r_hi  ** 2 - r_ctr ** 2)      # NE/SE z-face area
        A_zw   = 0.5 * (r_ctr ** 2 - r_lo  ** 2)      # NW/SW z-face area
        A_g    = 0.25 * dr * dz                        # corner Cartesian area
        Ve     = A_ze * dz_h                           # NE/SE corner volume
        Vw     = A_zw * dz_h                           # NW/SW corner volume

        # sigma arrays for all i, corner c: (Ir,)
        sh0 = sigma_hat[:, j, 0]; sh1 = sigma_hat[:, j, 1]
        sh2 = sigma_hat[:, j, 2]; sh3 = sigma_hat[:, j, 3]
        ss0 = sigma_s[:, j, 0];   ss1 = sigma_s[:, j, 1]
        ss2 = sigma_s[:, j, 2];   ss3 = sigma_s[:, j, 3]

        # --- Gather incoming fluxes (vectorised over i) ---
        # z-hi incoming (used by NE and NW corners — forward sweep reads old j+1)
        if j + 1 < Iz:
            inc_zhi_NE = I_new[:, j + 1, 3, :]        # SE corner of cell above (Ir,S)
            inc_zhi_NW = I_new[:, j + 1, 2, :]        # SW corner of cell above
        elif zhi_bc is not None:
            inc_zhi_NE = zhi_bc                        # (Ir, S) broadcast
            inc_zhi_NW = zhi_bc
        elif reflect_zhi:
            inc_zhi_NE = zmask[None, :] * I_new[:, j, 0, :]
            inc_zhi_NW = zmask[None, :] * I_new[:, j, 1, :]
        else:
            inc_zhi_NE = np.zeros((Ir, S))
            inc_zhi_NW = np.zeros((Ir, S))

        # z-lo incoming (used by SW and SE corners — already-updated in forward j)
        if j - 1 >= 0:
            inc_zlo_SW = I_new[:, j - 1, 1, :]        # NW corner of cell below
            inc_zlo_SE = I_new[:, j - 1, 0, :]        # NE corner of cell below
        elif zlo_bc is not None:
            inc_zlo_SW = zlo_bc                        # (Ir, S) broadcast
            inc_zlo_SE = zlo_bc
        elif reflect_zlo:
            inc_zlo_SW = zmask[None, :] * I_new[:, j, 2, :]
            inc_zlo_SE = zmask[None, :] * I_new[:, j, 3, :]
        else:
            inc_zlo_SW = np.zeros((Ir, S))
            inc_zlo_SE = np.zeros((Ir, S))

        # r-hi incoming (NE, SE corners — use old values, Jacobi in r)
        inc_rhi_NE = np.zeros((Ir, S))
        inc_rhi_SE = np.zeros((Ir, S))
        if Ir > 1:
            inc_rhi_NE[:Ir - 1] = I_new[1:, j, 1, :]  # NW corner of right neighbour
            inc_rhi_SE[:Ir - 1] = I_new[1:, j, 2, :]  # SW corner of right neighbour
        if rhi_bc is not None:
            inc_rhi_NE[Ir - 1] = rhi_bc[j]
            inc_rhi_SE[Ir - 1] = rhi_bc[j]
        elif reflect_rhi:
            inc_rhi_NE[Ir - 1] = rmask * I_new[Ir - 1, j, 0, :]
            inc_rhi_SE[Ir - 1] = rmask * I_new[Ir - 1, j, 3, :]
        # else zeros (vacuum)

        # r-lo incoming (NW, SW corners — axis has zero area so value unused)
        inc_rlo_NW = np.zeros((Ir, S))
        inc_rlo_SW = np.zeros((Ir, S))
        if Ir > 1:
            inc_rlo_NW[1:] = I_new[:Ir - 1, j, 0, :]  # NE corner of left neighbour
            inc_rlo_SW[1:] = I_new[:Ir - 1, j, 3, :]  # SE corner of left neighbour
        if rlo_bc is not None:
            inc_rlo_NW[0] = rlo_bc[j]
            inc_rlo_SW[0] = rlo_bc[j]
        elif reflect_rlo:
            inc_rlo_NW[0] = rmask * I_new[0, j, 1, :]
            inc_rlo_SW[0] = rmask * I_new[0, j, 2, :]
        # else zeros (axis — zero area makes this irrelevant)

        # --- Assemble RHS for all i at once  (shape: Ir × 4S) ---
        rhs = np.zeros((Ir, 4 * S))

        # NE (corner 0): s_r=+1, s_z=+1  → r_hi, z_hi faces
        # rhs[NE] = V_e * src_NE − A_rhi*(Ar_m_NE @ inc_rhi_NE) − A_ze*(Az_m_NE @ inc_zhi_NE)
        rhs[:, 0 * S:1 * S] = (
            Ve[:, None] * src_full[:, j, 0, :]
            - A_rhi[:, None] * (inc_rhi_NE @ Ar_m_NE.T)
            - A_ze[:, None]  * (inc_zhi_NE @ Az_m_NE.T)
        )
        # NW (corner 1): s_r=-1, s_z=+1  → r_lo, z_hi faces
        rhs[:, 1 * S:2 * S] = (
            Vw[:, None] * src_full[:, j, 1, :]
            - A_rlo[:, None] * (inc_rlo_NW @ Ar_m_NW.T)
            - A_zw[:, None]  * (inc_zhi_NW @ Az_m_NW.T)
        )
        # SW (corner 2): s_r=-1, s_z=-1  → r_lo, z_lo faces
        rhs[:, 2 * S:3 * S] = (
            Vw[:, None] * src_full[:, j, 2, :]
            - A_rlo[:, None] * (inc_rlo_SW @ Ar_m_SW.T)
            - A_zw[:, None]  * (inc_zlo_SW @ Az_m_SW.T)
        )
        # SE (corner 3): s_r=+1, s_z=-1  → r_hi, z_lo faces
        rhs[:, 3 * S:4 * S] = (
            Ve[:, None] * src_full[:, j, 3, :]
            - A_rhi[:, None] * (inc_rhi_SE @ Ar_m_SE.T)
            - A_ze[:, None]  * (inc_zlo_SE @ Az_m_SE.T)
        )

        # --- Assemble cell matrices for all i  (shape: Ir × 4S × 4S) ---
        # Use broadcasting: (Ir,) × (S,S) → (Ir, S, S) via [:,None,None]
        M = np.zeros((Ir, 4 * S, 4 * S))

        # Helper to add a (Ir,S,S) contribution to a block:
        def _add(block_r, block_c, coeff_vec, mat_2d):
            """M[:, block_r, block_c] += coeff_vec[:,None,None] * mat_2d[None,:,:]"""
            M[:, block_r, block_c] += coeff_vec[:, None, None] * mat_2d[None, :, :]

        # --- NE (c=0) diagonal block: [0:S, 0:S] ---
        _add(slice(0 * S, 1 * S), slice(0 * S, 1 * S),  A_rhi,   Ar_p_NE)
        _add(slice(0 * S, 1 * S), slice(0 * S, 1 * S), -0.5*A_rint, Ax)  # Ar=Ax for NE
        _add(slice(0 * S, 1 * S), slice(0 * S, 1 * S),  A_ze,    Az_p_NE)
        _add(slice(0 * S, 1 * S), slice(0 * S, 1 * S), -0.5*A_ze, Az)    # Azs=Az for NE
        M[:, 0 * S:1 * S, 0 * S:1 * S] += sh0[:, None, None] * Ve[:, None, None] * eye_S[None]
        _add(slice(0 * S, 1 * S), slice(0 * S, 1 * S), -A_g,    G_mat)
        # NE off-diag to NW (pair_r=1) and SE (pair_z=3):
        _add(slice(0 * S, 1 * S), slice(1 * S, 2 * S), -0.5*A_rint, Ax)
        _add(slice(0 * S, 1 * S), slice(3 * S, 4 * S), -0.5*A_ze,   Az)
        # Scattering on moment 0 of NE:
        M[:, 0 * S, 0 * S] -= ss0 * Ve

        # --- NW (c=1) diagonal block: [S:2S, S:2S] ---
        _add(slice(1 * S, 2 * S), slice(1 * S, 2 * S),  A_rlo,   Ar_p_NW)
        _add(slice(1 * S, 2 * S), slice(1 * S, 2 * S), -0.5*A_rint, Ar_NW)
        _add(slice(1 * S, 2 * S), slice(1 * S, 2 * S),  A_zw,    Az_p_NW)
        _add(slice(1 * S, 2 * S), slice(1 * S, 2 * S), -0.5*A_zw,  Az)
        M[:, 1 * S:2 * S, 1 * S:2 * S] += sh1[:, None, None] * Vw[:, None, None] * eye_S[None]
        _add(slice(1 * S, 2 * S), slice(1 * S, 2 * S), -A_g,    G_mat)
        # NW off-diag to NE (pair_r=0) and SW (pair_z=2):
        _add(slice(1 * S, 2 * S), slice(0 * S, 1 * S), -0.5*A_rint, Ar_NW)
        _add(slice(1 * S, 2 * S), slice(2 * S, 3 * S), -0.5*A_zw,   Az)
        M[:, 1 * S, 1 * S] -= ss1 * Vw

        # --- SW (c=2) diagonal block: [2S:3S, 2S:3S] ---
        _add(slice(2 * S, 3 * S), slice(2 * S, 3 * S),  A_rlo,   Ar_p_SW)
        _add(slice(2 * S, 3 * S), slice(2 * S, 3 * S), -0.5*A_rint, Ar_SW)
        _add(slice(2 * S, 3 * S), slice(2 * S, 3 * S),  A_zw,    Az_p_SW)
        _add(slice(2 * S, 3 * S), slice(2 * S, 3 * S), -0.5*A_zw,  Az_SW)
        M[:, 2 * S:3 * S, 2 * S:3 * S] += sh2[:, None, None] * Vw[:, None, None] * eye_S[None]
        _add(slice(2 * S, 3 * S), slice(2 * S, 3 * S), -A_g,    G_mat)
        # SW off-diag to SE (pair_r=3) and NW (pair_z=1):
        _add(slice(2 * S, 3 * S), slice(3 * S, 4 * S), -0.5*A_rint, Ar_SW)
        _add(slice(2 * S, 3 * S), slice(1 * S, 2 * S), -0.5*A_zw,   Az_SW)
        M[:, 2 * S, 2 * S] -= ss2 * Vw

        # --- SE (c=3) diagonal block: [3S:4S, 3S:4S] ---
        _add(slice(3 * S, 4 * S), slice(3 * S, 4 * S),  A_rhi,   Ar_p_SE)
        _add(slice(3 * S, 4 * S), slice(3 * S, 4 * S), -0.5*A_rint, Ar_SE)
        _add(slice(3 * S, 4 * S), slice(3 * S, 4 * S),  A_ze,    Az_p_SE)
        _add(slice(3 * S, 4 * S), slice(3 * S, 4 * S), -0.5*A_ze,  Az_SE)
        M[:, 3 * S:4 * S, 3 * S:4 * S] += sh3[:, None, None] * Ve[:, None, None] * eye_S[None]
        _add(slice(3 * S, 4 * S), slice(3 * S, 4 * S), -A_g,    G_mat)
        # SE off-diag to SW (pair_r=2) and NE (pair_z=0):
        _add(slice(3 * S, 4 * S), slice(2 * S, 3 * S), -0.5*A_rint, Ar_SE)
        _add(slice(3 * S, 4 * S), slice(0 * S, 1 * S), -0.5*A_ze,   Az_SE)
        M[:, 3 * S, 3 * S] -= ss3 * Ve

        # Optional filter on diagonal blocks:
        if sigma_filter is not None and filter_diag_mat is not None:
            for c_idx, (Vloc, sfl) in enumerate([(Ve, sigma_filter[:, j, 0]),
                                                   (Vw, sigma_filter[:, j, 1]),
                                                   (Vw, sigma_filter[:, j, 2]),
                                                   (Ve, sigma_filter[:, j, 3])]):
                sl = slice(c_idx * S, (c_idx + 1) * S)
                M[:, sl, sl] += (sfl * Vloc)[:, None, None] * filter_diag_mat[None]

        # --- Batch solve: (Ir, 4S, 4S) x (Ir, 4S) → (Ir, 4S) ---
        sol = np.linalg.solve(M, rhs[:, :, None]).squeeze(-1)  # (Ir, 4S)
        I_new[:, j, :, :] = sol.reshape(Ir, 4, S)


def _sweep_batch_rz_fwd_bwd(
    I_new: np.ndarray,
    sigma_hat: np.ndarray,
    sigma_s: np.ndarray,
    src_full: np.ndarray,
    r_faces: np.ndarray,
    z_faces: np.ndarray,
    Ir: int,
    Iz: int,
    S: int,
    jac: dict[str, np.ndarray | int],
    G_mat: np.ndarray,
    reflect_rlo: bool,
    reflect_rhi: bool,
    reflect_zlo: bool,
    reflect_zhi: bool,
    rmask: np.ndarray,
    zmask: np.ndarray,
    n_sweeps: int = 1,
    boundary_moments: dict[str, np.ndarray] | None = None,
    sigma_filter: np.ndarray | None = None,
    filter_diag_mat: np.ndarray | None = None,
):
    """n_sweeps pairs of forward+backward batch sweeps."""
    kwargs = dict(sigma_hat=sigma_hat, sigma_s=sigma_s, src_full=src_full,
                  r_faces=r_faces, z_faces=z_faces, Ir=Ir, Iz=Iz, S=S,
                  jac=jac, G_mat=G_mat,
                  reflect_rlo=reflect_rlo, reflect_rhi=reflect_rhi,
                  reflect_zlo=reflect_zlo, reflect_zhi=reflect_zhi,
                  rmask=rmask, zmask=zmask,
                  boundary_moments=boundary_moments,
                  sigma_filter=sigma_filter, filter_diag_mat=filter_diag_mat)
    for _ in range(n_sweeps):
        _sweep_batch_rz(I_new, forward_j=True,  **kwargs)
        _sweep_batch_rz(I_new, forward_j=False, **kwargs)

    phi = _SQRT4PI * I_new[:, :, :, 0]
    return I_new, phi


# ===========================================================================
# Public solve routines
# ===========================================================================

def sweep_pn_rz(
    I_in: np.ndarray,
    sigma_hat: np.ndarray,
    sigma_s: np.ndarray,
    src_full: np.ndarray,
    r_faces: np.ndarray,
    z_faces: np.ndarray,
    Ir: int,
    Iz: int,
    S: int,
    jac: dict[str, np.ndarray | int],
    G_mat: np.ndarray,
    reflect_rlo: bool,
    reflect_rhi: bool,
    reflect_zlo: bool,
    reflect_zhi: bool,
    rmask: np.ndarray,
    zmask: np.ndarray,
    n_gs: int = 1,
    boundary_moments: dict[str, np.ndarray] | None = None,
    cell_lu_cache=None,
    sigma_filter: np.ndarray | None = None,
    filter_diag_mat: np.ndarray | None = None,
):
    I_new = I_in.copy()

    fwd_i = range(Ir)
    fwd_j = range(Iz)
    bwd_i = range(Ir - 1, -1, -1)
    bwd_j = range(Iz - 1, -1, -1)

    for _ in range(n_gs):
        _gs_pass_rz(
            I_new,
            sigma_hat,
            sigma_s,
            src_full,
            r_faces,
            z_faces,
            Ir,
            Iz,
            S,
            jac,
            G_mat,
            reflect_rlo,
            reflect_rhi,
            reflect_zlo,
            reflect_zhi,
            rmask,
            zmask,
            fwd_i,
            fwd_j,
            boundary_moments,
            cell_lu_cache=cell_lu_cache,
            sigma_filter=sigma_filter,
            filter_diag_mat=filter_diag_mat,
        )
        _gs_pass_rz(
            I_new,
            sigma_hat,
            sigma_s,
            src_full,
            r_faces,
            z_faces,
            Ir,
            Iz,
            S,
            jac,
            G_mat,
            reflect_rlo,
            reflect_rhi,
            reflect_zlo,
            reflect_zhi,
            rmask,
            zmask,
            bwd_i,
            bwd_j,
            boundary_moments,
            cell_lu_cache=cell_lu_cache,
            sigma_filter=sigma_filter,
            filter_diag_mat=filter_diag_mat,
        )

    phi = _SQRT4PI * I_new[:, :, :, 0]
    return I_new, phi


def solve_pn_step_rz(
    sigma_hat: np.ndarray,
    sigma_s: np.ndarray,
    src_full: np.ndarray,
    I_init: np.ndarray,
    r_faces: np.ndarray,
    z_faces: np.ndarray,
    Ir: int,
    Iz: int,
    S: int,
    jac: dict[str, np.ndarray | int],
    G_mat: np.ndarray,
    reflect_rlo: bool,
    reflect_rhi: bool,
    reflect_zlo: bool,
    reflect_zhi: bool,
    rmask: np.ndarray,
    zmask: np.ndarray,
    n_gs: int = 1,
    tol: float = 1e-8,
    maxits: int = 300,
    loud: bool = False,
    boundary_moments: dict[str, np.ndarray] | None = None,
    use_cached_lu: bool = True,
    deduplicate_lu: bool = True,
    sigma_filter: np.ndarray | None = None,
    filter_moments: np.ndarray | None = None,
):
    filter_diag_mat = (np.diag(filter_moments)
                       if (filter_moments is not None and sigma_filter is not None)
                       else None)

    cell_lu_cache = None
    if use_cached_lu and _HAVE_SCIPY_LU:
        cell_lu_cache = _build_cell_lu_cache_rz(
            sigma_hat,
            sigma_s,
            r_faces,
            z_faces,
            Ir,
            Iz,
            S,
            jac,
            G_mat,
            sigma_filter=sigma_filter,
            filter_diag_mat=filter_diag_mat,
            deduplicate=deduplicate_lu,
        )

    I_curr = I_init.copy()

    for k in range(maxits):
        I_new, phi_new = sweep_pn_rz(
            I_curr,
            sigma_hat,
            sigma_s,
            src_full,
            r_faces,
            z_faces,
            Ir,
            Iz,
            S,
            jac,
            G_mat,
            reflect_rlo,
            reflect_rhi,
            reflect_zlo,
            reflect_zhi,
            rmask,
            zmask,
            n_gs=n_gs,
            boundary_moments=boundary_moments,
            cell_lu_cache=cell_lu_cache,
            sigma_filter=sigma_filter,
            filter_diag_mat=filter_diag_mat,
        )

        phi_curr = _SQRT4PI * I_curr[:, :, :, 0]
        denom = float(np.max(np.abs(phi_new))) + 1e-30
        res = float(np.max(np.abs(phi_new - phi_curr))) / denom
        if loud:
            print(f"    GS it {k + 1:3d}: rel residual = {res:.3e}")

        I_curr = I_new
        if res < tol:
            return I_curr, phi_new, k + 1

    return I_curr, _SQRT4PI * I_curr[:, :, :, 0], maxits


def solve_pn_step_rz_batch(
    sigma_hat: np.ndarray,
    sigma_s: np.ndarray,
    src_full: np.ndarray,
    I_init: np.ndarray,
    r_faces: np.ndarray,
    z_faces: np.ndarray,
    Ir: int,
    Iz: int,
    S: int,
    jac: dict[str, np.ndarray | int],
    G_mat: np.ndarray,
    reflect_rlo: bool,
    reflect_rhi: bool,
    reflect_zlo: bool,
    reflect_zhi: bool,
    rmask: np.ndarray,
    zmask: np.ndarray,
    n_gs: int = 1,
    tol: float = 1e-8,
    maxits: int = 300,
    loud: bool = False,
    boundary_moments: dict[str, np.ndarray] | None = None,
    use_cached_lu: bool = True,       # accepted for API compatibility, unused
    deduplicate_lu: bool = True,      # accepted for API compatibility, unused
    sigma_filter: np.ndarray | None = None,
    filter_moments: np.ndarray | None = None,
):
    """Like solve_pn_step_rz but uses the vectorised batch sweep.

    Avoids the per-cell Python loop in _gs_pass_rz by processing all r-cells
    simultaneously per z-row.  Recommended for small S (SP3/SP7) where Python
    overhead otherwise dominates.  The z-direction retains GS ordering;
    r-direction is Jacobi within each z-row (convergence is similar in
    practice because the problem is dominated by z-streaming).
    """
    filter_diag_mat = (np.diag(filter_moments)
                       if (filter_moments is not None and sigma_filter is not None)
                       else None)

    I_curr = I_init.copy()
    sweep_kwargs = dict(
        sigma_hat=sigma_hat, sigma_s=sigma_s, src_full=src_full,
        r_faces=r_faces, z_faces=z_faces, Ir=Ir, Iz=Iz, S=S,
        jac=jac, G_mat=G_mat,
        reflect_rlo=reflect_rlo, reflect_rhi=reflect_rhi,
        reflect_zlo=reflect_zlo, reflect_zhi=reflect_zhi,
        rmask=rmask, zmask=zmask,
        boundary_moments=boundary_moments,
        sigma_filter=sigma_filter, filter_diag_mat=filter_diag_mat,
    )

    for k in range(maxits):
        I_new = I_curr.copy()
        _sweep_batch_rz(I_new, forward_j=True,  **sweep_kwargs)
        _sweep_batch_rz(I_new, forward_j=False, **sweep_kwargs)
        phi_new = _SQRT4PI * I_new[:, :, :, 0]

        phi_curr = _SQRT4PI * I_curr[:, :, :, 0]
        denom = float(np.max(np.abs(phi_new))) + 1e-30
        res = float(np.max(np.abs(phi_new - phi_curr))) / denom
        if loud:
            print(f"    batch it {k + 1:3d}: rel residual = {res:.3e}")

        I_curr = I_new
        if res < tol:
            return I_curr, phi_new, k + 1

    return I_curr, _SQRT4PI * I_curr[:, :, :, 0], maxits


def temp_solve_pn_rz(
    Ir: int,
    Iz: int,
    dr_arr: np.ndarray,
    dz_arr: np.ndarray,
    r_faces: np.ndarray,
    z_faces: np.ndarray,
    q_ext,
    sigma_func: Callable[[np.ndarray], np.ndarray],
    scat_func: Callable[[np.ndarray], np.ndarray],
    N_pn: int,
    jacobian_dir: str,
    EOS: Callable[[np.ndarray], np.ndarray],
    invEOS: Callable[[np.ndarray], np.ndarray],
    phi_init: np.ndarray,
    T_init: np.ndarray,
    *,
    I_init: np.ndarray | None = None,
    dt_start: float = 1e-4,
    t_end: float = 1.0,
    reflect_rlo: bool = False,
    reflect_rhi: bool = False,
    reflect_zlo: bool = False,
    reflect_zhi: bool = False,
    tolerance: float = 1e-8,
    maxits: int = 300,
    W: int = 0,
    n_gs: int = 1,
    loud: bool = False,
    print_stride: int = 10,
    dt_max: float | None = None,
    T_floor: float = 1e-6,
    omega_T: float = 1.0,
    tau_T: float = 1e-4,
    step_callback=None,
    boundary_moments: dict[str, np.ndarray] | None = None,
    use_cached_lu: bool = True,
    deduplicate_lu: bool = True,
    filter_type: str = 'lanczos',
    filter_strength=None,
    filter_exp_order: int = 4,
):
    """Gray TRT time-step loop for cylindrical P_N SCB."""
    dr_arr = np.asarray(dr_arr, dtype=float)
    dz_arr = np.asarray(dz_arr, dtype=float)
    r_faces = np.asarray(r_faces, dtype=float)
    z_faces = np.asarray(z_faces, dtype=float)

    if dt_max is None:
        dt_max = t_end

    jac = load_pn_jacobians(N_pn, jacobian_dir)
    S = int(jac["S"])
    G_mat = load_cylindrical_G(N_pn, jacobian_dir)
    filter_moments = build_filter_moments_2d(
        N_pn, filter_type=filter_type, exp_order=filter_exp_order)

    rmask = reflect_mask_r(N_pn)
    zmask = reflect_mask_z(N_pn)

    phi = phi_init.astype(float).copy()
    T = T_init.astype(float).copy()
    e = EOS(T)

    if I_init is not None:
        I = np.asarray(I_init, dtype=float).copy()
        phi = _SQRT4PI * I[:, :, :, 0]
    else:
        I = np.zeros((Ir, Iz, 4, S), dtype=float)
        I[:, :, :, 0] = phi / _SQRT4PI

    t_now = 0.0
    dt = dt_start
    dt_old = dt_start
    T_old2 = T.copy()
    deriv_val = 0.0
    delta_step = 1e-3
    history: list[dict[str, float | int]] = []
    total_its = 0
    step_num = 0
    curr_tick = 0
    _T_max = 1e10

    print(f"P{N_pn} r-z cylindrical SCB solver:  S={S}  mesh={Ir}x{Iz}")

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

        T_star = np.minimum(T_old, _T_max)
        do_final = False
        k_outer = 0

        while True:
            h_cv = np.maximum(np.abs(T_star) * 1e-4, 1e-12)
            Cv = np.maximum(
                (EOS(T_star + h_cv) - EOS(np.maximum(T_star - h_cv, T_floor))) / (2.0 * h_cv),
                1e-30,
            )

            beta_val = 4.0 * a * T_star ** 3 / Cv
            sigma_abs = sigma_func(T_star)
            scat = scat_func(T_star)
            sigma_filter = _evaluate_filter_strength(filter_strength, T_star, Ir, Iz)

            f = 1.0 / (1.0 + beta_val * c * dt * sigma_abs)
            sigma_a = f * sigma_abs
            sigma_s = (1.0 - f) * sigma_abs + scat
            sigma_hat = sigma_abs + scat + icdt
            emission = sigma_a * ac * T_star ** 4

            delta_e_src = -(1.0 - f) * (EOS(T_star) - e_old) / dt
            source_iso_fixed = emission + delta_e_src + q_now

            src_fixed = icdt * I_old.copy()
            src_fixed[:, :, :, 0] += source_iso_fixed / _SQRT4PI

            tau_phi = (
                tolerance
                if (W == 0 or do_final)
                else tolerance * (1e-3 / tolerance) ** (1.0 - k_outer / max(W, 1))
            )

            I, phi, n_its = solve_pn_step_rz(
                sigma_hat,
                sigma_s,
                src_fixed,
                I_old,
                r_faces,
                z_faces,
                Ir,
                Iz,
                S,
                jac,
                G_mat,
                reflect_rlo,
                reflect_rhi,
                reflect_zlo,
                reflect_zhi,
                rmask,
                zmask,
                n_gs=n_gs,
                tol=tau_phi,
                maxits=maxits,
                loud=loud,
                boundary_moments=boundary_moments,
                use_cached_lu=use_cached_lu,
                deduplicate_lu=deduplicate_lu,
                sigma_filter=sigma_filter,
                filter_moments=filter_moments,
            )

            total_its += n_its
            its_step += n_its

            if W == 0 or do_final:
                break

            delta_e_star = EOS(T_star) - e_old
            e_cand = e_old + sigma_a * dt * (phi - ac * T_star ** 4) + (1.0 - f) * delta_e_star
            T_cand = invEOS(e_cand)

            T_star_prev = T_star.copy()
            T_star = np.clip((1.0 - omega_T) * T_star + omega_T * T_cand, T_floor, _T_max)

            eta_T = float(np.sqrt(np.mean(((T_star - T_star_prev) / (np.abs(T_star_prev) + T_floor)) ** 2)))
            k_outer += 1
            if eta_T < tau_T or k_outer >= W:
                do_final = True

        delta_e_star = EOS(T_star) - e_old
        e = e_old + sigma_a * dt * (phi - ac * T_star ** 4) + (1.0 - f) * delta_e_star
        T = invEOS(e)
        t_now += dt

        dT_max = float(np.max(np.abs(T - T_old)))
        history.append(
            {
                "t": float(t_now),
                "dt": float(dt),
                "T_max": float(np.max(T)),
                "dT_max": dT_max,
                "sweeps": int(its_step),
                "T_iters": int(k_outer),
            }
        )

        if print_stride > 0 and step_num % print_stride == 0:
            print(
                f"  step {step_num:5d}  t={t_now:.4e} ns  dt={dt:.3e}  "
                f"T_max={np.max(T):.6f} keV  dT_max={dT_max:.3e}  "
                f"SI_its={its_step}  T_iters={k_outer}"
            )
        elif step_num <= 3 or (step_num <= 20 and step_num % 5 == 0):
            print(
                f"  step {step_num:5d}  t={t_now:.4e} ns  dt={dt:.3e}  "
                f"T_max={np.max(T):.6f} keV  dT_max={dT_max:.3e}  "
                f"SI_its={its_step}  T_iters={k_outer}"
            )

        if step_callback is not None:
            step_callback(t_now, phi, T)

        if step_num >= 2:
            T_flat = T.ravel()
            T_old_flat = T_old.ravel()
            T_old2_flat = T_old2.ravel()
            denom = np.mean(
                np.abs(
                    T_flat / (dt ** 2)
                    - (dt + dt_old) / (dt ** 2 * dt_old) * T_old_flat
                    + T_old2_flat / (dt_old * dt)
                )
            )
            mean_T = np.mean(T_flat)
            if denom > 0 and np.isfinite(mean_T) and np.isfinite(denom):
                deriv_val = mean_T / denom
            else:
                deriv_val = dt_max ** 2 / delta_step

        T_old2 = T_old.copy()

    print()
    print(f"Total source iterations: {total_its}")
    return phi, T, I, t_now, history
