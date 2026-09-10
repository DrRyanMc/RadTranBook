"""
2-D axisymmetric cylindrical (r-z) SP_N solver.

Uses the same cylindrical SCB GS-sweep machinery as pn_solver_2d_rz.py but
with analytically constructed SP_N Jacobians.

The cylindrical SP_N equations are:
    even rows:      (1/c) dI/dt + (A_r/r) d(rI)/dr + A_z dI/dz + Σ I = Q
    odd r rows:     (1/c) dI/dt +  A_r    dI/dr     +             Σ I = Q
    odd z rows:     (1/c) dI/dt +                A_z dI/dz +      Σ I = Q

SP_N has no full-P_N cylindrical angular-coupling G matrix.  The existing
PN r-z SCB machinery writes the radial operator in conservative cylindrical
form for every row, so this wrapper supplies an SP_N-only source correction
that removes the extra A_r I/r term from the odd radial-gradient equations.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

# ---- Reach solvers ----------------------------------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SPN_DIR  = os.path.dirname(_THIS_DIR)   # SPN/
_SH_DIR   = os.path.dirname(_SPN_DIR)    # SphericalHarmonics/
for _d in (_SPN_DIR, _SH_DIR):
    if _d not in sys.path:
        sys.path.insert(0, _d)

from .spn_solver_2d import (
    build_spn_jacobians,
    reflect_mask_x_spn,
    reflect_mask_z_spn,
    build_spn_filter_moments,
    _SQRT4PI, c, a, ac,
    _evaluate_filter_strength,
)
from ...src.pn_solver_2d_rz import solve_pn_step_rz_batch as solve_pn_step_rz, _real_sh

__all__ = ['temp_solve_spn_rz', 'build_spn_cylindrical_G']


# ===========================================================================
# SPN moment list and cylindrical G matrix
# ===========================================================================

def _spn_moment_list(N: int) -> list[tuple[int, int]]:
    """List of (l, m) pairs in SPN state-vector order.

    Even scalars use m=0; odd vector pairs use (m=1, m=0) for (r, z).
    For SP3: [(0,0), (1,1), (1,0), (2,0), (3,1), (3,0)]
    """
    moments: list[tuple[int, int]] = []
    for n in range(N + 1):
        if n % 2 == 0:
            moments.append((n, 0))
        else:
            moments.append((n, 1))   # r-component
            moments.append((n, 0))   # z-component
    return moments


_G_SPN_CACHE: dict[int, np.ndarray] = {}


def build_spn_cylindrical_G(N: int, n_mu: int = 64, n_phi: int = 128) -> np.ndarray:
    """Compute the cylindrical angular-coupling G matrix for SP_N.

    Uses the same Gauss-Legendre quadrature formula as
    ``pn_solver_2d_rz.build_cylindrical_G`` but restricted to the SP_N moment
    subspace: only (l, 0) and (l, 1) moments are included.

    The result is cached after the first call for a given N.
    """
    cached = _G_SPN_CACHE.get(N)
    if cached is not None:
        return cached

    moments = _spn_moment_list(N)
    S = len(moments)  # = 3*(N+1)//2

    mu_nodes, w_mu = np.polynomial.legendre.leggauss(n_mu)
    phi_nodes = np.linspace(0.0, 2.0 * math.pi, n_phi, endpoint=False)
    w_phi = 2.0 * math.pi / n_phi

    MU, PHI = np.meshgrid(mu_nodes, phi_nodes, indexing="ij")
    TH = np.arccos(np.clip(MU, -1.0, 1.0))

    mu_flat  = MU.ravel()
    phi_flat = PHI.ravel()
    th_flat  = TH.ravel()
    w_flat   = np.repeat(w_mu, n_phi) * w_phi

    Y_col = np.empty((S, mu_flat.size), dtype=float)
    Y_neg = np.empty((S, mu_flat.size), dtype=float)
    m_arr = np.empty(S, dtype=float)

    for idx, (l, m) in enumerate(moments):
        Y_col[idx, :] = _real_sh(l,  m, th_flat, phi_flat)
        Y_neg[idx, :] = _real_sh(l, -m, th_flat, phi_flat)
        m_arr[idx] = float(m)

    pref = np.sqrt(np.maximum(0.0, 1.0 - mu_flat ** 2)) * np.sin(phi_flat)
    A    = (m_arr[:, None] * Y_neg) * pref[None, :]

    G = (A * w_flat[None, :]) @ Y_col.T
    G = 0.5 * (G + G.T)

    _G_SPN_CACHE[N] = G
    return G


def _build_spn_radial_gradient_correction(N: int, Ax: np.ndarray) -> np.ndarray:
    """Rows that are radial gradients need -A_r I/r, not divergence form.

    The shared r-z PN SCB kernel discretizes ``A_r/r * d(rI)/dr`` by using
    cylindrical face areas.  That is correct for SP_N scalar/even rows, whose
    radial terms are divergences of odd radial moments.  For odd radial-vector
    rows, however, the SP_N equation contains ``A_r * dI/dr``.  After
    multiplying by the cylindrical volume, this differs from the conservative
    form by ``-int A_r I dr dz``.  The PN kernel already subtracts
    ``A_g * G_mat`` on each corner, so returning the selected rows of ``Ax``
    through that slot applies exactly this correction.
    """
    correction = np.zeros_like(Ax)
    row = 0
    for n in range(N + 1):
        if n % 2 == 0:
            row += 1
        else:
            correction[row, :] = Ax[row, :]
            row += 2
    return correction


def temp_solve_spn_rz(
    Ir: int,
    Iz: int,
    dr_arr,
    dz_arr,
    r_faces,
    z_faces,
    q_ext,
    sigma_func,
    scat_func,
    N_spn: int,
    EOS,
    invEOS,
    phi_init,
    T_init,
    *,
    I_init=None,
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
    dt_max=None,
    T_floor: float = 1e-6,
    omega_T: float = 1.0,
    tau_T: float = 1e-4,
    step_callback=None,
    state_callback=None,
    boundary_moments=None,
    use_cached_lu: bool = True,
    deduplicate_lu: bool = True,
    filter_type: str = 'lanczos',
    filter_strength=None,
    filter_exp_order: int = 4,
    stability_eps: float = 1.0,
):
    """Gray TRT time-step loop for cylindrical r-z SP_N SCB.

    Interface mirrors ``temp_solve_pn_rz`` in pn_solver_2d_rz but without
    a ``jacobian_dir`` argument (Jacobians are built analytically).

    Parameters
    ----------
    N_spn : int
        SP_N order; must be odd (1, 3, 5, …).
    stability_eps : float
        Cross-direction dissipation coefficient (default 1.0).  The SP_N
        Jacobians have zero rows for "non-streaming" components (e.g., the
        z-components of odd moments have zero rows in A_r), which gives
        those components zero radial upwind dissipation.  Setting
        stability_eps > 0 blends that fraction of each |A| into the other
        direction, preventing checkerboard instabilities in cylindrical
        geometry.  Set to 0.0 to recover the unmodified Jacobians.
    All other parameters are identical to ``temp_solve_pn_rz``.
    """
    dr_arr  = np.asarray(dr_arr, dtype=float)
    dz_arr  = np.asarray(dz_arr, dtype=float)
    r_faces = np.asarray(r_faces, dtype=float)
    z_faces = np.asarray(z_faces, dtype=float)

    if dt_max is None:
        dt_max = t_end

    # Analytic SP_N Jacobians.  SPN has no full-PN angular G term, but the
    # shared cylindrical PN kernel needs this row-selective correction because
    # odd radial-vector rows contain gradients, not divergences.
    jac   = build_spn_jacobians(N_spn)
    S     = jac['S']
    G_mat = _build_spn_radial_gradient_correction(N_spn, jac['Ax'])

    # ---- Optional cross-direction dissipation (stability_eps) ---------------
    # The SP_N Jacobians have zero rows for z-components (in A_x) and for
    # r-components (in A_z).  Blending a fraction of |A_z| into |A_x| (and
    # vice-versa) adds upwind dissipation for those decoupled rows.
    if stability_eps > 0.0:
        _Ax = jac['Ax'];  _Az = jac['Az']
        _aAx = jac['abs_Ax'];  _aAz = jac['abs_Az']
        aAx_s = _aAx + stability_eps * _aAz
        aAz_s = _aAz + stability_eps * _aAx
        jac = dict(jac)           # shallow copy before mutation
        jac['abs_Ax']  = aAx_s
        jac['abs_Az']  = aAz_s
        jac['Ax_plus']  = 0.5 * (_Ax + aAx_s)
        jac['Ax_minus'] = 0.5 * (_Ax - aAx_s)
        jac['Az_plus']  = 0.5 * (_Az + aAz_s)
        jac['Az_minus'] = 0.5 * (_Az - aAz_s)
    # -------------------------------------------------------------------------

    rmask = reflect_mask_x_spn(N_spn)   # r-boundary: same sign rule as x in Cartesian
    zmask = reflect_mask_z_spn(N_spn)   # z-boundary sign rule

    filter_moments = build_spn_filter_moments(
        N_spn, filter_type=filter_type, exp_order=filter_exp_order)

    phi = phi_init.astype(float).copy()
    T   = T_init.astype(float).copy()
    e   = EOS(T)

    if I_init is not None:
        I   = np.asarray(I_init, dtype=float).copy()
        phi = _SQRT4PI * I[:, :, :, 0]
    else:
        I = np.zeros((Ir, Iz, 4, S), dtype=float)
        I[:, :, :, 0] = phi / _SQRT4PI

    t_now      = 0.0
    dt         = dt_start
    dt_old     = dt_start
    T_old2     = T.copy()
    deriv_val  = 0.0
    delta_step = 1e-3
    history: list[dict] = []
    total_its  = 0
    step_num   = 0
    curr_tick  = 0
    _T_max     = 1e10

    print(f"SP{N_spn} r-z cylindrical SCB solver:  S={S}  mesh={Ir}x{Iz}")

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
            print(curr_tick, end='', flush=True)

        icdt     = 1.0 / (c * dt)
        its_step = 0

        q_now = q_ext(t_now) if callable(q_ext) else q_ext
        T_old = T.copy()
        e_old = e.copy()
        I_old = I.copy()

        T_star   = np.minimum(T_old, _T_max)
        do_final = False
        k_outer  = 0

        while True:
            h_cv = np.maximum(np.abs(T_star) * 1e-4, 1e-12)
            Cv = np.maximum(
                (EOS(T_star + h_cv) - EOS(np.maximum(T_star - h_cv, T_floor)))
                / (2.0 * h_cv), 1e-30)

            beta_val  = 4.0 * a * T_star ** 3 / Cv
            sigma_abs = sigma_func(T_star)
            scat      = scat_func(T_star)
            sigma_filter = _evaluate_filter_strength(filter_strength, T_star, Ir, Iz)

            f        = 1.0 / (1.0 + beta_val * c * dt * sigma_abs)
            sigma_a  = f * sigma_abs
            sigma_s  = (1.0 - f) * sigma_abs + scat
            sigma_hat = sigma_abs + scat + icdt
            emission  = sigma_a * ac * T_star ** 4

            delta_e_src      = -(1.0 - f) * (EOS(T_star) - e_old) / dt
            source_iso_fixed = emission + delta_e_src + q_now

            src_fixed = icdt * I_old.copy()
            src_fixed[:, :, :, 0] += source_iso_fixed / _SQRT4PI

            tau_phi = (tolerance if (W == 0 or do_final)
                       else tolerance * (1e-3 / tolerance) **
                            (1.0 - k_outer / max(W, 1)))

            I, phi, n_its = solve_pn_step_rz(
                sigma_hat, sigma_s, src_fixed, I_old,
                r_faces, z_faces, Ir, Iz, S,
                jac, G_mat,           # SP_N Jacobians + zero G
                reflect_rlo, reflect_rhi, reflect_zlo, reflect_zhi,
                rmask, zmask,
                n_gs=n_gs, tol=tau_phi, maxits=maxits, loud=loud,
                boundary_moments=boundary_moments,
                use_cached_lu=use_cached_lu,
                deduplicate_lu=deduplicate_lu,
                sigma_filter=sigma_filter,
                filter_moments=filter_moments)

            total_its += n_its
            its_step  += n_its

            if W == 0 or do_final:
                break

            delta_e_star = EOS(T_star) - e_old
            e_cand = (e_old + sigma_a * dt * (phi - ac * T_star ** 4)
                      + (1.0 - f) * delta_e_star)
            T_cand = invEOS(e_cand)
            T_star_prev = T_star.copy()
            T_star = np.clip(
                (1.0 - omega_T) * T_star + omega_T * T_cand, T_floor, _T_max)
            eta_T = float(np.sqrt(np.mean(
                ((T_star - T_star_prev) / (np.abs(T_star_prev) + T_floor)) ** 2)))
            k_outer += 1
            if eta_T < tau_T or k_outer >= W:
                do_final = True

        delta_e_star = EOS(T_star) - e_old
        e = (e_old + sigma_a * dt * (phi - ac * T_star ** 4)
             + (1.0 - f) * delta_e_star)
        # Clip e to be non-negative: SPN can produce slightly negative φ
        # in thin regions; without clipping the tiny-cv invEOS can produce
        # runaway negative temperatures.
        e = np.maximum(e, 1e-30)
        T   = np.maximum(invEOS(e), T_floor)
        t_now += dt

        dT_max = float(np.max(np.abs(T - T_old)))
        entry = {
            't':      float(t_now),
            'dt':     float(dt),
            'T_max':  float(np.max(T)),
            'dT_max': dT_max,
            'sweeps': int(its_step),
            'T_iters': int(k_outer),
        }
        history.append(entry)

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

        if state_callback is not None:
            state_callback(t_now, dt, phi, T, I, entry)

        if step_num >= 2:
            T_flat     = T.ravel()
            T_old_flat = T_old.ravel()
            T_old2_flat = T_old2.ravel()
            denom = np.mean(np.abs(
                T_flat / (dt ** 2)
                - (dt + dt_old) / (dt ** 2 * dt_old) * T_old_flat
                + T_old2_flat / (dt_old * dt)))
            mean_T = np.mean(T_flat)
            if denom > 0 and np.isfinite(mean_T) and np.isfinite(denom):
                deriv_val = mean_T / denom
            else:
                deriv_val = dt_max ** 2 / delta_step

        T_old2 = T_old.copy()

    print()
    print(f"Total source iterations: {total_its}")
    return phi, T, I, t_now, history
