"""
2-D Cartesian Simplified Spherical Harmonics (SP_N) solver.

Uses the same Simple Corner Balance discretization and Fleck–Cummings TRT
time-loop as pn_solver_2d.py, but with analytically constructed Jacobian
matrices drawn from the 1-D P_N recurrence coefficients n/(2n+1) and
(n+1)/(2n+1), rather than spherical-harmonic coupling integrals.

Unknown ordering for odd order N  (S = 3(N+1)/2 per corner):
    I = [φ₀, φ₁_x, φ₁_z, φ₂, φ₃_x, φ₃_z, ..., φ_{N-1}, φ_N_x, φ_N_z]

Key properties
--------------
- No CSV Jacobian files are needed.
- In 1-D slab geometry, SP_N and P_N are mathematically equivalent, enabling
  direct numerical validation.
- The scalar flux is  φ = √(4π) I[…, 0],  consistent with pn_solver_2d.
- Boundary reflection masks differ from P_N: at an x-boundary the x-component
  of every odd vector moment changes sign; at a z-boundary the z-components do.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np

# ---- Reach the PN SCB machinery one level up --------------------------------
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_SH_DIR = os.path.dirname(_THIS_DIR)
if _SH_DIR not in sys.path:
    sys.path.insert(0, _SH_DIR)

from ...src.pn_solver_2d import (
    _SQRT4PI, c, a, ac,
    solve_pn_step,
    _evaluate_filter_strength,
)

# Physical constants re-exported for convenience
__all__ = [
    'c', 'a', 'ac',
    'build_spn_jacobians',
    '_spn_moment_orders',
    'reflect_mask_x_spn',
    'reflect_mask_z_spn',
    'build_spn_filter_moments',
    'temp_solve_spn_2d',
]


# ===========================================================================
# Index helper
# ===========================================================================

def _spn_row(n: int, comp: int = 0) -> int:
    """Row index in the SPN state vector for moment n, component comp.

    comp = 0 : scalar (for even n) or x-component (for odd n)
    comp = 1 : z-component (for odd n only)
    """
    r = 0
    for k in range(n):
        r += 1 if k % 2 == 0 else 2
    if n % 2 == 1 and comp == 1:
        r += 1
    return r


def _spn_moment_orders(N: int) -> list[int]:
    """Return the spherical-harmonic order l for each entry of the SPN vector.

    For SP_3: [0, 1, 1, 2, 3, 3]
    """
    orders: list[int] = []
    for n in range(N + 1):
        if n % 2 == 0:
            orders.append(n)
        else:
            orders.append(n)   # x/r-component
            orders.append(n)   # z-component
    return orders


# ===========================================================================
# Analytic Jacobian builder
# ===========================================================================

def build_spn_jacobians(N: int) -> dict:
    """Build SP_N Jacobian matrices analytically for odd order N.

    Parameters
    ----------
    N : int
        SP_N order; must be odd (1, 3, 5, …).

    Returns
    -------
    dict
        Same keys as ``load_pn_jacobians`` in pn_solver_2d:
        N, S, Ax, Az, abs_Ax, abs_Az, Ax_plus, Ax_minus, Az_plus, Az_minus.

    Notes
    -----
    Non-zero pattern of A_x for SP_3 (ordering [φ₀, φ₁_x, φ₁_z, φ₂, φ₃_x, φ₃_z]):

        A_x = [[0,   1,   0,   0,   0,   0  ],
                [1/3, 0,   0,   2/3, 0,   0  ],
                [0,   0,   0,   0,   0,   0  ],
                [0,   2/5, 0,   0,   3/5, 0  ],
                [0,   0,   0,   3/7, 0,   0  ],
                [0,   0,   0,   0,   0,   0  ]]

    A_z has the same structure with z-components (column 2, 5) coupling to scalars.
    """
    if N % 2 != 1:
        raise ValueError(f"SPN order N={N} must be odd (1, 3, 5, ...)")

    S = 3 * (N + 1) // 2
    Ax = np.zeros((S, S))
    Az = np.zeros((S, S))

    for n in range(N + 1):
        if n == 0:
            # φ₀ eqn: ∂φ₁_x/∂x + ∂φ₁_z/∂z
            Ax[_spn_row(0), _spn_row(1, 0)] = 1.0
            Az[_spn_row(0), _spn_row(1, 1)] = 1.0

        elif n % 2 == 1:  # odd n – vector equation
            c_m = n / (2 * n + 1)
            c_p = (n + 1) / (2 * n + 1)
            rx = _spn_row(n, 0)          # x-component row
            rz = _spn_row(n, 1)          # z-component row
            rm1 = _spn_row(n - 1)        # φ_{n-1} (even scalar)

            # x-component: c_m ∂φ_{n-1}/∂x [+ c_p ∂φ_{n+1}/∂x]
            Ax[rx, rm1] = c_m
            # z-component: c_m ∂φ_{n-1}/∂z [+ c_p ∂φ_{n+1}/∂z]
            Az[rz, rm1] = c_m
            if n + 1 <= N:
                rp1 = _spn_row(n + 1)    # φ_{n+1} (even scalar)
                Ax[rx, rp1] = c_p
                Az[rz, rp1] = c_p

        else:  # even n > 0 – scalar equation, couples to ∇·φ⃗_{n±1}
            c_m = n / (2 * n + 1)
            c_p = (n + 1) / (2 * n + 1)
            rn   = _spn_row(n)
            rm1x = _spn_row(n - 1, 0)   # x-component of φ_{n-1} (odd)
            rm1z = _spn_row(n - 1, 1)   # z-component of φ_{n-1} (odd)

            Ax[rn, rm1x] = c_m
            Az[rn, rm1z] = c_m
            if n + 1 <= N:
                rp1x = _spn_row(n + 1, 0)
                rp1z = _spn_row(n + 1, 1)
                Ax[rn, rp1x] = c_p
                Az[rn, rp1z] = c_p

    # Compute |Ax| and |Az| with a symmetry-restoring similarity transform.
    # The SPN recurrence uses raw Legendre coefficients that make Ax/Az
    # non-symmetric (e.g. Ax[0,1]=1 but Ax[1,0]=1/3 for SP3).  Calling
    # eigh() directly on a non-symmetric matrix uses only the lower triangle,
    # giving wrong eigenvalues (±0.636 instead of the correct ±0.861 for SP3).
    # The transform  Ax_sym = diag(S) * Ax * diag(S)^{-1}  with  S_k = sqrt(2l_k+1)
    # produces a symmetric matrix with the correct eigenvalues (Gauss-Legendre
    # nodes).  The flux-splitting matrices are recovered via the inverse
    # transform:  abs_Ax = diag(S)^{-1} * abs(Ax_sym) * diag(S).
    l_orders = np.array(_spn_moment_orders(N), dtype=float)
    S_sc     = np.sqrt(2.0 * l_orders + 1.0)   # shape (S,)
    Ax_sym   = (S_sc[:, None] * Ax) / S_sc[None, :]
    Az_sym   = (S_sc[:, None] * Az) / S_sc[None, :]

    ax_vals, ax_vecs = np.linalg.eigh(Ax_sym)
    az_vals, az_vecs = np.linalg.eigh(Az_sym)
    abs_Ax_sym = ax_vecs @ np.diag(np.abs(ax_vals)) @ ax_vecs.T
    abs_Az_sym = az_vecs @ np.diag(np.abs(az_vals)) @ az_vecs.T

    # Inverse similarity: abs_Ax[i,j] = abs_Ax_sym[i,j] * S[j] / S[i]
    abs_Ax = abs_Ax_sym * (S_sc[None, :] / S_sc[:, None])
    abs_Az = abs_Az_sym * (S_sc[None, :] / S_sc[:, None])

    return dict(
        N=N, S=S,
        Ax=Ax,          Az=Az,
        abs_Ax=abs_Ax,  abs_Az=abs_Az,
        Ax_plus=0.5 * (Ax + abs_Ax),
        Ax_minus=0.5 * (Ax - abs_Ax),
        Az_plus=0.5 * (Az + abs_Az),
        Az_minus=0.5 * (Az - abs_Az),
    )


# ===========================================================================
# Boundary reflection masks
# ===========================================================================

def reflect_mask_x_spn(N: int) -> np.ndarray:
    """Ghost-cell sign mask for x-reflecting BC in SP_N.

    At an x-boundary the x-components of odd vector moments reverse sign
    (they are odd functions in x), while scalars and z-components are even.

    For SP_3:  [+1, -1, +1, +1, -1, +1]
    """
    masks = []
    for n in range(N + 1):
        if n % 2 == 0:
            masks.append(1.0)   # scalar: even in x
        else:
            masks.append(-1.0)  # x-component: odd in x (flips)
            masks.append(1.0)   # z-component: even in x (unchanged)
    return np.array(masks, dtype=float)


def reflect_mask_z_spn(N: int) -> np.ndarray:
    """Ghost-cell sign mask for z-reflecting BC in SP_N.

    At a z-boundary the z-components of odd vector moments reverse sign;
    scalars and x-components are unchanged.

    For SP_3:  [+1, +1, -1, +1, +1, -1]
    """
    masks = []
    for n in range(N + 1):
        if n % 2 == 0:
            masks.append(1.0)   # scalar
        else:
            masks.append(1.0)   # x-component: even in z (unchanged)
            masks.append(-1.0)  # z-component: odd in z (flips)
    return np.array(masks, dtype=float)


# ===========================================================================
# Filter profile for SP_N
# ===========================================================================

def build_spn_filter_moments(N: int, filter_type: str = 'lanczos',
                              exp_order: int = 4) -> np.ndarray:
    """Build Lanczos/spline/exponential filter profile for SP_N unknowns.

    The filter level for moment n is ζ = n/(N+1).  Both x- and z-components
    of odd moments share the same level as the scalar moment of the same order.
    The zeroth unknown (φ₀, the energy moment) is always left unfiltered.
    """
    zeta_vals: list[float] = []
    for n in range(N + 1):
        zeta = n / float(N + 1)
        if n % 2 == 0:
            zeta_vals.append(zeta)
        else:
            zeta_vals.append(zeta)   # x-component
            zeta_vals.append(zeta)   # z-component

    zeta = np.array(zeta_vals, dtype=np.float64)
    ftype = str(filter_type).strip().lower()

    if ftype == 'lanczos':
        rho = np.sinc(zeta)
    elif ftype in ('spline', 'sspline'):
        rho = 1.0 / (1.0 + zeta ** 4)
    elif ftype in ('exp', 'exponential'):
        alpha = int(exp_order)
        if alpha <= 0:
            raise ValueError("exp_order must be positive")
        c0 = math.log(np.finfo(np.float64).eps)
        rho = np.exp(c0 * zeta ** alpha)
    else:
        raise ValueError(f"Unknown filter_type '{filter_type}'.")

    rho = np.clip(rho, 1e-300, 1.0)
    f_mom = -np.log(rho)
    f_mom[0] = 0.0   # preserve φ₀ (energy moment)
    return f_mom


# ===========================================================================
# Main TRT time-step loop
# ===========================================================================

def temp_solve_spn_2d(
    Ix, Iy, dx_arr, dy_arr,
    q_ext,
    sigma_func,
    scat_func,
    N_spn,
    EOS, invEOS,
    phi_init, T_init,
    *,
    I_init=None,
    dt_start: float = 1e-4,
    t_end: float = 1.0,
    reflect_xlo: bool = False,
    reflect_xhi: bool = False,
    reflect_ylo: bool = False,
    reflect_yhi: bool = False,
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
    use_cached_lu: bool = True,
    deduplicate_lu: bool = True,
    boundary_moments=None,
    filter_type: str = 'lanczos',
    filter_strength=None,
    filter_exp_order: int = 4,
):
    """Gray TRT time-step loop using the 2-D Cartesian SP_N SCB solver.

    Interface is identical to ``temp_solve_pn_2d`` in pn_solver_2d, except
    that no ``jacobian_dir`` argument is required: the SP_N Jacobians are
    built analytically from N.

    Parameters
    ----------
    N_spn : int
        SP_N order (must be odd: 1, 3, 5, …).
    All other parameters are identical to temp_solve_pn_2d.

    Returns
    -------
    phi, T, I, t_now, history
        Same shapes and semantics as temp_solve_pn_2d.
    """
    dx_arr = np.asarray(dx_arr, dtype=np.float64)
    dy_arr = np.asarray(dy_arr, dtype=np.float64)
    if dt_max is None:
        dt_max = t_end

    # Build analytic SPN Jacobians (no CSV files)
    jac = build_spn_jacobians(N_spn)
    S = jac['S']

    xmask = reflect_mask_x_spn(N_spn)
    zmask = reflect_mask_z_spn(N_spn)
    stream_cache: dict = {}
    filter_moments = build_spn_filter_moments(
        N_spn, filter_type=filter_type, exp_order=filter_exp_order)

    # Initialise state
    phi = phi_init.astype(np.float64).copy()
    T   = T_init.astype(np.float64).copy()
    e   = EOS(T)

    if I_init is not None:
        I   = np.asarray(I_init, dtype=np.float64).copy()
        phi = _SQRT4PI * I[:, :, :, 0]
    else:
        I = np.zeros((Ix, Iy, 4, S), dtype=np.float64)
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

    print(f"SP{N_spn} 2-D SCB solver:  S={S}  mesh={Ix}×{Iy}")

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

        icdt = 1.0 / (c * dt)
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
            sigma_filter = _evaluate_filter_strength(filter_strength, T_star, Ix, Iy)

            f        = 1.0 / (1.0 + beta_val * c * dt * sigma_abs)
            sigma_a  = f * sigma_abs
            sigma_s  = (1.0 - f) * sigma_abs + scat
            sigma_hat = sigma_abs + scat + icdt
            emission  = sigma_a * ac * T_star ** 4

            delta_e_src      = -(1.0 - f) * (EOS(T_star) - e_old) / dt
            source_iso_fixed = emission + delta_e_src + q_now

            # SPN source: isotropic part goes into row 0 (φ₀),
            # same normalisation φ₀ = √(4π) I[…,0] as PN.
            src_fixed = icdt * I_old.copy()
            src_fixed[:, :, :, 0] += source_iso_fixed / _SQRT4PI

            tau_phi = (tolerance if (W == 0 or do_final)
                       else tolerance * (1e-3 / tolerance) **
                            (1.0 - k_outer / max(W, 1)))

            # Reuse the PN SCB step solver with our SPN jac and masks
            I, phi, n_its = solve_pn_step(
                sigma_hat, sigma_s, src_fixed, I_old,
                dx_arr, dy_arr, Ix, Iy, S,
                jac, stream_cache,
                reflect_xlo, reflect_xhi, reflect_ylo, reflect_yhi,
                xmask, zmask,
                use_gmres=False,
                n_gs=n_gs, tol=tau_phi, maxits=maxits, loud=loud,
                use_cached_lu=use_cached_lu,
                deduplicate_lu=deduplicate_lu,
                boundary_moments=boundary_moments,
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
                deriv_val = dt_max ** 2 / delta_step

        T_old2 = T_old.copy()

    print()
    print(f"Total source iterations: {total_its}")
    return phi, T, I, t_now, history
