"""
1-D Discrete Ordinates (S_N) solver using Linear Discontinuous (LD) Galerkin
elements with mass lumping.

Spatial discretisation follows Chapter 11 of the textbook:

  - Two unknowns per cell: left-edge (index 0) and right-edge (index 1).
  - 2×2 cell solve (eqs 11.65/11.66) with upwind flux selection.
  - Conservative negative-intensity fixup (Algorithm 5, Section 11.3.2).
  - DMD-accelerated source iteration (Algorithm 6, Section 11.4).

Array shapes
------------
- phi, T, e, sigma  : (I, 2)  — 0 = left edge, 1 = right edge per cell
- psi, source       : (I, N, 2)
- BCs               : (N, 2)  — [n, 0] inflow at right boundary (mu < 0),
                                [n, 1] inflow at left  boundary (mu > 0)

Physical constants are in CGS units with time in nanoseconds.

DMD acceleration utilities are imported from sn_solver.
"""

import math
import numpy as np
from numba import jit, prange

from sn_solver import (
    c, a, ac,
    _get_quadrature,
    solver_with_dmd_inc,
)


# ---------------------------------------------------------------------------
# JIT-compiled parallel sweep kernel — scalar flux only
# ---------------------------------------------------------------------------

@jit(nopython=True, parallel=True, cache=True)
def _sweep_all_phi_ld(I, hx, source, sigma_hat, MU, W, BCs, fix):
    """Sweep all N angles in parallel; return scalar flux phi (I, 2).

    Implements the LD 2×2 cell solve (eqs 11.65/11.66) with upwinding and
    the conservative negative-intensity fixup of Algorithm 5.

    Parameters
    ----------
    I : int
        Number of spatial cells.
    hx : float
        Cell width (uniform mesh).
    source : (I, N, 2)  float64
        Source at each cell edge and angle.  ``source[j, n, 0]`` is the
        left-edge source for cell ``j`` and ordinate ``n``; index 1 is
        the right edge.
    sigma_hat : (I, 2)  float64
        Effective total opacity  σ̂ = σ_t + 1/(c Δt) at left (0) and
        right (1) edges.
    MU : (N,)  float64
        Gauss-Legendre ordinates on [-1, 1].
    W : (N,)  float64
        Quadrature weights normalised to sum to 1.
    BCs : (N, 2)  float64
        Incoming boundary intensities.  ``BCs[n, 1]`` is the left-boundary
        inflow (used when MU[n] > 0); ``BCs[n, 0]`` is the right-boundary
        inflow (used when MU[n] < 0).
    fix : int
        If > 0 apply the conservative negative-intensity fixup.

    Returns
    -------
    phi : (I, 2)  float64
        Scalar flux at left (0) and right (1) cell edges.
    """
    N = MU.size
    psi_l_all = np.zeros((N, I))
    psi_r_all = np.zeros((N, I))

    for n in prange(N):
        mu = MU[n]
        psi_l = np.zeros(I)
        psi_r = np.zeros(I)

        if mu > 0.0:
            # ── Left-to-right sweep (eq 11.65) ──────────────────────────────
            I_prev = BCs[n, 1]        # inflow from left boundary
            for j in range(I):
                # 2×2 matrix entries
                a00 = mu * 0.5 + hx * 0.5 * sigma_hat[j, 0]
                a01 = mu * 0.5
                a10 = -mu * 0.5
                a11 = mu * 0.5 + hx * 0.5 * sigma_hat[j, 1]
                rhs0 = mu * I_prev + hx * 0.5 * source[j, n, 0]
                rhs1 = hx * 0.5 * source[j, n, 1]
                det = a00 * a11 - a01 * a10
                I_l_raw = (rhs0 * a11 - a01 * rhs1) / det
                I_r_raw = (a00 * rhs1 - rhs0 * a10) / det
                # Conservative negative-intensity fixup (Algorithm 5)
                if fix > 0:
                    I_bar = 0.5 * (I_l_raw + I_r_raw)
                    I_min = min(I_l_raw, I_r_raw)
                    if I_min >= 0.0:
                        psi_l[j] = I_l_raw
                        psi_r[j] = I_r_raw
                    elif I_bar >= 0.0:
                        theta = I_bar / (I_bar - I_min)
                        psi_l[j] = I_bar + theta * (I_l_raw - I_bar)
                        psi_r[j] = I_bar + theta * (I_r_raw - I_bar)
                    else:
                        psi_l[j] = 0.0
                        psi_r[j] = 0.0
                else:
                    psi_l[j] = I_l_raw
                    psi_r[j] = I_r_raw
                I_prev = psi_r[j]   # fixed-up right face → next cell inflow
        else:
            # ── Right-to-left sweep (eq 11.66) ──────────────────────────────
            abs_mu = -mu
            I_next = BCs[n, 0]        # inflow from right boundary
            for j in range(I - 1, -1, -1):
                a00 = abs_mu * 0.5 + hx * 0.5 * sigma_hat[j, 0]
                a01 = -abs_mu * 0.5
                a10 = abs_mu * 0.5
                a11 = abs_mu * 0.5 + hx * 0.5 * sigma_hat[j, 1]
                rhs0 = hx * 0.5 * source[j, n, 0]
                rhs1 = abs_mu * I_next + hx * 0.5 * source[j, n, 1]
                det = a00 * a11 - a01 * a10
                I_l_raw = (rhs0 * a11 - a01 * rhs1) / det
                I_r_raw = (a00 * rhs1 - rhs0 * a10) / det
                if fix > 0:
                    I_bar = 0.5 * (I_l_raw + I_r_raw)
                    I_min = min(I_l_raw, I_r_raw)
                    if I_min >= 0.0:
                        psi_l[j] = I_l_raw
                        psi_r[j] = I_r_raw
                    elif I_bar >= 0.0:
                        theta = I_bar / (I_bar - I_min)
                        psi_l[j] = I_bar + theta * (I_l_raw - I_bar)
                        psi_r[j] = I_bar + theta * (I_r_raw - I_bar)
                    else:
                        psi_l[j] = 0.0
                        psi_r[j] = 0.0
                else:
                    psi_l[j] = I_l_raw
                    psi_r[j] = I_r_raw
                I_next = psi_l[j]   # fixed-up left face → next cell inflow

        psi_l_all[n] = psi_l
        psi_r_all[n] = psi_r

    # Accumulate scalar flux: phi = sum_n w_n psi_n
    phi = np.zeros((I, 2))
    for n in range(N):
        for j in range(I):
            phi[j, 0] += W[n] * psi_l_all[n, j]
            phi[j, 1] += W[n] * psi_r_all[n, j]
    return phi


# ---------------------------------------------------------------------------
# JIT-compiled parallel sweep kernel — full angular flux
# ---------------------------------------------------------------------------

@jit(nopython=True, parallel=True, cache=True)
def _sweep_all_psi_ld(I, hx, source, sigma_hat, MU, W, BCs, fix):
    """Sweep all N angles in parallel; return angular flux psi (I, N, 2).

    Same parameters as ``_sweep_all_phi_ld``.

    Returns
    -------
    psi : (I, N, 2)  float64
    """
    N = MU.size
    psi = np.zeros((I, N, 2))

    for n in prange(N):
        mu = MU[n]
        psi_l = np.zeros(I)
        psi_r = np.zeros(I)

        if mu > 0.0:
            I_prev = BCs[n, 1]
            for j in range(I):
                a00 = mu * 0.5 + hx * 0.5 * sigma_hat[j, 0]
                a01 = mu * 0.5
                a10 = -mu * 0.5
                a11 = mu * 0.5 + hx * 0.5 * sigma_hat[j, 1]
                rhs0 = mu * I_prev + hx * 0.5 * source[j, n, 0]
                rhs1 = hx * 0.5 * source[j, n, 1]
                det = a00 * a11 - a01 * a10
                I_l_raw = (rhs0 * a11 - a01 * rhs1) / det
                I_r_raw = (a00 * rhs1 - rhs0 * a10) / det
                if fix > 0:
                    I_bar = 0.5 * (I_l_raw + I_r_raw)
                    I_min = min(I_l_raw, I_r_raw)
                    if I_min >= 0.0:
                        psi_l[j] = I_l_raw
                        psi_r[j] = I_r_raw
                    elif I_bar >= 0.0:
                        theta = I_bar / (I_bar - I_min)
                        psi_l[j] = I_bar + theta * (I_l_raw - I_bar)
                        psi_r[j] = I_bar + theta * (I_r_raw - I_bar)
                    else:
                        psi_l[j] = 0.0
                        psi_r[j] = 0.0
                else:
                    psi_l[j] = I_l_raw
                    psi_r[j] = I_r_raw
                I_prev = psi_r[j]
        else:
            abs_mu = -mu
            I_next = BCs[n, 0]
            for j in range(I - 1, -1, -1):
                a00 = abs_mu * 0.5 + hx * 0.5 * sigma_hat[j, 0]
                a01 = -abs_mu * 0.5
                a10 = abs_mu * 0.5
                a11 = abs_mu * 0.5 + hx * 0.5 * sigma_hat[j, 1]
                rhs0 = hx * 0.5 * source[j, n, 0]
                rhs1 = abs_mu * I_next + hx * 0.5 * source[j, n, 1]
                det = a00 * a11 - a01 * a10
                I_l_raw = (rhs0 * a11 - a01 * rhs1) / det
                I_r_raw = (a00 * rhs1 - rhs0 * a10) / det
                if fix > 0:
                    I_bar = 0.5 * (I_l_raw + I_r_raw)
                    I_min = min(I_l_raw, I_r_raw)
                    if I_min >= 0.0:
                        psi_l[j] = I_l_raw
                        psi_r[j] = I_r_raw
                    elif I_bar >= 0.0:
                        theta = I_bar / (I_bar - I_min)
                        psi_l[j] = I_bar + theta * (I_l_raw - I_bar)
                        psi_r[j] = I_bar + theta * (I_r_raw - I_bar)
                    else:
                        psi_l[j] = 0.0
                        psi_r[j] = 0.0
                else:
                    psi_l[j] = I_l_raw
                    psi_r[j] = I_r_raw
                I_next = psi_l[j]

        for j in range(I):
            psi[j, n, 0] = psi_l[j]
            psi[j, n, 1] = psi_r[j]

    return psi


# ---------------------------------------------------------------------------
# Public sweep wrappers
# ---------------------------------------------------------------------------

def single_sweep_phi_ld(I, hx, source, sigma_hat, N, BCs, fix=1):
    """One parallel LD sweep over all angles; return scalar flux phi (I, 2).

    Parameters
    ----------
    I : int
        Number of spatial cells.
    hx : float
        Cell width.
    source : (I, N, 2)
        Source at each cell edge and ordinate.  ``source[j, n, 0]`` is the
        left-edge value; index 1 is the right edge.
    sigma_hat : (I, 2)
        Effective total opacity  σ̂ = σ_t + 1/(c Δt)  at left (0) and
        right (1) edges.  For a steady solve set the 1/(c Δt) part to zero.
    N : int
        Number of discrete ordinates.
    BCs : (N, 2)
        Incoming boundary intensities.  ``BCs[n, 1]`` is the inflow from
        the left wall for μ_n > 0; ``BCs[n, 0]`` is the inflow from the
        right wall for μ_n < 0.  Reflected or vacuum values should be set
        by the caller (see ``build_reflecting_BCs_ld``).
    fix : int
        Positivity fix-up flag.  1 = on (default), 0 = off.

    Returns
    -------
    phi : (I, 2)  float64
    """
    MU, W = _get_quadrature(N)
    return _sweep_all_phi_ld(
        I, hx,
        np.ascontiguousarray(source, dtype=np.float64),
        np.ascontiguousarray(sigma_hat, dtype=np.float64),
        MU, W,
        np.ascontiguousarray(BCs, dtype=np.float64),
        fix)


def single_sweep_psi_ld(I, hx, source, sigma_hat, N, BCs, fix=1):
    """One parallel LD sweep over all angles; return angular flux psi (I, N, 2).

    Same parameters as ``single_sweep_phi_ld``.

    Returns
    -------
    psi : (I, N, 2)  float64
    """
    MU, W = _get_quadrature(N)
    return _sweep_all_psi_ld(
        I, hx,
        np.ascontiguousarray(source, dtype=np.float64),
        np.ascontiguousarray(sigma_hat, dtype=np.float64),
        MU, W,
        np.ascontiguousarray(BCs, dtype=np.float64),
        fix)


# ---------------------------------------------------------------------------
# Reflecting / albedo boundary conditions
# ---------------------------------------------------------------------------

def build_reflecting_BCs_ld(bcs, psi_old, reflect_left, reflect_right, N):
    """Fill incoming boundary intensities from previous-step outgoing angular flux.

    For the symmetric Gauss-Legendre quadrature used here, the ordinate paired
    with MU[n] is MU[N-1-n].

    Parameters
    ----------
    bcs : (N, 2)
        Boundary array to update in-place.  Entries that are not overwritten
        retain whatever values the caller set (typically vacuum/incident).
    psi_old : (I, N, 2)
        Previous-step angular flux.
    reflect_left : bool
        Reflecting wall at z = 0.
    reflect_right : bool
        Reflecting wall at z = Z.
    N : int
        Number of ordinates.

    Returns
    -------
    bcs : (N, 2)  updated in-place
    """
    MU, _ = _get_quadrature(N)
    for n in range(N):
        n_ref = N - 1 - n
        if reflect_left and MU[n] > 0.0:
            # Incoming right-moving ray (μ > 0) at z = 0:
            # its inflow value equals the left face of cell 0 for the
            # mirrored left-moving ray (μ < 0, index n_ref).
            bcs[n, 1] = psi_old[0, n_ref, 0]
        elif reflect_right and MU[n] < 0.0:
            # Incoming left-moving ray (μ < 0) at z = Z:
            # its inflow value equals the right face of the last cell for
            # the mirrored right-moving ray (μ > 0, index n_ref).
            bcs[n, 0] = psi_old[-1, n_ref, 1]
    return bcs


# ---------------------------------------------------------------------------
# Gray time-dependent TRT solver
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


def temp_solve_ld(
    I, hx, q, sigma_func, scat_func, N, BCs, EOS, invEOS,
    phi, psi, T,
    dt_min=1e-5, dt_max=0.001, tfinal=1.0,
    Linf_tol=1.0e-5, tolerance=1.0e-8, maxits=100,
    LOUD=False, fix=1, K=100, R=3,
    time_outputs=None,
    reflect_left=False, reflect_right=False,
    print_stride=0,
    use_dmd=True,
    W=0,
    tau_phi_max=0.1,
    tau_T=1e-6,
    C_T=1.0,
    omega_T=1.0,
    T_floor=1e-10,
):
    r"""Time-dependent gray TRT with LD-S_N and DMD-accelerated source iteration.

    Implements the implicit linearised Fleck-Cummings scheme (Section 11.1.4)
    with the LD-DG spatial discretisation (Section 11.2) and Algorithm 6
    (DMD-accelerated source iteration).

    The time-discretised transport equation for each ordinate n is

        (1/(cΔt) + σ̂) I_n = σ_s φ + σ_a acT⁴_* + I^m_n / (cΔt) + q_n

    where σ̂ = σ + scat + 1/(cΔt),  σ_a = f σ,  σ_s = (1−f)σ + scat,  and
    f = 1 / (1 + β c σ Δt) is the Fleck factor.

    Parameters
    ----------
    I : int
        Number of spatial cells.
    hx : float
        Cell width (uniform mesh).
    q : (I, N, 2)
        Fixed external source (usually zero).
    sigma_func : callable T → (I, 2)
        Absorption opacity as a function of temperature.
    scat_func : callable T → (I, 2)
        Scattering cross-section as a function of temperature.
    N : int
        Number of discrete ordinates.
    BCs : callable t → (N, 2)
        Boundary conditions.  ``BCs(t)[n, 1]`` is the left-boundary inflow
        for μ_n > 0; ``BCs(t)[n, 0]`` is the right-boundary inflow for
        μ_n < 0.  Reflected values are set internally when
        ``reflect_left`` / ``reflect_right`` are True.
    EOS : callable T → (I, 2)
        Internal energy from temperature (node-wise).
    invEOS : callable e → (I, 2)
        Temperature from internal energy (node-wise).
    phi : (I, 2)
        Initial scalar flux.
    psi : (I, N, 2)
        Initial angular flux.
    T : (I, 2)
        Initial temperature.
    dt_min, dt_max : float
        Adaptive time-step bounds.
    tfinal : float
        Final simulation time.
    Linf_tol, tolerance : float
        L∞ and L² convergence tolerances for the DMD iterative solver.
    maxits : int
        Maximum number of DMD outer iterations per time step.
    LOUD : bool
        Verbosity flag.
    fix : int
        Positivity fix-up flag (default 1 = on).
    K : int
        Number of DMD inner iterations.
    R : int
        Number of Richardson iterations between DMD acceleration steps.
    time_outputs : ndarray or None
        Extra output times to hit exactly.
    reflect_left : bool
        Reflecting wall at z = 0.
    reflect_right : bool
        Reflecting wall at z = Z.

    Returns
    -------
    phis : list of (I, 2)
        Scalar flux saved at each time step.
    Ts : list of (I, 2)
        Temperature saved at each time step.
    iterations : int
        Total number of transport sweeps performed.
    ts : ndarray
        Array of saved time values.
    its_per_step : list of int
        Transport sweeps performed at each time step (length = number of steps).
    """
    t_current = 0.0
    phis = [phi.copy()]
    Ts = [T.copy()]
    its_per_step = []

    psi_old = psi.copy()
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

    print(f"LD-S_N: I={I}, N={N}")
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
            dt = tfinal - t_current
        try:
            if (time_outputs is not None and
                    t_current + dt > time_outputs[t_output_index] and
                    t_output_index < time_outputs.size):
                snap_dt = time_outputs[t_output_index] - t_current
                t_output_index += 1
                # Only snap if the residual is large enough to matter;
                # a tiny snap (floating-point noise) would waste steps.
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
        zero_BCs = np.zeros((N, 2))
        iterations_step = 0   # sweeps for this time step only

        # Preserve start-of-step ψ for icdt*ψ^n streaming source
        psi_step_start = psi_old.copy()

        # ── T_star nonlinear outer loop ─────────────────────────────────────
        # W=0: single linearisation at T_old (standard Fleck-Cummings).
        # W>0: up to W outer T_star updates using the material energy equation,
        #       with an inexact inner solve whose tolerance adapts to η_T:
        #         τ_φ(k=0) = tau_phi_max  (loose first pass)
        #         τ_φ(k>0) = clip(C_T·η_T, tolerance, tau_phi_max)
        #       where η_T = ‖T_star^(k)−T_star^(k-1)‖ / (‖T_star^(k)‖+T_floor).
        #       Outer loop exits when η_T < tau_T; a final tight solve at the
        #       converged T_star is then performed before the energy update.
        #       T_star is updated via: T_star ← (1−ω_T)T_star + ω_T·T_cand.
        # Float64-safe cap: T_star**4 overflows above ~1e77; cap well below that.
        # T_star is still allowed far above T_bc (the cap is ~1e50 keV).
        _T_max = 1e50
        T_star = np.minimum(T_old.copy(), _T_max)
        T_star_prev = T_star.copy()
        psi_bc_state = psi_old.copy()
        max_reflect_its = 20 if (reflect_left or reflect_right) else 1
        reflect_tol = 1e-12
        x_sol = phi.ravel()
        sigma_a = None   # set on first linearisation
        total_its = 0    # for print_stride
        do_final = False  # True → this pass is the final tight solve
        k_outer = 0

        while True:
            # ── Inner tolerance for this T_star cycle ─────────────────────
            # W=0 or do_final: must be tight (this is the solve that counts).
            # k_outer==0 with W>0: T_star hasn't been updated yet, so start
            #   loose — no point converging tightly to a linearisation point
            #   that is about to change.
            # k_outer>0: tighten adaptively as η_T shrinks.
            if W == 0 or do_final:
                tau_phi = tolerance              # tight: this solve counts
            elif k_outer == 0:
                tau_phi = tau_phi_max           # loose: T_star = T_old, first guess
            else:
                eta_T_k = float(np.sqrt(np.mean(
                    np.clip((T_star - T_star_prev) / (np.abs(T_star_prev) + T_floor),
                            -1e30, 1e30)**2)))
                tau_phi = float(np.clip(C_T * eta_T_k, tolerance, tau_phi_max))

            # ── Linearise at T_star ────────────────────────────────────────
            # Use a relative step so the stencil never crosses T=0.
            h_cv = np.maximum(np.abs(T_star) * 1e-4, 1e-12)
            Cv = np.maximum(
                (EOS(T_star + h_cv) - EOS(np.maximum(T_star - h_cv, T_floor))) / (2.0 * h_cv),
                1e-30)
            beta_val = 4.0 * a * T_star**3 / Cv
            sigma = sigma_func(T_star)               # (I, 2)
            scat = scat_func(T_star)                 # (I, 2)
            f = 1.0 / (1.0 + beta_val * c * dt * sigma)
            sigma_a = f * sigma                      # effective absorption (I, 2)
            sigma_s = (1.0 - f) * sigma + scat      # effective scattering (I, 2)
            sigma_t = sigma + scat                   # physical total (I, 2)
            sigma_hat = sigma_t + icdt               # σ̂ for the LD cell solve (I, 2)
            emission = sigma_a * ac * T_star**4      # (I, 2)
            # eq. (11.30): -(1-f)*Δe/Δt source term.
            # Weights sum to 1, so no 1/2 factor needed.
            # For k_outer==0, T_star==T_old so delta_e_src==0 (no W=0 regression).
            delta_e_src = -(1.0 - f) * (EOS(T_star) - e_old) / dt   # (I, 2)
            source = q + emission[:, None, :] + icdt * psi_step_start + delta_e_src[:, None, :]  # (I, N, 2)

            def _make_mv(ss, sh, nI, nhx, nN, zbc):
                """Factory to capture loop-step variables by value."""
                _ss = ss
                _sh = sh
                def mv(phi_vec):
                    phi_2d = phi_vec.reshape((nI, 2))
                    # Isotropic scattering source: same for all angles
                    src_scatter = (_ss * phi_2d)[:, None, :] * np.ones((1, nN, 1))
                    return single_sweep_phi_ld(
                        nI, nhx,
                        np.ascontiguousarray(src_scatter),
                        _sh, nN, zbc, fix=1,
                    ).ravel()
                return mv

            mv = _make_mv(sigma_s, sigma_hat, I, hx, N, zero_BCs)

            # ── Reflecting BC convergence loop ─────────────────────────────
            for _ in range(max_reflect_its):
                bc_use = BCs(t_current - dt / 2.0).copy()
                if reflect_left or reflect_right:
                    bc_use = build_reflecting_BCs_ld(
                        bc_use, psi_bc_state, reflect_left, reflect_right, N)

                b_vec = single_sweep_phi_ld(
                    I, hx, source, sigma_hat, N, bc_use, fix=fix
                ).ravel()

                try:
                    x_sol_cand, total_its, _chg, _chgL, _At, _Yp, _Ym = solver_with_dmd_inc(
                        matvec=mv, b=b_vec, K=K, max_its=maxits, steady=1,
                        x=x_sol, Rits=R, LOUD=LOUD,
                        L2_tol=tau_phi, Linf_tol=Linf_tol) if use_dmd else \
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

                # Reconstruct angular flux ψ at end-of-step
                bc_psi = BCs(t_current).copy()
                if reflect_left or reflect_right:
                    bc_psi = build_reflecting_BCs_ld(
                        bc_psi, psi_bc_state, reflect_left, reflect_right, N)
                full_src = source + (sigma_s * phi)[:, None, :] * np.ones((1, N, 1))
                psi_candidate = single_sweep_psi_ld(
                    I, hx, full_src, sigma_hat, N, bc_psi, fix=fix)

                if not (reflect_left or reflect_right):
                    psi_old = psi_candidate
                    break

                # Check boundary fixed-point convergence
                bc_old = build_reflecting_BCs_ld(
                    np.zeros((N, 2)), psi_bc_state, reflect_left, reflect_right, N)
                bc_new = build_reflecting_BCs_ld(
                    np.zeros((N, 2)), psi_candidate, reflect_left, reflect_right, N)
                bc_change = np.max(np.abs(bc_new - bc_old))
                psi_bc_state = psi_candidate.copy()
                psi_old = psi_candidate
                if bc_change < reflect_tol:
                    break

            # ── Exit condition ─────────────────────────────────────────────
            if W == 0 or do_final:
                # Single linearisation (W=0) or final tight solve done.
                break

            # ── T_cand and damped T_star update ───────────────────────────
            # Textbook eq. (11.26):
            #   [e(T_{m+1}) - e(T_m)] / dt = f*sigma*(phi - ac*T_star^4)
            #                                + (1-f) * [e(T_star) - e(T_m)] / dt
            # For k_outer==0: T_star == T_old so Delta_e == 0 (same as W=0).
            delta_e_star = EOS(T_star) - e_old          # e(T_star) - e(T_m)
            e_cand = (e_old
                      + sigma_a * dt * (phi - ac * T_star**4)
                      + (1.0 - f) * delta_e_star)
            T_cand = invEOS(e_cand)
            T_star_prev = T_star.copy()
            T_star = np.clip(
                (1.0 - omega_T) * T_star + omega_T * T_cand,
                T_floor, _T_max)

            # ── Check T_star convergence ───────────────────────────────────
            Tstar_l2 = np.sqrt(np.mean(T_star**2))
            diff = T_star - T_star_prev
            eta_T_new = float(
                np.sqrt(np.mean(np.clip(diff / (np.abs(T_star_prev) + T_floor),
                                        -1e30, 1e30)**2)))
            k_outer += 1
            if eta_T_new < tau_T or k_outer >= W:
                # Either T_star converged or we hit the iteration cap.
                # In both cases, do one final tight solve at the current T_star
                # before the energy update so phi and T_star are consistent.
                do_final = True

        # ── Final material energy update ──────────────────────────────────
        # Same textbook formula (11.26); for W=0, T_star==T_old so Delta_e==0.
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

    print()
    return phis, Ts, iterations, np.array(ts), its_per_step
