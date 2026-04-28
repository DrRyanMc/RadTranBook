"""
Multigroup 1-D Discrete Ordinates (S_N) solver for time-dependent
radiative transfer.

Groups are coupled through the Fleck-factor linearised emission.
The time-discretised transport equation for group *g* is

  (1/(cΔt) + σ_{t,g}) ψ_g = S_g  +  μ-derivative terms

with the isotropic source

  S_g = σ_{s,g} φ_g  +  α_g Σ_{g'} σ_{a,g'} φ_{g'}     ← φ-dependent
        + [σ_{a,g} B_g(T^n) − α_g Σ_{g'} σ_{a,g'} B_{g'}(T^n)]
        + (1/(cΔt)) ψ_g^n  +  Q_g                        ← fixed

where

  f   = 1 / (1 + (Δt/C_v) Σ_g σ_{a,g} dB_g/dT)          Fleck factor
  α_g = σ_{a,g} (dB_g/dT) f Δt / C_v                    coupling weight

and B_g, dB_g/dT are the "4π" forms (i.e. 4πB_g, 4π dB_g/dT).
Note Σ_g α_g = 1 − f.

Uses all low-level sweep routines from ``sn_solver``.
"""

import math
import numpy as np
from numba import njit

import sn_solver
from sn_solver import (
    c, a, ac,
    single_source_iteration,
    single_source_iteration_psi,
    solver_with_dmd_inc,
    build_reflecting_BCs,
)


# ── helpers ────────────────────────────────────────────────────────────────
def _compute_fleck_and_alpha(T, dt, sigma_a_funcs, Bg_funcs, dBdT_funcs,
                              Cv_func, G):
    """Multigroup Fleck factor *f* and coupling weights *α_g*.

    Returns
    -------
    f      : (I, nop1)
    alpha_g : list of G  (I, nop1)
    """
    Cv = Cv_func(T)
    coupling_sum = np.zeros_like(T)
    dBdT_vals = []
    for g in range(G):
        dB = dBdT_funcs[g](T)
        dBdT_vals.append(dB)
        coupling_sum += sigma_a_funcs[g](T) * dB
    f = 1.0 / (1.0 + dt / Cv * coupling_sum)
    alpha_g = []
    for g in range(G):
        alpha_g.append(sigma_a_funcs[g](T) * dBdT_vals[g] * f * dt / Cv)
    return f, alpha_g


def mg_temp_solve_dmd_inc(
    # geometry
    I, hx,
    # groups
    G,
    sigma_a_funcs,      # list of G callables  T → (I, nop1)
    scat_funcs,         # list of G callables  T → (I, nop1)
    Bg_funcs,           # list of G callables  T → (I, nop1)  (= 4π B_g)
    dBdT_funcs,         # list of G callables  T → (I, nop1)  (= 4π dB_g/dT)
    q_ext,              # list of G  (I, N, nop1)  fixed sources
    # angular
    N,
    BCs,                # callable(t) → (N, nop1)  (vacuum / incident)
    # material
    EOS, invEOS,        # e(T), T(e)  node-wise
    Cv_func,            # callable  T → (I, nop1)
    # initial state
    phi_g, psi_g, T,
    # time stepping
    dt_min=1e-5, dt_max=0.001, tfinal=1.0,
    # convergence
    Linf_tol=1e-10, tolerance=1e-12, maxits=100,
    LOUD=False, order=4, fix=0, K=100, R=3,
    time_outputs=None,
    chi=None,
    reflect_left=False,
    reflect_right=False,
    energy_diagnostics=False,
    diagnostics_stride=10,
    diagnostics_store=None,
    fleck_mode="legacy",
):
    r"""Time-dependent multigroup radiative transfer with DMD-accelerated S_N.

    Parameters
    ----------
    I : int
        Spatial zones.
    hx : float
        Zone width.
    G : int
        Number of energy groups.
    sigma_a_funcs : list of G callables  ``T → (I, nop1)``
        Absorption opacity per group.
    scat_funcs : list of G callables  ``T → (I, nop1)``
        Scattering cross-section per group (often zero).
    Bg_funcs : list of G callables  ``T → (I, nop1)``
        Normalised Planck emission per group: returns :math:`4\pi B_g(T)`.
    dBdT_funcs : list of G callables  ``T → (I, nop1)``
        Temperature derivative: returns :math:`4\pi\, dB_g/dT`.
    q_ext : list of G ndarrays  ``(I, N, nop1)``
        Fixed external source per group.
    N : int
        Number of discrete ordinates.
    BCs : callable ``t → (N, nop1)`` or list of G callables
        Boundary condition function.  A single callable is used for all
        groups; a list of G callables provides per-group boundary
        conditions (needed for multigroup Marshak waves).
    EOS, invEOS : callables
        ``e(T)`` and ``T(e)`` acting node-wise on ``(I, nop1)``.
    Cv_func : callable ``T → (I, nop1)``
        Heat capacity at each node.
    phi_g : list of G ndarrays ``(I, nop1)``
        Initial group scalar fluxes.
    psi_g : list of G ndarrays ``(I, N, nop1)``
        Initial group angular fluxes.
    T : ndarray ``(I, nop1)``
        Initial temperature.
    dt_min, dt_max : float
        Adaptive time-step bounds.
    tfinal : float
        Final time.
    Linf_tol, tolerance : float
        Convergence tolerances.
    maxits : int
        Max DMD iterations per step.
    order : int
        Bernstein polynomial order.
    fix : int
        Positivity fix-up flag.
    K : int
        DMD inner iterations.
    R : int
        Richardson iterations between DMD blocks.
    time_outputs : ndarray or None
        Specific output times.
    chi : ndarray (G,) or None
        Prescribed emission fractions.  When provided, the coupling
        weights are set to ``chi[g] * (1 - f)`` instead of the
        Fleck-linearisation–derived ``α_g = σ_{a,g} dB_g/dT f Δt/C_v``.
        This matters for problems like the Su-Olson picket fence where
        the emission spectrum is defined independently of the opacities.
    reflect_left, reflect_right : bool
        Enable reflecting boundaries by mirroring previous-step outgoing
        angular flux into incoming boundary states.
    energy_diagnostics : bool
        If True, print per-step open-boundary energy diagnostics.
    diagnostics_stride : int
        Print diagnostics every ``diagnostics_stride`` steps.
    diagnostics_store : list or None
        Optional list that receives one per-step diagnostics dict when
        ``energy_diagnostics`` is enabled.
    fleck_mode : {"legacy", "imc"}
        Fleck linearisation mode.
        ``"legacy"`` uses the existing multigroup dB/dT coupling.
        ``"imc"`` uses the multigroup IMC form with
        :math:`f = 1/(1 + \beta c \sigma_P \Delta t)` and
        effective-scatter redistribution proportional to
        :math:`\sigma_{a,g} b_g^\star`.

    Returns
    -------
    phi_g_hist : list of lists  (len = # saved steps, inner = G arrays)
    T_hist     : list of (I, nop1)
    iterations : int   total sweeps
    ts         : ndarray of saved times
    """
    nop1 = order + 1
    _bc_per_group = isinstance(BCs, (list, tuple))
    t_current = 0.0
    phi_g_hist = []
    T_hist = []

    # deep-copy initial state
    phi_g = [p.copy() for p in phi_g]
    psi_g = [p.copy() for p in psi_g]
    psi_g_old = [p.copy() for p in psi_g]

    T_old = T.copy()
    T_old2 = T.copy()
    e = EOS(T)
    e_old = e.copy()

    phi_g_hist.append([p.copy() for p in phi_g])
    T_hist.append(T_old.copy())

    ts = [t_current]
    iterations = 0
    step_num = 0
    dt_old = dt_min
    dt = dt_min
    deriv_val = 0.0
    delta_step = 1e-3
    curr_step = 0
    t_output_index = 0

    # Energy diagnostics state
    cumulative_residual = 0.0
    E_prev = None
    if energy_diagnostics:
        E_mat0 = hx * np.sum(np.mean(e_old, axis=1))
        E_rad0 = 0.0
        for g in range(G):
            E_rad0 += hx * np.sum(np.mean(phi_g[g], axis=1) / c)
        E_prev = E_mat0 + E_rad0
        print(
            f"\n[diag:init] E0={E_prev:.8e} "
            f"(E_mat={E_mat0:.8e}, E_rad={E_rad0:.8e})"
        )

    print(f"MG-S_N: G={G}, I={I}, N={N}, order={order}")
    print("|", end="")

    while t_current < tfinal:
        dt_old2 = dt_old
        dt_old = dt
        step_num += 1

        # adaptive dt
        if step_num > 2:
            dt_prop = np.sqrt(delta_step * deriv_val)
            if dt_prop > dt_max:
                dt_prop = dt_max
            if dt_prop < dt_min:
                dt_prop = dt_min
            if dt_prop > 2 * dt:
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
                dt = time_outputs[t_output_index] - t_current
                t_output_index += 1
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

        # per-group opacities and Planck evaluations at T_old
        sigma_ag = [sigma_a_funcs[g](T_old) for g in range(G)]
        scat_g   = [scat_funcs[g](T_old)    for g in range(G)]
        Bg_vals  = [Bg_funcs[g](T_old)      for g in range(G)]

        # Fleck factor and coupling weights α_g
        if fleck_mode == "legacy":
            f, alpha_g = _compute_fleck_and_alpha(
                T_old, dt, sigma_a_funcs, Bg_funcs, dBdT_funcs, Cv_func, G)

            # override with prescribed emission fractions if given
            if chi is not None:
                one_minus_f = 1.0 - f
                alpha_g = [chi[g] * one_minus_f for g in range(G)]
        elif fleck_mode == "imc":
            Cv = Cv_func(T_old)
            beta = 4.0 * a * T_old**3 / Cv

            if chi is None:
                sum_B = np.zeros_like(T_old)
                for g in range(G):
                    sum_B += np.maximum(Bg_vals[g], 0.0)
                b_star = [np.maximum(Bg_vals[g], 0.0) / (sum_B + 1e-300)
                          for g in range(G)]
            else:
                b_star = [chi[g] + np.zeros_like(T_old) for g in range(G)]

            sigma_P = np.zeros_like(T_old)
            for g in range(G):
                sigma_P += sigma_ag[g] * b_star[g]

            f = 1.0 / (1.0 + beta * c * sigma_P * dt)
            f = np.clip(f, 0.0, 1.0)
            one_minus_f = 1.0 - f
            alpha_g = []
            for g in range(G):
                alpha_g.append(one_minus_f * sigma_ag[g] * b_star[g] / (sigma_P + 1e-300))
        else:
            raise ValueError(f"Unknown fleck_mode: {fleck_mode}")

        # precompute Σ_{g'} σ_{a,g'} B_{g'}(T^n)
        sum_sigma_B = np.zeros_like(T_old)
        for g in range(G):
            sum_sigma_B += sigma_ag[g] * Bg_vals[g]

        # --- build fixed source and sigma_t per group ---
        sigma_t_g = []
        fixed_source_g = []
        for g in range(G):
            sigma_t = sigma_ag[g] + scat_g[g] + icdt

            # fixed emission: σ_{a,g} B_g − α_g Σ_{g'} σ_{a,g'} B_{g'}
            emission_fixed = sigma_ag[g] * Bg_vals[g] - alpha_g[g] * sum_sigma_B

            # full fixed source (angle-resolved because of time-deriv term)
            source = (q_ext[g]
                      + emission_fixed[:, None, :]
                      + icdt * psi_g_old[g])

            sigma_t_g.append(sigma_t)
            fixed_source_g.append(source)

        # --- coupled matvec for DMD ---
        block = I * nop1
        total_unknowns = G * block
        zero_BCs = np.zeros((N, nop1))

        # zero source template for broadcasting (I, 1, nop1) → (I, N, nop1)
        zero_src_template = np.zeros((I, N, nop1))

        def _make_mv(scat_g, alpha_g, sigma_ag, sigma_t_g):
            """Factory to capture current-step variables."""
            _scat    = [s.copy() for s in scat_g]
            _alpha   = [a.copy() for a in alpha_g]
            _sigma_a = [s.copy() for s in sigma_ag]
            _sigma_t = [s.copy() for s in sigma_t_g]
            _G = len(scat_g)
            _zero = zero_src_template

            def mv(x_vec):
                result = np.zeros_like(x_vec)
                phi_parts = [x_vec[g*block:(g+1)*block].reshape((I, nop1))
                             for g in range(_G)]
                # precompute Σ_{g'} σ_{a,g'} φ_{g'} (used by all groups)
                sum_sigma_phi = np.zeros((I, nop1))
                for gp in range(_G):
                    sum_sigma_phi += _sigma_a[gp] * phi_parts[gp]
                for g in range(_G):
                    # within-group scattering + cross-group coupling
                    #   σ_{s,g} φ_g + α_g Σ_{g'} σ_{a,g'} φ_{g'}
                    src_g = ((_scat[g] * phi_parts[g]
                              + _alpha[g] * sum_sigma_phi)[:, None, :]
                             + _zero)
                    phi_new = single_source_iteration(
                        I, hx, src_g, _sigma_t[g], N, zero_BCs,
                        order=order, fix=fix)
                    result[g*block:(g+1)*block] = phi_new.ravel()
                return result
            return mv

        mv = _make_mv(scat_g, alpha_g, sigma_ag, sigma_t_g)
        # Solve transport. For reflecting boundaries, iterate BC coupling so
        # incoming boundary states are consistent with same-step outgoing flux.
        psi_bc_state = [p.copy() for p in psi_g_old]
        max_reflect_its = 20 if (reflect_left or reflect_right) else 1
        reflect_tol = 1e-12
        x_sol = np.concatenate([p.ravel() for p in phi_g])

        for _ in range(max_reflect_its):
            # b = one sweep per group with fixed source only (+ BCs)
            b_vec = np.zeros(total_unknowns)
            if not _bc_per_group:
                _bc_b = BCs(t_current - dt / 2)
            for g in range(G):
                bc_g = BCs[g](t_current - dt / 2) if _bc_per_group else _bc_b
                if reflect_left or reflect_right:
                    bc_g = build_reflecting_BCs(
                        bc_g.copy(), psi_bc_state[g], reflect_left,
                        reflect_right, N, order)
                phi_new = single_source_iteration(
                    I, hx, fixed_source_g[g], sigma_t_g[g], N, bc_g,
                    order=order, fix=fix)
                b_vec[g*block:(g+1)*block] = phi_new.ravel()

            x_sol, total_its, change, change_linf, Atil, Yp, Ym = \
                solver_with_dmd_inc(
                    matvec=mv, b=b_vec, K=K, max_its=maxits, steady=1,
                    x=x_sol, Rits=R, LOUD=LOUD, order=order,
                    L2_tol=tolerance, Linf_tol=Linf_tol)
            iterations += total_its

            # unpack group fluxes
            for g in range(G):
                phi_g[g] = x_sol[g*block:(g+1)*block].reshape((I, nop1))

            # reconstruct angular fluxes ψ_g (for next step's time-derivative)
            if not _bc_per_group:
                _bc_psi = BCs(t_current)
            sum_sigma_phi = np.zeros_like(T_old)
            for gp in range(G):
                sum_sigma_phi += sigma_ag[gp] * phi_g[gp]

            psi_candidate = [None] * G
            for g in range(G):
                bc_g = BCs[g](t_current) if _bc_per_group else _bc_psi
                if reflect_left or reflect_right:
                    bc_g = build_reflecting_BCs(
                        bc_g.copy(), psi_bc_state[g], reflect_left,
                        reflect_right, N, order)
                full_src = ((scat_g[g] * phi_g[g]
                            + alpha_g[g] * sum_sigma_phi)[:, None, :]
                            + fixed_source_g[g])
                psi_candidate[g] = single_source_iteration_psi(
                    I, hx, full_src, sigma_t_g[g], N, bc_g,
                    order=order, fix=fix)

            if not (reflect_left or reflect_right):
                psi_g = psi_candidate
                break

            # Converge the boundary fixed-point map for reflecting walls.
            bc_change = 0.0
            for g in range(G):
                bc_old = build_reflecting_BCs(
                    np.zeros((N, nop1)), psi_bc_state[g],
                    reflect_left, reflect_right, N, order)
                bc_new = build_reflecting_BCs(
                    np.zeros((N, nop1)), psi_candidate[g],
                    reflect_left, reflect_right, N, order)
                bc_change = max(bc_change, np.max(np.abs(bc_new - bc_old)))

            psi_bc_state = [p.copy() for p in psi_candidate]
            psi_g = psi_candidate
            if bc_change < reflect_tol:
                break

        # material energy update:
        #   e^{n+1} = e^n + f Δt Σ_g σ_{a,g} (φ_g − B_g(T^n))
        energy_dep = np.zeros_like(T_old)
        for g in range(G):
            energy_dep += sigma_ag[g] * (phi_g[g] - Bg_vals[g])
        e = e_old + f * dt * energy_dep
        T = invEOS(e)

        # second-derivative adaptive dt control
        if step_num >= 2:
            denom = np.mean(np.abs(
                T / (dt**2) - (dt + dt_old) / (dt**2 * dt_old) * T_old
                + T_old2 / (dt_old * dt)))
            if denom > 0:
                deriv_val = np.mean(T) / denom
            else:
                deriv_val = dt_max**2 / delta_step

        if energy_diagnostics:
            MU, W = np.polynomial.legendre.leggauss(N)
            W = W / np.sum(W)

            def _incoming_power_at(t_eval):
                p_left = 0.0
                p_right = 0.0
                if _bc_per_group:
                    bc_list = [BCs[g](t_eval) for g in range(G)]
                else:
                    bc_all = BCs(t_eval)
                    bc_list = [bc_all for _ in range(G)]

                for g in range(G):
                    bc = bc_list[g]
                    for n in range(N):
                        mu = MU[n]
                        if mu > 0.0:
                            p_left += W[n] * mu * bc[n, nop1 - 1]
                        elif mu < 0.0:
                            p_right += W[n] * (-mu) * bc[n, 0]
                return p_left, p_right

            def _outgoing_power_from_psi(psi_state):
                p_left = 0.0
                p_right = 0.0
                for g in range(G):
                    for n in range(N):
                        mu = MU[n]
                        if mu > 0.0:
                            p_right += W[n] * mu * psi_state[g][I - 1, n, nop1 - 1]
                        elif mu < 0.0:
                            p_left += W[n] * (-mu) * psi_state[g][0, n, 0]
                return p_left, p_right

            t_start = t_current - dt
            t_end = t_current

            p_in_left_start, p_in_right_start = _incoming_power_at(t_start)
            p_in_left_end, p_in_right_end = _incoming_power_at(t_end)
            p_out_left_start, p_out_right_start = _outgoing_power_from_psi(psi_g_old)
            p_out_left_end, p_out_right_end = _outgoing_power_from_psi(psi_g)

            E_in_left_step = 0.5 * dt * (p_in_left_start + p_in_left_end)
            E_in_right_step = 0.5 * dt * (p_in_right_start + p_in_right_end)
            E_out_left_step = 0.5 * dt * (p_out_left_start + p_out_left_end)
            E_out_right_step = 0.5 * dt * (p_out_right_start + p_out_right_end)
            E_in_step = E_in_left_step + E_in_right_step
            E_out_step = E_out_left_step + E_out_right_step

            E_mat_now = hx * np.sum(np.mean(e, axis=1))
            E_rad_now = 0.0
            for g in range(G):
                E_rad_now += hx * np.sum(np.mean(phi_g[g], axis=1) / c)
            E_now = E_mat_now + E_rad_now

            dE = E_now - E_prev
            dE_expected = E_in_step - E_out_step
            residual = dE - dE_expected
            cumulative_residual += residual
            E_prev = E_now

            if (diagnostics_stride > 0) and (step_num % diagnostics_stride == 0):
                print(
                    f"\n[diag] step={step_num} t={t_current:.6e} "
                    f"net_boundary={dE_expected:.6e} "
                    f"cum_residual={cumulative_residual:.6e}"
                )

            if diagnostics_store is not None:
                diagnostics_store.append({
                    'step': step_num,
                    'time': float(t_current),
                    'dt': float(dt),
                    'boundary_in': float(E_in_step),
                    'boundary_in_left': float(E_in_left_step),
                    'boundary_in_right': float(E_in_right_step),
                    'boundary_out': float(E_out_step),
                    'boundary_out_left': float(E_out_left_step),
                    'boundary_out_right': float(E_out_right_step),
                    'net_boundary': float(dE_expected),
                    'dE': float(dE),
                    'residual': float(residual),
                    'cumulative_residual': float(cumulative_residual),
                    'E_mat': float(E_mat_now),
                    'E_rad': float(E_rad_now),
                    'E_total': float(E_now),
                })

        # shift
        e_old = e.copy()
        T_old2 = T_old.copy()
        T_old = T.copy()
        psi_g_old = [p.copy() for p in psi_g]

        phi_g_hist.append([p.copy() for p in phi_g])
        T_hist.append(T.copy())

    print()
    return phi_g_hist, T_hist, iterations, np.array(ts)
