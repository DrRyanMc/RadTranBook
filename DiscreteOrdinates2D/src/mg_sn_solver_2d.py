"""
2-D Cartesian multigroup Discrete Ordinates (S_N) TRT solver.

This extends the gray 2-D solver by coupling groups through the
Fleck-linearized emission term, following the 1-D multigroup solvers.

Unknown layout
--------------
- Per-group scalar flux: phi_g[g] has shape (Ix, Iy, 4)
- Per-group angular flux: psi_g[g] has shape (Ix, Iy, M, 4)
  with M = number of 2-D ordinates from get_2d_quadrature.

The linear solve at each time step uses a stacked unknown vector over all
groups and matrix-free source iteration with optional DMD acceleration.
"""

import math
import os
import sys
import numpy as np

# Make sibling imports robust regardless of invocation directory.
_this_file_dir = os.path.dirname(os.path.abspath(__file__))
if _this_file_dir not in sys.path:
    sys.path.insert(0, _this_file_dir)
sys.path.insert(0, os.path.join(_this_file_dir, '..', 'DiscreteOrdinates'))

from DiscreteOrdinates2D.src.quadratures import get_2d_quadrature
from DiscreteOrdinates.src.sn_solver import solver_with_dmd_inc as _solver_with_dmd_inc_1d
from .sn_solver_2d import (
    c,
    a,
    sweep_2d_iso,
    sweep_2d_transient,
    build_reflecting_BCs_2d,
    _build_reflection_maps,
)

from DiscreteOrdinates.src.compton import operator_split_compton_update
from DiscreteOrdinates.src.kompaneets import operator_split_kompaneets_update


def _compute_fleck_and_alpha_2d(T, dt, sigma_a_funcs, dBdT_funcs, Cv_func, G):
    """Return Fleck factor and multigroup coupling weights on (Ix, Iy, 4)."""
    Cv = Cv_func(T)
    coupling_sum = np.zeros_like(T)
    dBdT_vals = []
    for g in range(G):
        dB = dBdT_funcs[g](T)
        dBdT_vals.append(dB)
        coupling_sum += sigma_a_funcs[g](T) * dB
    f = 1.0 / (1.0 + dt / Cv * coupling_sum)
    alpha_g = [sigma_a_funcs[g](T) * dBdT_vals[g] * f * dt / Cv for g in range(G)]
    return f, alpha_g


def _richardson_solve(matvec, b, x, max_its, L2_tol, Linf_tol, LOUD):
    total_its = 0
    for _ in range(max_its):
        x_new = matvec(x) + b
        L2err = np.sqrt(np.mean(((x - x_new) / (np.abs(x_new) + 1e-14)) ** 2))
        Linferr = np.max(np.abs(x - x_new) / (np.max(np.abs(x_new)) + 1e-14))
        x = x_new
        total_its += 1
        if LOUD:
            print(f"  Richardson {total_its}: L2={L2err:.3e}  Linf={Linferr:.3e}")
        if L2err < L2_tol and Linferr < Linf_tol:
            break
    return x, total_its


def _zero_like_boundary_arrays(Ix, Iy, M):
    return (
        np.zeros((Iy, M, 2)),
        np.zeros((Iy, M, 2)),
        np.zeros((Ix, M, 2)),
        np.zeros((Ix, M, 2)),
    )


def mg_temp_solve_2d(
    Ix,
    Iy,
    dx_arr,
    dy_arr,
    G,
    sigma_a_funcs,
    scat_funcs,
    Bg_funcs,
    dBdT_funcs,
    q_ext,
    quad_type,
    N_quad,
    BCs_func,
    EOS,
    invEOS,
    Cv_func,
    phi_g,
    psi_g,
    T,
    dt_min=1e-5,
    dt_max=1e-3,
    tfinal=1.0,
    Linf_tol=1e-5,
    tolerance=1e-8,
    maxits=100,
    LOUD=False,
    K=50,
    R=3,
    time_outputs=None,
    chi=None,
    reflect_xlo=False,
    reflect_xhi=False,
    reflect_ylo=False,
    reflect_yhi=False,
    use_dmd=True,
    print_stride=0,
    store_full_history=True,
    step_callback=None,
    step_callback_groups=None,
    step_callback_angular=None,
    fleck_mode="legacy",
    q_ext_ang=None,
    compton_matrix_func=None,
    kompaneets_options=None,
    compton_induced_mode="wien",
    compton_tolerance=1.0e-10,
    compton_max_iterations=50,
    compton_relaxation=1.0,
    compton_positivity_limiter="damp",
    compton_on_nonconvergence="raise",
    compton_warning_stride=100,
    compton_max_halvings=0,
    initial_time=0.0,
    checkpoint_callback=None,
):
    """Time-dependent multigroup TRT in 2-D Cartesian geometry.

    When ``compton_matrix_func`` is provided, a local operator-split
    full-Boltzmann Compton frequency redistribution is applied after the
    spatial transport/material update.  The callable receives the current
    temperature array and returns ``DiscreteOrdinates.compton.ComptonMatrices``.
    ``scat_funcs`` remains the physical angle-scattering opacity used in the
    S_N transport sweep.

    When ``kompaneets_options`` is provided instead, the local linear
    Kompaneets frequency operator is applied at the same split point.  It is
    mutually exclusive with ``compton_matrix_func``.

    ``initial_time`` permits a restart from the state supplied in ``phi_g``,
    ``psi_g``, and ``T``.  When supplied, ``checkpoint_callback`` is called
    after each completed step with ``(time, phi_g, psi_g, e, T)``.
    """
    if initial_time < 0.0:
        raise ValueError("initial_time must be non-negative")
    if compton_matrix_func is not None and kompaneets_options is not None:
        raise ValueError(
            "use either compton_matrix_func or kompaneets_options, not both"
        )
    if compton_on_nonconvergence not in {"raise", "warn", "accept"}:
        raise ValueError("compton_on_nonconvergence must be 'raise', 'warn', or 'accept'")
    if compton_warning_stride < 1:
        raise ValueError("compton_warning_stride must be at least one")
    if compton_max_halvings < 0:
        raise ValueError("compton_max_halvings must be nonnegative")
    Omega_x, Omega_y, weights = get_2d_quadrature(quad_type, N_quad)
    M = len(Omega_x)

    Omega_x = np.ascontiguousarray(Omega_x, dtype=np.float64)
    Omega_y = np.ascontiguousarray(Omega_y, dtype=np.float64)
    weights = np.ascontiguousarray(weights, dtype=np.float64)
    dx_arr = np.ascontiguousarray(dx_arr, dtype=np.float64)
    dy_arr = np.ascontiguousarray(dy_arr, dtype=np.float64)

    _bc_per_group = isinstance(BCs_func, (list, tuple))

    phi_g = [np.ascontiguousarray(p.copy(), dtype=np.float64) for p in phi_g]
    psi_g = [np.ascontiguousarray(p.copy(), dtype=np.float64) for p in psi_g]
    psi_g_old = [p.copy() for p in psi_g]

    T_old = np.ascontiguousarray(T.copy(), dtype=np.float64)
    T_old2 = T_old.copy()
    e_old = EOS(T_old)

    phi_g_hist = [[p.copy() for p in phi_g]]
    T_hist = [T_old.copy()]
    ts = [initial_time]
    its_per_step = []

    zero_xlo, zero_xhi, zero_ylo, zero_yhi = _zero_like_boundary_arrays(Ix, Iy, M)
    phi_angle_workspace = np.empty((M, Ix, Iy, 4))

    reflection_enabled = (reflect_xlo or reflect_xhi or reflect_ylo or reflect_yhi)
    if reflection_enabled:
        reflect_map_x, reflect_map_y = _build_reflection_maps(Omega_x, Omega_y)
        psi_xlo_out_state = [np.zeros((Iy, M, 2)) for _ in range(G)]
        psi_xhi_out_state = [np.zeros((Iy, M, 2)) for _ in range(G)]
        psi_ylo_out_state = [np.zeros((Ix, M, 2)) for _ in range(G)]
        psi_yhi_out_state = [np.zeros((Ix, M, 2)) for _ in range(G)]
    else:
        reflect_map_x = None
        reflect_map_y = None

    iterations = 0
    step_num = 0
    t_current = initial_time
    dt = dt_min
    dt_old = dt_min
    deriv_val = 0.0
    delta_step = 1e-3
    curr_step = 0
    t_output_index = 0

    if time_outputs is not None:
        time_outputs = np.asarray(time_outputs, dtype=float)

    if step_callback is not None:
        phi_total = np.sum(np.stack(phi_g, axis=0), axis=0)
        step_callback(t_current, phi_total, T_old)
    if step_callback_groups is not None:
        step_callback_groups(t_current, phi_g, T_old)
    if step_callback_angular is not None:
        step_callback_angular(t_current, phi_g, psi_g, T_old)

    print(f"MG-2D-S_N Cartesian: G={G}, Ix={Ix}, Iy={Iy}, M={M}, quad={quad_type} N={N_quad}")
    print(f"  dt range: [{dt_min:.2e}, {dt_max:.2e}], tfinal={tfinal}")
    print(f"  tolerances: L2={tolerance:.1e}, Linf={Linf_tol:.1e}, maxits={maxits}")
    print("|", end="", flush=True)

    while t_current < tfinal:
        dt_old2 = dt_old
        dt_old = dt
        step_num += 1

        if step_num > 2:
            dt_prop = math.sqrt(delta_step * deriv_val) if deriv_val > 0 else dt_max
            dt_prop = max(dt_min, min(dt_max, dt_prop))
            if dt_prop > 2.0 * dt:
                dt_prop = dt * 1.5
            dt = dt_prop
        else:
            dt = dt_min

        if (tfinal - t_current) < dt:
            snap = tfinal - t_current
            if snap > 1e-10 * dt_min:
                dt = snap
            else:
                break

        if time_outputs is not None:
            out_tol = max(1e-12, 1e-10 * max(abs(t_current), dt_min))
            while (t_output_index < len(time_outputs) and
                   time_outputs[t_output_index] <= t_current + out_tol):
                t_output_index += 1
            if t_output_index < len(time_outputs) and t_current + dt > time_outputs[t_output_index]:
                snap_dt = time_outputs[t_output_index] - t_current
                if snap_dt > 1e-10 * dt_min:
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

        if _bc_per_group:
            bc_data_all = [BCs_func[g](t_current - dt / 2.0) for g in range(G)]
        else:
            bc_data_one = BCs_func(t_current - dt / 2.0)
            bc_data_all = [bc_data_one for _ in range(G)]

        BCs_xlo_base = []
        BCs_xhi_base = []
        BCs_ylo_base = []
        BCs_yhi_base = []
        for g in range(G):
            BCs_xlo_g, BCs_xhi_g, BCs_ylo_g, BCs_yhi_g = _zero_like_boundary_arrays(Ix, Iy, M)
            bc_data = bc_data_all[g]
            if bc_data is not None:
                if bc_data.get("xlo") is not None:
                    bc_xlo = bc_data["xlo"]
                    BCs_xlo_g = np.full((Iy, M, 2), bc_xlo, dtype=np.float64) if np.isscalar(bc_xlo) else np.ascontiguousarray(bc_xlo, dtype=np.float64)
                if bc_data.get("xhi") is not None:
                    bc_xhi = bc_data["xhi"]
                    BCs_xhi_g = np.full((Iy, M, 2), bc_xhi, dtype=np.float64) if np.isscalar(bc_xhi) else np.ascontiguousarray(bc_xhi, dtype=np.float64)
                if bc_data.get("ylo") is not None:
                    bc_ylo = bc_data["ylo"]
                    BCs_ylo_g = np.full((Ix, M, 2), bc_ylo, dtype=np.float64) if np.isscalar(bc_ylo) else np.ascontiguousarray(bc_ylo, dtype=np.float64)
                if bc_data.get("yhi") is not None:
                    bc_yhi = bc_data["yhi"]
                    BCs_yhi_g = np.full((Ix, M, 2), bc_yhi, dtype=np.float64) if np.isscalar(bc_yhi) else np.ascontiguousarray(bc_yhi, dtype=np.float64)
            BCs_xlo_base.append(BCs_xlo_g)
            BCs_xhi_base.append(BCs_xhi_g)
            BCs_ylo_base.append(BCs_ylo_g)
            BCs_yhi_base.append(BCs_yhi_g)

        q_now = []
        for g in range(G):
            qg = q_ext[g]
            q_now.append(qg(t_current) if callable(qg) else qg)

        q_ang_now = [None] * G
        if q_ext_ang is not None:
            for g in range(G):
                qag = q_ext_ang[g]
                qag_now = qag(t_current) if callable(qag) else qag
                q_ang_now[g] = np.ascontiguousarray(qag_now, dtype=np.float64)

        sigma_ag = [sigma_a_funcs[g](T_old) for g in range(G)]
        scat_g = [scat_funcs[g](T_old) for g in range(G)]
        Bg_vals = [Bg_funcs[g](T_old) for g in range(G)]

        if fleck_mode == "legacy":
            f, alpha_g = _compute_fleck_and_alpha_2d(
                T_old, dt, sigma_a_funcs, dBdT_funcs, Cv_func, G
            )
            if chi is not None:
                one_minus_f = 1.0 - f
                alpha_g = [chi[g] * one_minus_f for g in range(G)]
        elif fleck_mode == "imc":
            Cv = Cv_func(T_old)
            beta = 4.0 * a * T_old ** 3 / Cv
            if chi is None:
                sum_B = np.zeros_like(T_old)
                for g in range(G):
                    sum_B += np.maximum(Bg_vals[g], 0.0)
                b_star = [np.maximum(Bg_vals[g], 0.0) / (sum_B + 1e-300) for g in range(G)]
            else:
                b_star = [chi[g] + np.zeros_like(T_old) for g in range(G)]

            sigma_P = np.zeros_like(T_old)
            for g in range(G):
                sigma_P += sigma_ag[g] * b_star[g]

            f = 1.0 / (1.0 + beta * c * sigma_P * dt)
            f = np.clip(f, 0.0, 1.0)
            one_minus_f = 1.0 - f
            alpha_g = [one_minus_f * sigma_ag[g] * b_star[g] / (sigma_P + 1e-300) for g in range(G)]
        else:
            raise ValueError(f"Unknown fleck_mode: {fleck_mode!r}")

        sum_sigma_B = np.zeros_like(T_old)
        for g in range(G):
            sum_sigma_B += sigma_ag[g] * Bg_vals[g]

        sigma_hat_g = []
        source_iso_fixed_g = []
        for g in range(G):
            sigma_hat_g.append(np.ascontiguousarray(sigma_ag[g] + scat_g[g] + icdt))
            emission_fixed = sigma_ag[g] * Bg_vals[g] - alpha_g[g] * sum_sigma_B
            source_iso_fixed_g.append(np.ascontiguousarray(q_now[g] + emission_fixed))

        block = Ix * Iy * 4
        x_sol = np.concatenate([phi_g[g].ravel() for g in range(G)])

        def _make_mv(scat_g_in, alpha_g_in, sigma_ag_in, sigma_hat_g_in):
            _scat = [s.copy() for s in scat_g_in]
            _alpha = [a_.copy() for a_ in alpha_g_in]
            _sigma_a = [s.copy() for s in sigma_ag_in]
            _sigma_hat = [s.copy() for s in sigma_hat_g_in]
            _G = len(_scat)

            def mv(x_vec):
                result = np.zeros_like(x_vec)
                phi_parts = [x_vec[g * block:(g + 1) * block].reshape((Ix, Iy, 4)) for g in range(_G)]
                sum_sigma_phi = np.zeros((Ix, Iy, 4))
                for gp in range(_G):
                    sum_sigma_phi += _sigma_a[gp] * phi_parts[gp]

                for g in range(_G):
                    scatter_iso = np.ascontiguousarray(_scat[g] * phi_parts[g] + _alpha[g] * sum_sigma_phi)
                    phi_new, _, _, _, _ = sweep_2d_iso(
                        Ix, Iy, dx_arr, dy_arr, _sigma_hat[g],
                        scatter_iso,
                        Omega_x, Omega_y, weights,
                        zero_xlo, zero_xhi, zero_ylo, zero_yhi,
                        True, False, phi_angle_workspace
                    )
                    result[g * block:(g + 1) * block] = phi_new.ravel()
                return result

            return mv

        matvec = _make_mv(scat_g, alpha_g, sigma_ag, sigma_hat_g)

        psi_bc_state = [p.copy() for p in psi_g_old]
        max_reflect_its = 20 if reflection_enabled else 1
        reflect_tol = 1e-10

        for _ref_it in range(max_reflect_its):
            b_vec = np.zeros(G * block)

            BCs_xlo_use = []
            BCs_xhi_use = []
            BCs_ylo_use = []
            BCs_yhi_use = []

            for g in range(G):
                xlo_g = BCs_xlo_base[g].copy()
                xhi_g = BCs_xhi_base[g].copy()
                ylo_g = BCs_ylo_base[g].copy()
                yhi_g = BCs_yhi_base[g].copy()

                if reflection_enabled:
                    ref_xlo, ref_xhi, ref_ylo, ref_yhi = build_reflecting_BCs_2d(
                        Omega_x, Omega_y,
                        psi_xlo_out_state[g], psi_xhi_out_state[g],
                        psi_ylo_out_state[g], psi_yhi_out_state[g],
                        reflect_xlo, reflect_xhi, reflect_ylo, reflect_yhi,
                        reflect_map_x, reflect_map_y,
                    )
                    if reflect_xlo:
                        xlo_g = ref_xlo
                    if reflect_xhi:
                        xhi_g = ref_xhi
                    if reflect_ylo:
                        ylo_g = ref_ylo
                    if reflect_yhi:
                        yhi_g = ref_yhi

                BCs_xlo_use.append(np.ascontiguousarray(xlo_g))
                BCs_xhi_use.append(np.ascontiguousarray(xhi_g))
                BCs_ylo_use.append(np.ascontiguousarray(ylo_g))
                BCs_yhi_use.append(np.ascontiguousarray(yhi_g))

                if q_ang_now[g] is None:
                    _, b_phi, _, _, _, _ = sweep_2d_transient(
                        Ix, Iy, dx_arr, dy_arr, sigma_hat_g[g],
                        source_iso_fixed_g[g],
                        psi_g_old[g], icdt,
                        Omega_x, Omega_y, weights,
                        BCs_xlo_use[g], BCs_xhi_use[g], BCs_ylo_use[g], BCs_yhi_use[g],
                        True, False, False, phi_angle_workspace,
                    )
                else:
                    psi_time_eff = np.ascontiguousarray(
                        psi_g_old[g] + q_ang_now[g] / icdt,
                        dtype=np.float64,
                    )
                    _, b_phi, _, _, _, _ = sweep_2d_transient(
                        Ix, Iy, dx_arr, dy_arr, sigma_hat_g[g],
                        source_iso_fixed_g[g],
                        psi_time_eff, icdt,
                        Omega_x, Omega_y, weights,
                        BCs_xlo_use[g], BCs_xhi_use[g], BCs_ylo_use[g], BCs_yhi_use[g],
                        True, False, False, phi_angle_workspace,
                    )
                b_vec[g * block:(g + 1) * block] = b_phi.ravel()

            if use_dmd:
                try:
                    x_sol, total_its, *_ = _solver_with_dmd_inc_1d(
                        matvec=matvec,
                        b=b_vec,
                        K=K,
                        max_its=maxits,
                        steady=1,
                        x=x_sol,
                        Rits=R,
                        LOUD=LOUD,
                        L2_tol=tolerance,
                        Linf_tol=Linf_tol,
                    )
                    if not np.all(np.isfinite(x_sol)):
                        raise np.linalg.LinAlgError("non-finite")
                except np.linalg.LinAlgError:
                    x_sol, total_its = _richardson_solve(
                        matvec, b_vec, x_sol, maxits, tolerance, Linf_tol, LOUD
                    )
            else:
                x_sol, total_its = _richardson_solve(
                    matvec, b_vec, x_sol, maxits, tolerance, Linf_tol, LOUD
                )

            iterations += total_its
            iterations_step += total_its

            phi_g = [x_sol[g * block:(g + 1) * block].reshape((Ix, Iy, 4)) for g in range(G)]

            sum_sigma_phi = np.zeros_like(T_old)
            for gp in range(G):
                sum_sigma_phi += sigma_ag[gp] * phi_g[gp]

            psi_candidate = [None] * G
            psi_xlo_new = [None] * G
            psi_xhi_new = [None] * G
            psi_ylo_new = [None] * G
            psi_yhi_new = [None] * G

            for g in range(G):
                coupled_iso = scat_g[g] * phi_g[g] + alpha_g[g] * sum_sigma_phi
                total_iso = np.ascontiguousarray(source_iso_fixed_g[g] + coupled_iso)
                if q_ang_now[g] is None:
                    psi_new_g, _, pxlo, pxhi, pylo, pyhi = sweep_2d_transient(
                        Ix, Iy, dx_arr, dy_arr, sigma_hat_g[g],
                        total_iso,
                        psi_g_old[g], icdt,
                        Omega_x, Omega_y, weights,
                        BCs_xlo_use[g], BCs_xhi_use[g], BCs_ylo_use[g], BCs_yhi_use[g],
                        False, True, True, phi_angle_workspace,
                    )
                else:
                    psi_time_eff = np.ascontiguousarray(
                        psi_g_old[g] + q_ang_now[g] / icdt,
                        dtype=np.float64,
                    )
                    psi_new_g, _, pxlo, pxhi, pylo, pyhi = sweep_2d_transient(
                        Ix, Iy, dx_arr, dy_arr, sigma_hat_g[g],
                        total_iso,
                        psi_time_eff, icdt,
                        Omega_x, Omega_y, weights,
                        BCs_xlo_use[g], BCs_xhi_use[g], BCs_ylo_use[g], BCs_yhi_use[g],
                        False, True, True, phi_angle_workspace,
                    )
                psi_candidate[g] = psi_new_g
                psi_xlo_new[g] = pxlo
                psi_xhi_new[g] = pxhi
                psi_ylo_new[g] = pylo
                psi_yhi_new[g] = pyhi

            if not reflection_enabled:
                psi_g = psi_candidate
                break

            bc_change = 0.0
            for g in range(G):
                bc_change = max(bc_change, np.max(np.abs(psi_xlo_new[g] - psi_xlo_out_state[g])))
                bc_change = max(bc_change, np.max(np.abs(psi_xhi_new[g] - psi_xhi_out_state[g])))
                bc_change = max(bc_change, np.max(np.abs(psi_ylo_new[g] - psi_ylo_out_state[g])))
                bc_change = max(bc_change, np.max(np.abs(psi_yhi_new[g] - psi_yhi_out_state[g])))

                psi_xlo_out_state[g] = psi_xlo_new[g]
                psi_xhi_out_state[g] = psi_xhi_new[g]
                psi_ylo_out_state[g] = psi_ylo_new[g]
                psi_yhi_out_state[g] = psi_yhi_new[g]

            psi_g = psi_candidate
            if bc_change < reflect_tol:
                break

        energy_dep = np.zeros_like(T_old)
        for g in range(G):
            energy_dep += sigma_ag[g] * (phi_g[g] - Bg_vals[g])

        e = e_old + f * dt * energy_dep
        T_new = invEOS(e)

        if compton_matrix_func is not None:
            compton_remaining = dt
            compton_substep = dt
            compton_substeps = 0
            compton_halvings = 0
            compton_diagnostics = None
            compton_time_tolerance = max(1.0e-15, 1.0e-12 * dt)
            while compton_remaining > compton_time_tolerance:
                trial_dt = min(compton_substep, compton_remaining)
                try:
                    result = operator_split_compton_update(
                        phi_g, psi_g, e, T_new, trial_dt, c, invEOS, Bg_funcs,
                        compton_matrix_func,
                        induced_mode=compton_induced_mode,
                        tolerance=compton_tolerance,
                        max_iterations=compton_max_iterations,
                        relaxation=compton_relaxation,
                        positivity_limiter=compton_positivity_limiter,
                        on_nonconvergence=(
                            "raise" if compton_on_nonconvergence == "raise" else "accept"
                        ),
                        return_diagnostics=True,
                    )
                except RuntimeError as error:
                    retryable = (
                        "Compton update" in str(error)
                        or "Compton Picard" in str(error)
                        or "operator-split Compton update did not converge" in str(error)
                    )
                    if not retryable or compton_halvings >= compton_max_halvings:
                        raise
                    compton_substep *= 0.5
                    compton_halvings += 1
                    continue

                phi_g, psi_g, e, T_new, _, trial_diagnostics = result
                compton_remaining -= trial_dt
                compton_substeps += 1
                if compton_diagnostics is None:
                    compton_diagnostics = trial_diagnostics.copy()
                else:
                    for key in (
                        "residual", "temperature_residual", "intensity_residual",
                        "positivity_limited_cells",
                    ):
                        compton_diagnostics[key] = max(
                            compton_diagnostics[key], trial_diagnostics[key]
                        )
                    for key in (
                        "minimum_positivity_damping", "minimum_picard_relaxation",
                    ):
                        compton_diagnostics[key] = min(
                            compton_diagnostics[key], trial_diagnostics[key]
                        )
                    compton_diagnostics["converged"] &= trial_diagnostics["converged"]
            compton_diagnostics["substeps"] = compton_substeps
            compton_diagnostics["halvings"] = compton_halvings
            if (compton_halvings > 0
                    and (step_num <= 20 or step_num % compton_warning_stride == 0)):
                print(
                    "\n  Compton subcycling at step "
                    f"{step_num}, t={t_current:.6e} ns: {compton_substeps} substeps, "
                    f"{compton_halvings} halvings, minimum dt={compton_substep:.3e} ns."
                )
            if (not compton_diagnostics["converged"]
                    and compton_on_nonconvergence == "warn"
                    and (step_num == 1 or step_num % compton_warning_stride == 0)):
                print(
                    "\nWARNING: accepting nonconverged Compton Picard update "
                    f"at step {step_num}, t={t_current:.6e} ns after "
                    f"{compton_max_iterations} iterations; residual="
                    f"{compton_diagnostics['residual']:.3e} "
                    f"(T={compton_diagnostics['temperature_residual']:.3e}, "
                    f"phi={compton_diagnostics['intensity_residual']:.3e}; "
                    f"tolerance={compton_diagnostics['tolerance']:.3e}, "
                    "minimum relaxation="
                    f"{compton_diagnostics['minimum_picard_relaxation']:.3e})."
                )
            if (compton_diagnostics["positivity_limited_cells"] > 0
                    and (step_num == 1 or step_num % compton_warning_stride == 0)):
                print(
                    "\nWARNING: positivity-limited Compton update at "
                    f"step {step_num}, t={t_current:.6e} ns; "
                    f"{compton_diagnostics['positivity_limited_cells']} local "
                    "group solves were limited, with minimum damping "
                    f"{compton_diagnostics['minimum_positivity_damping']:.3e}."
                )
        elif kompaneets_options is not None:
            phi_g, psi_g, e, T_new, _ = operator_split_kompaneets_update(
                phi_g, psi_g, e, T_new, dt, c, invEOS,
                **kompaneets_options,
            )

        if print_stride > 0 and step_num % print_stride == 0:
            print(
                f"  step {step_num:5d}  t={t_current:.4e} ns  dt={dt:.3e}  "
                f"T_max={np.max(T_new):.6f} keV  sweeps={iterations_step}"
            )
        elif step_num <= 3 or (step_num <= 20 and step_num % 5 == 0):
            print(
                f"  step {step_num:5d}  t={t_current:.4e} ns  dt={dt:.3e}  "
                f"T_max={np.max(T_new):.6f} keV  sweeps={iterations_step}"
            )

        if step_num >= 2:
            curvature = np.mean(np.abs(2.0 * (
                T_new.ravel() / (dt * (dt + dt_old))
                - T_old.ravel() / (dt * dt_old)
                + T_old2.ravel() / (dt_old * (dt + dt_old))
            )))
            mean_value = float(np.mean(np.abs(T_new)))
            if (curvature > 0.0 and math.isfinite(mean_value)
                    and math.isfinite(curvature)):
                deriv_val = mean_value / curvature
            else:
                deriv_val = dt_max ** 2 / delta_step

        e_old = e.copy()
        T_old2 = T_old.copy()
        T_old = T_new.copy()
        psi_g_old = [p.copy() for p in psi_g]
        its_per_step.append(iterations_step)

        phi_total = np.sum(np.stack(phi_g, axis=0), axis=0)
        if step_callback is not None:
            step_callback(t_current, phi_total, T_new)
        if step_callback_groups is not None:
            step_callback_groups(t_current, phi_g, T_new)
        if step_callback_angular is not None:
            step_callback_angular(t_current, phi_g, psi_g, T_new)
        if checkpoint_callback is not None:
            checkpoint_callback(t_current, phi_g, psi_g, e, T_new)

        store_step = store_full_history
        if not store_step:
            store_tol = max(1e-12, 1e-10 * max(abs(t_current), dt_min))
            at_final = abs(t_current - tfinal) <= store_tol
            at_output = (time_outputs is not None and np.any(np.abs(time_outputs - t_current) <= store_tol))
            store_step = at_final or at_output

        if store_step:
            phi_g_hist.append([p.copy() for p in phi_g])
            T_hist.append(T_new.copy())
            ts.append(t_current)

    print(
        f"\n  Finished: {step_num} steps, {iterations} total sweeps, "
        f"avg {iterations / max(step_num, 1):.1f} sweeps/step"
    )
    return phi_g_hist, T_hist, iterations, np.array(ts), its_per_step
