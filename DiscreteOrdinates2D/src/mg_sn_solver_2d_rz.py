"""
2-D axisymmetric cylindrical (r-z) multigroup Discrete Ordinates solver.

This extends the gray r-z solver to multigroup coupling. Group coupling follows
1-D multigroup logic through Fleck-linearized emission weights.

Notes
-----
- The current r-z sweep uses isotropic source terms. As in the existing gray
  r-z solver, the time derivative contribution is represented with icdt * phi.
- Unknowns are group corner-averaged scalar fluxes phi_g[g] with shape
  (Ir, Iz, 4).
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

from DiscreteOrdinates.src.sn_solver import solver_with_dmd_inc as _solver_with_dmd_inc_1d
from .sn_solver_2d import c, a
from .sn_solver_2d_rz import get_rz_quadrature, sweep_rz


def _compute_fleck_and_alpha_rz(T, dt, sigma_a_funcs, dBdT_funcs, Cv_func, G):
    """Return Fleck factor and multigroup coupling weights on (Ir, Iz, 4)."""
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


def mg_temp_solve_rz(
    Ir,
    Iz,
    dr_arr,
    dz_arr,
    r_faces,
    z_faces,
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
    reflect_zlo=False,
    reflect_zhi=False,
    use_dmd=True,
    print_stride=0,
    store_full_history=True,
    step_callback=None,
    fleck_mode="legacy",
):
    """Time-dependent multigroup TRT in 2-D r-z geometry."""
    levels = get_rz_quadrature(quad_type, N_quad)
    w_total = sum(lv["w_lk"].sum() for lv in levels)

    r_lo_arr = np.ascontiguousarray(r_faces[:-1], dtype=np.float64)
    r_hi_arr = np.ascontiguousarray(r_faces[1:], dtype=np.float64)
    r_ctr = 0.5 * (r_lo_arr + r_hi_arr)
    on_axis = r_faces[0] < 1e-15

    dr_arr = np.ascontiguousarray(dr_arr, dtype=np.float64)
    dz_arr = np.ascontiguousarray(dz_arr, dtype=np.float64)

    _bc_per_group = isinstance(BCs_func, (list, tuple))

    phi_g = [np.ascontiguousarray(p.copy(), dtype=np.float64) for p in phi_g]
    T_old = np.ascontiguousarray(T.copy(), dtype=np.float64)
    T_old2 = T_old.copy()
    e_old = EOS(T_old)

    phi_g_hist = [[p.copy() for p in phi_g]]
    T_hist = [T_old.copy()]
    ts = [0.0]
    its_per_step = []

    iterations = 0
    step_num = 0
    t_current = 0.0
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
        step_callback(0.0, phi_total, T_old)

    print(
        f"MG-2D-S_N r-z: G={G}, Ir={Ir}, Iz={Iz}, "
        f"quad={quad_type} N={N_quad}, levels={len(levels)}"
    )
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
            bc_one = BCs_func(t_current - dt / 2.0)
            bc_data_all = [bc_one for _ in range(G)]

        BCs_rlo_all = []
        BCs_rhi_all = []
        BCs_zlo_all = []
        BCs_zhi_all = []
        for g in range(G):
            bc_data = bc_data_all[g]
            BCs_rlo_g = None
            BCs_rhi_g = None
            BCs_zlo_g = None
            BCs_zhi_g = None
            if bc_data is not None:
                if bc_data.get("rlo") is not None:
                    BCs_rlo_g = np.ascontiguousarray(bc_data["rlo"], dtype=np.float64)
                if bc_data.get("rhi") is not None:
                    BCs_rhi_g = np.ascontiguousarray(bc_data["rhi"], dtype=np.float64)
                if bc_data.get("zlo") is not None:
                    BCs_zlo_g = np.ascontiguousarray(bc_data["zlo"], dtype=np.float64)
                if bc_data.get("zhi") is not None:
                    BCs_zhi_g = np.ascontiguousarray(bc_data["zhi"], dtype=np.float64)

            BCs_rlo_all.append(BCs_rlo_g)
            BCs_rhi_all.append(BCs_rhi_g)
            BCs_zlo_all.append(BCs_zlo_g)
            BCs_zhi_all.append(BCs_zhi_g)

        q_now = []
        for g in range(G):
            qg = q_ext[g]
            q_now.append(qg(t_current) if callable(qg) else qg)

        sigma_ag = [sigma_a_funcs[g](T_old) for g in range(G)]
        scat_g = [scat_funcs[g](T_old) for g in range(G)]
        Bg_vals = [Bg_funcs[g](T_old) for g in range(G)]

        if fleck_mode == "legacy":
            f, alpha_g = _compute_fleck_and_alpha_rz(
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
            # Keep same isotropic time-derivative treatment as existing gray r-z solver.
            source_iso_fixed_g.append(np.ascontiguousarray(q_now[g] + emission_fixed + icdt * phi_g[g]))

        block = Ir * Iz * 4
        x_sol = np.concatenate([phi_g[g].ravel() for g in range(G)])

        def _make_mv(scat_g_in, alpha_g_in, sigma_ag_in, sigma_hat_g_in):
            _scat = [s.copy() for s in scat_g_in]
            _alpha = [a_.copy() for a_ in alpha_g_in]
            _sigma_a = [s.copy() for s in sigma_ag_in]
            _sigma_hat = [s.copy() for s in sigma_hat_g_in]
            _G = len(_scat)

            def mv(x_vec):
                result = np.zeros_like(x_vec)
                phi_parts = [x_vec[g * block:(g + 1) * block].reshape((Ir, Iz, 4)) for g in range(_G)]
                sum_sigma_phi = np.zeros((Ir, Iz, 4))
                for gp in range(_G):
                    sum_sigma_phi += _sigma_a[gp] * phi_parts[gp]

                for g in range(_G):
                    scatter_iso = np.ascontiguousarray(_scat[g] * phi_parts[g] + _alpha[g] * sum_sigma_phi)
                    phi_new, _, _ = sweep_rz(
                        Ir, Iz, dr_arr, dz_arr, r_lo_arr, r_hi_arr, r_ctr,
                        _sigma_hat[g], scatter_iso, levels,
                        None, None, None, None,
                        on_axis, w_total,
                        reflect_zlo=False, reflect_zhi=False,
                    )
                    result[g * block:(g + 1) * block] = phi_new.ravel()
                return result

            return mv

        matvec = _make_mv(scat_g, alpha_g, sigma_ag, sigma_hat_g)

        b_vec = np.zeros(G * block)
        for g in range(G):
            phi_b, _, _ = sweep_rz(
                Ir, Iz, dr_arr, dz_arr, r_lo_arr, r_hi_arr, r_ctr,
                sigma_hat_g[g], source_iso_fixed_g[g], levels,
                BCs_rlo_all[g], BCs_rhi_all[g], BCs_zlo_all[g], BCs_zhi_all[g],
                on_axis, w_total,
                reflect_zlo=reflect_zlo, reflect_zhi=reflect_zhi,
            )
            b_vec[g * block:(g + 1) * block] = phi_b.ravel()

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

        phi_g = [x_sol[g * block:(g + 1) * block].reshape((Ir, Iz, 4)) for g in range(G)]

        energy_dep = np.zeros_like(T_old)
        for g in range(G):
            energy_dep += sigma_ag[g] * (phi_g[g] - Bg_vals[g])

        e = e_old + f * dt * energy_dep
        T_new = invEOS(e)

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
            denom = np.mean(np.abs(2.0 * (
                T_new.ravel() / (dt * (dt + dt_old))
                - T_old.ravel() / (dt * dt_old)
                + T_old2.ravel() / (dt_old * (dt + dt_old))
            )))
            mean_T = float(np.mean(T_new))
            if denom > 0 and math.isfinite(mean_T) and math.isfinite(denom):
                deriv_val = mean_T / denom
            else:
                deriv_val = dt_max ** 2 / delta_step

        e_old = e.copy()
        T_old2 = T_old.copy()
        T_old = T_new.copy()
        its_per_step.append(iterations_step)

        phi_total = np.sum(np.stack(phi_g, axis=0), axis=0)
        if step_callback is not None:
            step_callback(t_current, phi_total, T_new)

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
