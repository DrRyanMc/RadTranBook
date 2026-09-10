"""
Multigroup 1-D Discrete Ordinates (S_N) solver for **spherical** geometry
using Linear Discontinuous (LD) Galerkin elements.

This module composes the validated gray spherical LD sweep
(:func:`sn_solver_ld_sphere.single_sweep_phi_sph_ld` /
:func:`single_sweep_psi_sph_ld`) over *G* energy groups, coupling the groups
through the Fleck-factor linearised emission exactly as the slab multigroup
solver :mod:`mg_sn_solver_ld` does.

The time-discretised transport equation for group *g* is

  (1/(cΔt) + σ̂_{t,g}) ψ_{n,g} = S_g  +  (spherical angular/streaming terms)

with the isotropic source

  S_g = σ_{s,g} φ_g  +  α_g Σ_{g'} σ_{a,g'} φ_{g'}      ← φ-dependent
        + σ_{a,g} B_g(T^n) − α_g Σ_{g'} σ_{a,g'} B_{g'}(T^n)
        + (1/(cΔt)) ψ_{n,g}^n  +  Q_g                    ← fixed

where

  f   = 1 / (1 + (Δt/C_v) Σ_g σ_{a,g} dB_g/dT)          Fleck factor
  α_g = σ_{a,g} (dB_g/dT) f Δt / C_v                    coupling weight

and B_g, dB_g/dT are the "4π" forms (i.e. 4πB_g, 4π dB_g/dT) such that
Σ_g B_g(T) = a c T⁴.

Spherical-geometry specifics inherited from :mod:`sn_solver_ld_sphere`
---------------------------------------------------------------------
- An auxiliary starting-direction intensity ``g_g(r, μ=−1, t)`` is tracked
  per group alongside ``psi_g``.
- The quadrature weights ``W`` sum to 1, so the scalar flux is
  ``φ = 2 Σ_n W_n ψ_n`` and the isotropic emission/scattering source enters
  each ordinate with a ½ factor.
- For a hollow shell (``r_left[0] > 0``) the inner-wall inflow is supplied
  through ``BCs_inner`` (μ_n > 0 directions); for a full sphere
  (``r_left[0] == 0``) the origin regularity condition is applied
  automatically and ``BCs_inner`` is ignored.

Array shapes
------------
- phi_g[g], T, e          : (I, 2)      — 0 = left edge, 1 = right edge
- psi_g[g]                : (I, N, 2)
- g_g[g]                  : (I, 2)      — starting-direction intensity
- BCs_outer              : list of G callables  t → (bc_outer (N,2), bc_g_outer float)
- BCs_inner              : list of G callables  t → (N, 2)   or None
"""

import math
import numpy as np

from . import sn_solver_ld_sphere
from .sn_solver_ld_sphere import (
    c, a, ac,
    single_sweep_phi_sph_ld,
    single_sweep_psi_sph_ld,
    _get_quadrature,
)
from .sn_solver import solver_with_dmd_inc


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _compute_fleck_and_alpha(T, dt, sigma_a_funcs, dBdT_funcs, Cv_func, G):
    """Multigroup Fleck factor *f* and coupling weights *α_g* for LD arrays.

    Identical formulation to ``mg_sn_solver_ld._compute_fleck_and_alpha_ld``.

    Parameters
    ----------
    T : (I, 2)
        Temperature at left and right cell edges.
    dt : float
        Time step.
    sigma_a_funcs : list of G callables  T → (I, 2)
    dBdT_funcs : list of G callables  T → (I, 2)  (4π dB_g/dT)
    Cv_func : callable  T → (I, 2)
    G : int

    Returns
    -------
    f      : (I, 2)
    alpha_g : list of G  (I, 2)
    """
    Cv = Cv_func(T)
    coupling_sum = np.zeros_like(T)
    dBdT_vals = []
    for g in range(G):
        dB = dBdT_funcs[g](T)
        dBdT_vals.append(dB)
        coupling_sum += sigma_a_funcs[g](T) * dB
    f = 1.0 / (1.0 + dt / Cv * coupling_sum)
    alpha_g = [sigma_a_funcs[g](T) * dBdT_vals[g] * f * dt / Cv
               for g in range(G)]
    return f, alpha_g


def radiation_flux_by_group(psi_g, N):
    r"""Net radial radiative flux per group, ``F_g = ∫_{-1}^{1} μ I_g dμ``.

    With the spherical quadrature (``W`` summing to 1) the discrete moment is
    ``F_g = 2 Σ_n W_n μ_n ψ_{g,n}``.  The result has the same units as the
    scalar flux ``φ`` (i.e. ``c · E_rad``), namely GJ/(cm²·ns).

    Parameters
    ----------
    psi_g : list of G ndarrays  (I, N, 2)
    N : int

    Returns
    -------
    F : list of G ndarrays  (I, 2)
    """
    MU, W = _get_quadrature(N)
    F = []
    for psi in psi_g:
        # cell-centre value = mean of the two LD edge values
        psi_c = 0.5 * (psi[:, :, 0] + psi[:, :, 1])      # (I, N)
        F.append(2.0 * (psi_c * (W * MU)[None, :]).sum(axis=1))  # (I,)
    return F


# ---------------------------------------------------------------------------
# Multigroup spherical time-dependent solver
# ---------------------------------------------------------------------------

def mg_temp_solve_sph_ld(
    # geometry
    I, r_left, dr,
    # groups
    G,
    sigma_a_funcs,      # list of G callables  T → (I, 2)
    scat_funcs,         # list of G callables  T → (I, 2)
    Bg_funcs,           # list of G callables  T → (I, 2)   (= 4π B_g)
    dBdT_funcs,         # list of G callables  T → (I, 2)   (= 4π dB_g/dT)
    q_n_ext,            # list of G  (I, N, 2)  fixed external sources (or None)
    q_g_ext,            # list of G  (I, 2)     starting-dir sources   (or None)
    # angular
    N,
    BCs_outer,          # list of G callables  t → (bc_outer (N,2), bc_g_outer float)
    BCs_inner,          # list of G callables  t → (N,2)  or None (vacuum / full sphere)
    # material
    EOS, invEOS,        # e(T), T(e)  node-wise on (I, 2)
    Cv_func,            # callable  T → (I, 2)
    # initial state
    phi_g, psi_g, g_g, T,
    # time stepping
    dt_min=1e-5, dt_max=0.001, tfinal=1.0,
    # convergence
    Linf_tol=1e-8, tolerance=1e-8, maxits=200,
    LOUD=False, fix=1, K=100, R=3,
    time_outputs=None,
    print_stride=0,
    dt_T_frac=0.0,
):
    r"""Time-dependent multigroup TRT in spherical geometry (LD-S_N + DMD).

    Groups are coupled only through the shared material temperature via the
    Fleck-factor emission redistribution (no direct group-to-group transfer).
    Linearisation is performed at ``T_old`` each step (single linearisation,
    matching the slab multigroup solver).

    Parameters
    ----------
    I : int
        Number of radial cells.
    r_left : (I,)  float64
        Left-edge radii r_{j−1/2}.
    dr : (I,)  float64
        Cell widths Δr_j.
    G : int
        Number of energy groups.
    sigma_a_funcs, scat_funcs : list of G callables  ``T → (I, 2)``
        Absorption / scattering opacity per group.
    Bg_funcs, dBdT_funcs : list of G callables  ``T → (I, 2)``
        ``4π B_g(T)`` and ``4π dB_g/dT``; ``Σ_g B_g = a c T⁴``.
    q_n_ext : list of G ``(I, N, 2)`` or None
        Fixed external ordinate source.
    q_g_ext : list of G ``(I, 2)`` or None
        Fixed external starting-direction source.
    N : int
        Number of discrete ordinates (S_N order).
    BCs_outer : list of G callables  ``t → (bc_outer (N,2), bc_g_outer float)``
        Outer-boundary inflow.  ``bc_outer[n, 0]`` is the inflow for μ_n < 0;
        ``bc_g_outer`` is the starting-direction (μ = −1) inflow.
    BCs_inner : list of G callables  ``t → (N, 2)`` or None
        Inner-wall inflow for μ_n > 0 (``bc[n, 1]``).  Pass ``None`` for a
        vacuum inner wall or a full sphere.
    EOS, invEOS : callables  ``e(T)`` and ``T(e)`` on ``(I, 2)``.
    Cv_func : callable  ``T → (I, 2)``.
    phi_g : list of G ``(I, 2)``   Initial group scalar fluxes.
    psi_g : list of G ``(I, N, 2)``  Initial group angular fluxes.
    g_g   : list of G ``(I, 2)``   Initial starting-direction intensities.
    T : ``(I, 2)``  Initial temperature.
    dt_min, dt_max : float  Adaptive time-step bounds.
    tfinal : float  Final time.
    Linf_tol, tolerance : float  Convergence tolerances.
    maxits : int  Max DMD iterations per step.
    fix : int  Positivity fix-up flag (default 1 = on).
    K, R : int  DMD parameters.
    time_outputs : ndarray or None
        Times at which to store full snapshots (also used to snap dt).
    print_stride : int  Print per-step diagnostics every this many steps.
    dt_T_frac : float
        If > 0, throttle the adaptive dt so the mean material-temperature
        change per step stays near ``dt_T_frac`` of the peak temperature.
        ``0`` disables throttling (use pure 2nd-derivative control).

    Returns
    -------
    snapshots : list of dict
        One entry per requested output time, each with keys
        ``time``, ``phi_g`` (list of (I,2)), ``psi_g`` (list of (I,N,2)),
        ``g_g`` (list of (I,2)), ``T`` ((I,2)).
    iterations : int
        Total transport sweeps.
    ts : ndarray
        All time-step times.
    """
    r_left = np.ascontiguousarray(r_left, dtype=np.float64)
    dr = np.ascontiguousarray(dr, dtype=np.float64)
    _full_sphere = (r_left[0] < 1e-15)

    if q_n_ext is None:
        q_n_ext = [np.zeros((I, N, 2)) for _ in range(G)]
    if q_g_ext is None:
        q_g_ext = [np.zeros((I, 2)) for _ in range(G)]

    zero_bc_outer = np.zeros((N, 2))
    zero_bc_inner = np.zeros((N, 2))

    t_current = 0.0
    phi_g = [p.copy() for p in phi_g]
    psi_g = [p.copy() for p in psi_g]
    g_g = [gg.copy() for gg in g_g]
    psi_g_old = [p.copy() for p in psi_g]
    g_g_old = [gg.copy() for gg in g_g]

    T_old = T.copy()
    T_old2 = T.copy()
    e_old = EOS(T)

    snapshots = []
    ts = [t_current]
    iterations = 0
    step_num = 0
    dt_old = dt_min
    dt = dt_min
    deriv_val = 0.0
    delta_step = 1e-3
    curr_step = 0
    t_output_index = 0

    if time_outputs is not None:
        time_outputs = np.asarray(time_outputs, dtype=float)

    block = I * 2
    total_unknowns = G * block

    print(f"MG-spherical-LD-S_N: G={G}, I={I}, N={N}")
    print("|", end="")

    while t_current < tfinal:
        dt_old = dt
        step_num += 1

        # ── adaptive time step ────────────────────────────────────────────
        if step_num > 2:
            dt_prop = np.sqrt(delta_step * deriv_val)
            if dt_prop > dt_max:
                dt_prop = dt_max
            if dt_prop < dt_min:
                dt_prop = dt_min
            if dt_prop > 1.5 * dt:
                dt_prop = 1.5 * dt
            dt = dt_prop
        else:
            dt = dt_min
        if (tfinal - t_current) < dt:
            dt = tfinal - t_current
        # snap to next requested output time
        if (time_outputs is not None and t_output_index < time_outputs.size and
                t_current + dt > time_outputs[t_output_index] - 1e-12):
            snap_dt = time_outputs[t_output_index] - t_current
            if snap_dt > 1e-10 * dt_min:
                dt = snap_dt
        if math.isnan(dt) or dt <= 0.0:
            dt = dt_min

        if LOUD:
            print("t = %0.4e, dt = %0.4e" % (t_current, dt))
        t_current += dt
        ts.append(t_current)
        if int(10 * t_current / tfinal) > curr_step:
            curr_step += 1
            print(curr_step, end="", flush=True)

        icdt = 1.0 / (c * dt)

        # ── per-group opacities and Planck at T_old ───────────────────────
        sigma_ag = [sigma_a_funcs[g](T_old) for g in range(G)]
        scat_g   = [scat_funcs[g](T_old)    for g in range(G)]
        Bg_vals  = [Bg_funcs[g](T_old)      for g in range(G)]

        # ── Fleck factor and coupling weights ─────────────────────────────
        f, alpha_g = _compute_fleck_and_alpha(
            T_old, dt, sigma_a_funcs, dBdT_funcs, Cv_func, G)

        # ── Σ_{g'} σ_{a,g'} B_{g'}(T^n) ───────────────────────────────────
        sum_sigma_B = np.zeros_like(T_old)
        for g in range(G):
            sum_sigma_B += sigma_ag[g] * Bg_vals[g]

        # ── fixed source and σ̂ per group ──────────────────────────────────
        sigma_hat_g = []
        src_n_fixed = []
        src_g_fixed = []
        for g in range(G):
            sh = sigma_ag[g] + scat_g[g] + icdt                  # (I, 2)
            emission_fixed = sigma_ag[g] * Bg_vals[g] - alpha_g[g] * sum_sigma_B
            sn = (q_n_ext[g]
                  + 0.5 * emission_fixed[:, None, :]
                  + icdt * psi_g_old[g])                          # (I, N, 2)
            sg = (q_g_ext[g]
                  + 0.5 * emission_fixed
                  + icdt * g_g_old[g])                            # (I, 2)
            sigma_hat_g.append(sh)
            src_n_fixed.append(sn)
            src_g_fixed.append(sg)

        # ── coupled matvec: scattering + emission redistribution ──────────
        def _make_mv(_scat, _alpha, _sigma_a, _sh):
            def mv(x_vec):
                phi_parts = [x_vec[g * block:(g + 1) * block].reshape((I, 2))
                             for g in range(G)]
                sum_sigma_phi = np.zeros((I, 2))
                for gp in range(G):
                    sum_sigma_phi += _sigma_a[gp] * phi_parts[gp]
                result = np.empty_like(x_vec)
                for g in range(G):
                    s_scal = _scat[g] * phi_parts[g] + _alpha[g] * sum_sigma_phi
                    s_n = (0.5 * s_scal)[:, None, :] * np.ones((1, N, 1))
                    s_g = 0.5 * s_scal
                    phi_new = single_sweep_phi_sph_ld(
                        I, r_left, dr,
                        np.ascontiguousarray(s_n),
                        np.ascontiguousarray(s_g),
                        _sh[g], N, zero_bc_outer, 0.0,
                        bc_inner=zero_bc_inner, fix=fix)
                    result[g * block:(g + 1) * block] = phi_new.ravel()
                return result
            return mv

        mv = _make_mv(scat_g, alpha_g, sigma_ag, sigma_hat_g)

        # ── build b = per-group sweep with fixed source + real BCs ────────
        b_vec = np.empty(total_unknowns)
        for g in range(G):
            bc_outer_b, bc_g_outer_b = BCs_outer[g](t_current - 0.5 * dt)
            bc_outer_b = np.ascontiguousarray(bc_outer_b, dtype=np.float64)
            if (not _full_sphere) and BCs_inner is not None:
                bc_inner_b = np.ascontiguousarray(
                    BCs_inner[g](t_current - 0.5 * dt), dtype=np.float64)
            else:
                bc_inner_b = zero_bc_inner
            phi_b = single_sweep_phi_sph_ld(
                I, r_left, dr, src_n_fixed[g], src_g_fixed[g],
                sigma_hat_g[g], N, bc_outer_b, float(bc_g_outer_b),
                bc_inner=bc_inner_b, fix=fix)
            b_vec[g * block:(g + 1) * block] = phi_b.ravel()

        # ── coupled transport solve ───────────────────────────────────────
        x_sol = np.concatenate([p.ravel() for p in phi_g])
        try:
            x_sol, total_its, _c, _cl, _A, _Yp, _Ym = solver_with_dmd_inc(
                matvec=mv, b=b_vec, K=K, max_its=maxits, steady=1,
                x=x_sol, Rits=R, LOUD=LOUD,
                L2_tol=tolerance, Linf_tol=Linf_tol)
            if np.any(~np.isfinite(x_sol)):
                raise np.linalg.LinAlgError("non-finite DMD solution")
        except np.linalg.LinAlgError:
            # Fall back to plain source iteration
            x_sol = np.concatenate([p.ravel() for p in phi_g])
            total_its = 0
            for _ in range(maxits):
                x_new = mv(x_sol) + b_vec
                err = np.max(np.abs(x_new - x_sol) /
                             (np.max(np.abs(x_new)) + 1e-30))
                x_sol = x_new
                total_its += 1
                if err < tolerance:
                    break
        iterations += total_its
        for g in range(G):
            phi_g[g] = x_sol[g * block:(g + 1) * block].reshape((I, 2))

        # ── reconstruct ψ_g and g_g with end-of-step BCs ──────────────────
        sum_sigma_phi = np.zeros_like(T_old)
        for gp in range(G):
            sum_sigma_phi += sigma_ag[gp] * phi_g[gp]
        for g in range(G):
            bc_outer_p, bc_g_outer_p = BCs_outer[g](t_current)
            bc_outer_p = np.ascontiguousarray(bc_outer_p, dtype=np.float64)
            if (not _full_sphere) and BCs_inner is not None:
                bc_inner_p = np.ascontiguousarray(
                    BCs_inner[g](t_current), dtype=np.float64)
            else:
                bc_inner_p = zero_bc_inner
            s_scal = scat_g[g] * phi_g[g] + alpha_g[g] * sum_sigma_phi
            full_n = src_n_fixed[g] + (0.5 * s_scal)[:, None, :] * np.ones((1, N, 1))
            full_g = src_g_fixed[g] + 0.5 * s_scal
            psi_cand, _, g_cand = single_sweep_psi_sph_ld(
                I, r_left, dr,
                np.ascontiguousarray(full_n),
                np.ascontiguousarray(full_g),
                sigma_hat_g[g], N, bc_outer_p, float(bc_g_outer_p),
                bc_inner=bc_inner_p, fix=fix)
            psi_g[g] = psi_cand
            g_g[g] = g_cand

        # ── material energy update ────────────────────────────────────────
        energy_dep = np.zeros_like(T_old)
        for g in range(G):
            energy_dep += sigma_ag[g] * (phi_g[g] - Bg_vals[g])
        e = e_old + f * dt * energy_dep
        T = invEOS(e)

        # ── adaptive dt control ───────────────────────────────────────────
        if dt_T_frac > 0.0:
            dT = np.mean(np.abs(T - T_old))
            T_peak = max(np.max(np.abs(T)), 1e-30)
            target = dt_T_frac * T_peak
            ratio = target / (dT + 1e-30)
            deriv_val = (dt * min(max(ratio, 0.25), 4.0))**2 / delta_step
        elif step_num >= 2:
            denom = np.mean(np.abs(
                T / (dt**2)
                - (dt + dt_old) / (dt**2 * dt_old) * T_old
                + T_old2 / (dt_old * dt)))
            mean_T = np.mean(T)
            if denom > 0.0 and np.isfinite(mean_T) and np.isfinite(denom):
                deriv_val = mean_T / denom
            else:
                deriv_val = dt_max**2 / delta_step

        if print_stride > 0 and step_num % print_stride == 0:
            print(f"\n  step {step_num:5d}  t={t_current:.4e} ns  dt={dt:.3e}  "
                  f"T_max={np.max(T):.4f} keV  its={total_its}", flush=True)

        # ── store snapshot if at an output time ───────────────────────────
        if (time_outputs is not None and t_output_index < time_outputs.size and
                abs(t_current - time_outputs[t_output_index]) <= 1e-9 * max(tfinal, 1.0)):
            snapshots.append({
                'time': float(t_current),
                'phi_g': [p.copy() for p in phi_g],
                'psi_g': [p.copy() for p in psi_g],
                'g_g': [gg.copy() for gg in g_g],
                'T': T.copy(),
            })
            t_output_index += 1

        # ── shift time levels ─────────────────────────────────────────────
        e_old = e.copy()
        T_old2 = T_old.copy()
        T_old = T.copy()
        psi_g_old = [p.copy() for p in psi_g]
        g_g_old = [gg.copy() for gg in g_g]

    print()
    return snapshots, iterations, np.array(ts)
