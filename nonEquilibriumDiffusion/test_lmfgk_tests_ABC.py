#!/usr/bin/env python3
"""
LMFGK Tests A–C (Morel-style LMFGK preconditioner)

This script runs three diagnostics:

A) Weight identity + sanity (requires nu=1-f not ~0):
   - Builds a modest multigroup problem with non-uniform T
   - Computes gray weights lambda_tilde
   - Checks max |sum_g sigma_a,g * lambda_tilde_g - 1|
   - Checks lambda_tilde finiteness, (near) nonnegativity, and reasonable magnitudes

B) nu->0 guard correctness:
   - Builds a problem with *very small dt* so f ~ 1 and nu=1-f ~ 0
   - Calls create_lmfg_preconditioner
   - Checks that C acts like identity: ||C x - x|| / ||x|| is ~ 0

C) Stress test (where LMFGK should help):
   - Many groups with alternating HUGE opacity contrast
   - Runs one Newton step with GMRES with/without preconditioner
   - Reports iteration counts and convergence codes

How to run:
  python test_lmfgk_tests_ABC.py

If you renamed your solver module, set env var SOLVER_MODULE:
  SOLVER_MODULE=multigroup_diffusion_solver python test_lmfgk_tests_ABC.py
"""

import os
import importlib
import numpy as np

# -------------------------
# Import solver module
# -------------------------
MODNAME = os.environ.get("SOLVER_MODULE", "multigroup_diffusion_solver")
mg = importlib.import_module(MODNAME)

from diffusion_operator_solver import A_RAD


def neumann_bc(phi, r):
    # Codebase convention used in your tests: (A,B,C) for A*phi + B*dphi/dn = C
    # Pure Neumann dphi/dn=0 => A=0, B=1, C=0
    return 0.0, 1.0, 0.0


def exp_temperature_profile(r_centers, T_cold=0.1, T_hot=1.0, L=2.0):
    return T_cold + (T_hot - T_cold) * np.exp(-r_centers / L)


def build_solver(n_groups, n_cells, r_min, r_max, dt, geometry,
                 sigma_by_group_func, rho=1.0, cv=0.1, energy_edges=None):
    if energy_edges is None:
        # Generate energy_edges with n_groups+1 elements
        # Use logarithmic spacing from 1e-4 to 10 keV
        energy_edges = np.logspace(-4, 1, n_groups + 1)

    sigma_funcs = []
    diff_funcs = []
    for g in range(n_groups):
        def _sigma(T, r, gg=g):
            return float(sigma_by_group_func(gg, T, r))
        def _diff(T, r, gg=g):
            return float(1.0 / (3.0 * sigma_by_group_func(gg, T, r)))
        sigma_funcs.append(_sigma)
        diff_funcs.append(_diff)

    chi = np.ones(n_groups) / n_groups

    solver = mg.MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=r_min,
        r_max=r_max,
        n_cells=n_cells,
        energy_edges=energy_edges,
        geometry=geometry,
        dt=dt,
        diffusion_coeff_funcs=diff_funcs,
        absorption_coeff_funcs=sigma_funcs,
        left_bc_funcs=[neumann_bc] * n_groups,
        right_bc_funcs=[neumann_bc] * n_groups,
        emission_fractions=chi,
        rho=rho,
        cv=cv
    )
    return solver


def reset_state(solver, T_init):
    n = solver.n_cells
    solver.T = T_init.copy()
    solver.T_old = T_init.copy()
    solver.E_r = A_RAD * solver.T**4
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n)
    solver.kappa_old = np.zeros(n)


def ensure_fleck_factor(solver, T_star):
    # Update absorption coefficients at the current temperature
    solver.update_absorption_coefficients(T_star)
    
    if hasattr(solver, "compute_fleck_factor"):
        solver.fleck_factor = solver.compute_fleck_factor(T_star)
    f = getattr(solver, "fleck_factor", None)
    if f is None:
        raise RuntimeError("Could not obtain solver.fleck_factor. Ensure compute_fleck_factor exists or run solver.step() once.")
    return f


# -------------------------
# TEST A
# -------------------------
def test_A():
    print("\n" + "="*78)
    print("TEST A: Weight identity + sanity")
    print("="*78)

    n_groups = 10
    n_cells = 30
    r_min, r_max = 0.0, 5.0
    dt = 5000.
    geometry = "planar"

    # Moderate, smoothly varying opacity by group to avoid degeneracy,
    # but not extreme enough to break GMRES.
    # sigma_g = base * 10^(g/(G-1)) in [1, 100]
    def sigma_by_group(g, T, r):
        base = 1.0
        top = 100.0
        if n_groups == 1:
            return base
        a = g / (n_groups - 1)
        return base * (top/base)**a

    # Use smaller cv to get reasonable Fleck factor (not f~1)
    solver = build_solver(n_groups, n_cells, r_min, r_max, dt, geometry, sigma_by_group, rho=1.0, cv=0.01)
    print(f"DEBUG: solver.cv = {solver.cv}, solver.dt = {solver.dt}")

    T_init = exp_temperature_profile(solver.r_centers, T_cold=0.1, T_hot=1.0, L=2.0)
    reset_state(solver, T_init)

    T_star = solver.T_old.copy()
    f = ensure_fleck_factor(solver, T_star)
    nu = 1.0 - f

    print(f"DEBUG: T_star range = [{T_star.min():.3f}, {T_star.max():.3f}] keV")
    print(f"DEBUG: f[0] = {f[0]}, f[-1] = {f[-1]}")
    print(f"DEBUG: nu[0] = {nu[0]}, nu[-1] = {nu[-1]}")
    print(f"Fleck factor f min/max: {float(np.min(f)):.6e} / {float(np.max(f)):.6e}")
    print(f"nu=(1-f)   min/max: {float(np.min(nu)):.6e} / {float(np.max(nu)):.6e}")
    if np.max(np.abs(nu)) < 1e-12:
        print("WARNING: nu ~ 0 everywhere; Test A is ill-posed in this regime. Increase dt or adjust cv/T.")
        return False

    # Compute weights
    lambda_tilde = solver.compute_gray_weights(T_star, verbose=False)

    # Handle both numpy array (n_groups, n_cells) and list/tuple of arrays
    if isinstance(lambda_tilde, np.ndarray):
        if lambda_tilde.ndim == 2 and lambda_tilde.shape == (n_groups, n_cells):
            lam_stack = lambda_tilde
        else:
            raise RuntimeError(f"compute_gray_weights returned unexpected array shape: {lambda_tilde.shape}, expected {(n_groups, n_cells)}")
    elif isinstance(lambda_tilde, (list, tuple)):
        if len(lambda_tilde) != n_groups:
            raise RuntimeError(f"compute_gray_weights returned unexpected length: {len(lambda_tilde)}, expected {n_groups}")
        lam_stack = np.vstack([np.asarray(lg).reshape(1, -1) for lg in lambda_tilde])
        if lam_stack.shape != (n_groups, n_cells):
            raise RuntimeError(f"lambda_tilde shape mismatch: got {lam_stack.shape}, expected {(n_groups, n_cells)}")
    else:
        raise RuntimeError(f"compute_gray_weights returned unexpected type: {type(lambda_tilde)}")

    if not np.isfinite(lam_stack).all():
        print("FAIL: lambda_tilde contains NaN/Inf.")
        return False

    lam_min = float(np.min(lam_stack))
    lam_max = float(np.max(lam_stack))
    lam_sum_min = float(np.min(np.sum(lam_stack, axis=0)))
    lam_sum_max = float(np.max(np.sum(lam_stack, axis=0)))

    # Build sigma_a,g at each cell and check identity sum sigma*lambda ~ 1
    sig = np.zeros((n_groups, n_cells))
    for g in range(n_groups):
        for i, r in enumerate(solver.r_centers):
            sig[g, i] = sigma_by_group(g, T_star[i], r)

    sum_sig_lam = np.sum(sig * lam_stack, axis=0)
    max_abs_err = float(np.max(np.abs(sum_sig_lam - 1.0)))
    min_sum = float(np.min(sum_sig_lam))
    max_sum = float(np.max(sum_sig_lam))

    print(f"lambda_tilde min/max: {lam_min:.6e} / {lam_max:.6e}")
    print(f"sum_g lambda_tilde min/max: {lam_sum_min:.6e} / {lam_sum_max:.6e}")
    print(f"sum_g sigma_a,g*lambda_tilde min/max: {min_sum:.6e} / {max_sum:.6e}")
    print(f"max |sum_g sigma_a,g*lambda_tilde - 1| = {max_abs_err:.3e}")

    # Pass/fail heuristics
    ok = True
    if lam_min < -1e-12:
        print("FAIL: lambda_tilde has significantly negative values.")
        ok = False
    if lam_max > 1e6:
        print("FAIL: lambda_tilde has suspiciously huge values (>1e6).")
        ok = False
    if max_abs_err > 1e-8:
        print("FAIL: weight identity max error is too large (>1e-8).")
        ok = False

    print("TEST A:", "PASS" if ok else "FAIL")
    return ok


# -------------------------
# TEST B
# -------------------------
def test_B():
    print("\n" + "="*78)
    print("TEST B: nu->0 guard correctness (C ≈ I)")
    print("="*78)

    n_groups = 10
    n_cells = 30
    r_min, r_max = 0.0, 5.0
    dt = 1e-8  # VERY small to drive f -> 1
    geometry = "planar"

    def sigma_by_group(g, T, r):
        return 10.0  # constant

    solver = build_solver(n_groups, n_cells, r_min, r_max, dt, geometry, sigma_by_group, rho=1.0, cv=1000.)
    T0 = 0.5
    T_init = T0 * np.ones(n_cells)
    reset_state(solver, T_init)

    T_star = solver.T_old.copy()
    f = ensure_fleck_factor(solver, T_star)
    nu = 1.0 - f

    print(f"dt = {dt}, Fleck factor f min/max: {float(np.min(f)):.6e} / {float(np.max(f)):.6e}")
    print(f"nu=(1-f) min/max: {float(np.min(nu)):.6e} / {float(np.max(nu)):.6e}")

    # Create preconditioner (should short-circuit to identity if guard is implemented)
    C = solver.create_lmfg_preconditioner(T_star, verbose=True)

    x = np.random.randn(n_cells)
    Cx = C.matvec(x)
    rel = float(np.linalg.norm(Cx - x) / (np.linalg.norm(x) + 1e-30))

    print(f"||C x - x|| / ||x|| = {rel:.3e}")
    ok = rel < 1e-10  # identity within numerical error
    print("TEST B:", "PASS" if ok else "FAIL (guard may be missing or not triggered)")
    return ok


# -------------------------
# TEST C
# -------------------------
def test_C():
    print("\n" + "="*78)
    print("TEST C: Stress test (alternating opacities) — LMFGK should help")
    print("="*78)

    n_groups = 60
    n_cells = 60
    r_min, r_max = 0.0, 5.0
    dt = 0.01
    geometry = "planar"

    sigma_low = 1e-6
    sigma_high = 1e6

    def sigma_by_group(g, T, r):
        return sigma_low if (g % 2 == 0) else sigma_high

    # Use smaller cv to get non-trivial Fleck factors
    solver = build_solver(n_groups, n_cells, r_min, r_max, dt, geometry, sigma_by_group, rho=1.0, cv=3)

    T_init = exp_temperature_profile(solver.r_centers, T_cold=0.1, T_hot=1.0, L=2.0)



   # after solver constructed and T_init set
    T_star = exp_temperature_profile(solver.r_centers, T_cold=0.1, T_hot=1.0, L=2.0)
    solver.T = T_star.copy()
    solver.T_old = T_star.copy()

    # CRITICAL: make sure sigma_a,g(T_star) is actually evaluated/stored in solver
    if hasattr(solver, "update_absorption_coefficients"):
        solver.update_absorption_coefficients(T_star)
    else:
        raise RuntimeError("Solver missing update_absorption_coefficients(T).")

    # Now recompute fleck factor using the updated sigma_a
    if hasattr(solver, "compute_fleck_factor"):
        solver.fleck_factor = solver.compute_fleck_factor(T_star)

    f = solver.fleck_factor

    # Now weights/gray coeffs are meaningful
    lambda_tilde = solver.compute_gray_weights(T_star, verbose=False)
    sigma_a_gray, D_gray = solver.compute_gray_operator_coefficients(T_star, lambda_tilde, verbose=False)

    # Stack lambda if returned as list
    import numpy as np
    if isinstance(lambda_tilde, (list, tuple)):
        lam = np.vstack([np.asarray(L).reshape(1,-1) for L in lambda_tilde])
    else:
        lam = np.asarray(lambda_tilde)
    print("lambda_tilde shape", lam.shape)
    print("lambda_tilde min/max:", np.min(lam), np.max(lam))
    sum_sig_lam = np.zeros(solver.n_cells)
    for g in range(solver.n_groups):
        # use the same sigma_by_group used by test C
        sigg = sigma_by_group(g, T_star[0], solver.r_centers[0])  # if sigma_by_group depends on r/T, evaluate full grid
        # better full build:
    print("D_gray min/max:", np.min(D_gray), np.max(D_gray))
    # compute full sum sigma*lambda per cell (use your sigma_by_group function across r)
    sig = np.zeros((solver.n_groups, solver.n_cells))
    for g in range(solver.n_groups):
        for i,r in enumerate(solver.r_centers):
            sig[g,i] = sigma_by_group(g, T_star[i], r)
    sum_sig_lam = np.sum(sig * lam, axis=0)
    print("sum sigma*lambda min/max:", np.min(sum_sig_lam), np.max(sum_sig_lam))
    print("max |sum sigma*lambda - 1| = ", np.max(np.abs(sum_sig_lam-1.0)))

    # Run WITHOUT preconditioner
    reset_state(solver, T_init)
    # After reset_state(solver, T_init) and before solver.step(...)
    T_star = solver.T_old.copy()
    solver.update_absorption_coefficients(T_star)

    print("sigma_a group0 min/max:", np.min(solver.sigma_a[0]), np.max(solver.sigma_a[0]))
    print("sigma_a group1 min/max:", np.min(solver.sigma_a[1]), np.max(solver.sigma_a[1]))
    # --- Inspect actual sigma and D seen by the per-group diffusion solvers ---
    i0 = 0
    iM = solver.n_cells // 2

    r0 = float(solver.r_centers[i0])
    rM = float(solver.r_centers[iM])

    T0 = float(T_star[i0])
    TM = float(T_star[iM])

    # sigma_a stored by update_absorption_coefficients
    print("sigma_a group0 min/max:", float(np.min(solver.sigma_a[0])), float(np.max(solver.sigma_a[0])))
    print("sigma_a group1 min/max:", float(np.min(solver.sigma_a[1])), float(np.max(solver.sigma_a[1])))

    # diffusion coeff functions live on each DiffusionOperatorSolver1D in solver.solvers[g]
    D0_r0 = float(solver.solvers[0].diffusion_coeff_func(T0, r0))
    D0_rM = float(solver.solvers[0].diffusion_coeff_func(TM, rM))
    D1_r0 = float(solver.solvers[1].diffusion_coeff_func(T0, r0))
    D1_rM = float(solver.solvers[1].diffusion_coeff_func(TM, rM))

    print("D group0 at r0/rM:", D0_r0, D0_rM)
    print("D group1 at r0/rM:", D1_r0, D1_rM)

    # also check what the absorption coeff funcs return (should match sigma_a arrays)
    sig0_r0 = float(solver.solvers[0].absorption_coeff_func(T0, r0))
    sig0_rM = float(solver.solvers[0].absorption_coeff_func(TM, rM))
    sig1_r0 = float(solver.solvers[1].absorption_coeff_func(T0, r0))
    sig1_rM = float(solver.solvers[1].absorption_coeff_func(TM, rM))

    print("sigma_func g0 at r0/rM:", sig0_r0, sig0_rM)
    print("sigma_func g1 at r0/rM:", sig1_r0, sig1_rM)
    info1 = solver.step(
        max_newton_iter=1,
        newton_tol=1e-12,
        gmres_tol=1e-8,
        gmres_maxiter=2000,
        use_preconditioner=False,
        verbose=False
    )
    it_no = info1["gmres_info"].get("iterations", None)
    code_no = info1["gmres_info"].get("info", None)

    # Run WITH preconditioner
    reset_state(solver, T_init)
    info2 = solver.step(
        max_newton_iter=1,
        newton_tol=1e-12,
        gmres_tol=1e-8,
        gmres_maxiter=2000,
        use_preconditioner=True,
        verbose=False
    )
    it_pre = info2["gmres_info"].get("iterations", None)
    code_pre = info2["gmres_info"].get("info", None)

    # Fleck factor range
    f = getattr(solver, "fleck_factor", None)
    if f is not None:
        print(f"Fleck factor f min/max: {float(np.min(f)):.6e} / {float(np.max(f)):.6e}")

    print(f"GMRES no-prec: iterations={it_no}, info={code_no}")
    print(f"GMRES prec:    iterations={it_pre}, info={code_pre}")

    ok = True
    if code_no != 0 or code_pre != 0:
        print("WARNING: One or both GMRES solves did not report info==0 (may not have converged).")
        # Don't hard-fail; just report.
    if isinstance(it_no, int) and isinstance(it_pre, int) and it_no > 0:
        reduction = 100.0 * (it_no - it_pre) / it_no
        print(f"Iteration reduction: {reduction:.1f}%")
        # Expect some improvement; require at least 10% reduction as a soft check
        if reduction < 10.0:
            print("WARNING: LMFGK improvement is small in this stress test; may indicate a bug or that stiffness isn't multifrequency-dominated.")
    else:
        print("WARNING: iteration counts not available as ints; cannot compute reduction.")

    print("TEST C: COMPLETE (interpret reduction + info codes)")
    return ok


def main():
    print("USING SOLVER MODULE:", MODNAME)
    print("USING SOLVER FILE:  ", getattr(mg, "__file__", "(unknown)"))

    okA = test_A()
    okB = test_B()
    okC = test_C()

    print("\n" + "="*78)
    print("SUMMARY")
    print("="*78)
    print(f"Test A (weights): {'PASS' if okA else 'FAIL'}")
    print(f"Test B (nu->0 guard): {'PASS' if okB else 'FAIL'}")
    print("Test C (stress): see reported iteration counts + info codes + reduction")
    print("="*78)

    # Exit code: fail if A or B fail (C is heuristic)
    import sys
    sys.exit(0 if (okA and okB) else 1)


if __name__ == "__main__":
    main()
