#!/usr/bin/env python3
"""
LMFGK gray-solve diagnostic (random y)

This script performs the following check:

  Pick a random vector y (size n_cells).
  Form rhs_gray = (1 - f) * y.
  Solve the gray correction equation used by LMFGK:
      H U = rhs_gray
  Then verify (sanity check) that applying H to U reproduces rhs_gray:
      ||H(U) - rhs_gray|| / ||rhs_gray||  is small.

This directly checks the internal *gray operator H* build is consistent with
the gray solver's solve/apply_operator methods.

Requirements:
  - multigroup_diffusion_solver_patched_lmfgk.py (or your renamed equivalent)
  - diffusion_operator_solver.py

Usage:
  python lmfgk_random_y_diagnostic.py
"""

import numpy as np

# ---- Import your solver module here ----
# If you've renamed the file, change this import accordingly.
import multigroup_diffusion_solver_patched_lmfgk as mg
from diffusion_operator_solver import DiffusionOperatorSolver1D, C_LIGHT


def neumann_bc(phi, r):
    # A dphi/dr + B phi = C ; Neumann zero-flux corresponds to A=1, B=0, C=0 in this codebase
    return 0.0, 1.0, 0.0


def main():
    np.random.seed(0)

    # ---------------------------
    # Build a simple gray-limit problem
    # ---------------------------
    n_groups = 10
    n_cells = 30
    r_min, r_max = 0.0, 5.0
    dt = 0.5
    geometry = "planar"

    # Energy edges are required but irrelevant for this diagnostic
    energy_edges = np.array([1.00e-04, 3.16e-04, 1.00e-03, 3.16e-03, 1.00e-02,
                             3.16e-02, 1.00e-01, 3.16e-01, 1.00e+00, 3.16e+00, 1.00e+01])

    # Constant opacity for all groups everywhere (true gray-frequency limit)
    def sigma_gray(T, r):
        return 10.0

    sigma_funcs = [lambda T, r, g=g: sigma_gray(T, r) for g in range(n_groups)]
    diff_funcs  = [lambda T, r, g=g: 1.0 / (3.0 * sigma_gray(T, r)) for g in range(n_groups)]

    chi = np.ones(n_groups) / n_groups

    # Material properties (only matter insofar as they affect f via Fleck factor)
    rho = 1.0
    cv = 0.1

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

    # Choose a uniform temperature to keep coefficients smooth
    T0 = 0.5
    solver.T = T0 * np.ones(n_cells)
    solver.T_old = solver.T.copy()

    # Force any internal coefficient updates the solver does at start-of-step.
    # The Fleck factor is computed inside solver.step(), but we can trigger it by taking a cheap step
    # with max_newton_iter=0 if supported; otherwise just run one step with no changes.
    #
    # We'll do the robust option: call the solver's Fleck-factor computation if it exists,
    # else run a single setup step.
    if hasattr(solver, "compute_fleck_factor"):
        solver.fleck_factor = solver.compute_fleck_factor(solver.T_old)
    else:
        # Fall back: do one Newton iteration but skip solving by setting gmres_maxiter=0 if code allows.
        # If that fails in your branch, just comment this out and ensure fleck_factor is set elsewhere.
        pass

    # If fleck_factor wasn't set, it will likely be present from __init__/previous calls.
    f = getattr(solver, "fleck_factor", None)
    if f is None:
        raise RuntimeError("solver.fleck_factor is not set. Run one solver.step() first or ensure fleck_factor is computed.")
    
    print("USING SOLVER FILE:", mg.__file__)
    print("n_cells =", n_cells, "dt =", dt, "c =", C_LIGHT)
    print("Fleck factor f: min/max =", float(np.min(f)), "/", float(np.max(f)))

    # Use T_star at the linearization point
    T_star = solver.T_old.copy()

    # ---------------------------
    # Rebuild the *same* gray operator H used by LMFGK
    # ---------------------------
    lambda_tilde = solver.compute_gray_weights(T_star, verbose=False)
    sigma_a_gray, D_gray = solver.compute_gray_operator_coefficients(T_star, lambda_tilde, verbose=False)

    r_centers = solver.solvers[0].r_centers

    def gray_diffusion_coeff(T, r):
        # same logic as create_lmfg_preconditioner (linear interp on centers)
        if r <= r_centers[0]:
            return float(D_gray[0])
        if r >= r_centers[-1]:
            return float(D_gray[-1])
        for i in range(len(r_centers) - 1):
            if r_centers[i] <= r <= r_centers[i + 1]:
                alpha = (r - r_centers[i]) / (r_centers[i + 1] - r_centers[i])
                return float((1 - alpha) * D_gray[i] + alpha * D_gray[i + 1])
        return float(D_gray[0])

    def gray_absorption_coeff(T, r):
        # IMPORTANT: DiffusionOperatorSolver1D adds 1/(c dt) internally.
        # So this function should return only <sigma_a>(1-f).
        if r <= r_centers[0]:
            idx = 0
            return float(sigma_a_gray[idx] * (1.0 - f[idx]))
        if r >= r_centers[-1]:
            idx = len(r_centers) - 1
            return float(sigma_a_gray[idx] * (1.0 - f[idx]))
        for i in range(len(r_centers) - 1):
            if r_centers[i] <= r <= r_centers[i + 1]:
                alpha = (r - r_centers[i]) / (r_centers[i + 1] - r_centers[i])
                sigma_interp = (1 - alpha) * sigma_a_gray[i] + alpha * sigma_a_gray[i + 1]
                f_interp     = (1 - alpha) * f[i]           + alpha * f[i + 1]
                return float(sigma_interp * (1.0 - f_interp))
        idx = 0
        return float(sigma_a_gray[idx] * (1.0 - f[idx]))

    # Homogeneous BCs matching create_lmfg_preconditioner
    original_left_bc = solver.solvers[0].left_bc_func
    original_right_bc = solver.solvers[0].right_bc_func

    def homogeneous_left_bc(phi, r):
        A, B, C = original_left_bc(phi, r)
        return A, B, 0.0

    def homogeneous_right_bc(phi, r):
        A, B, C = original_right_bc(phi, r)
        return A, B, 0.0

    gray_solver = DiffusionOperatorSolver1D(
        r_min=solver.r_min,
        r_max=solver.r_max,
        n_cells=solver.n_cells,
        geometry=solver.geometry,
        dt=solver.dt,
        diffusion_coeff_func=gray_diffusion_coeff,
        absorption_coeff_func=gray_absorption_coeff,
        left_bc_func=homogeneous_left_bc,
        right_bc_func=homogeneous_right_bc,
    )

    # ---------------------------
    # Diagnostic: random y
    # ---------------------------
    y = np.random.randn(n_cells)
    if np.allclose(f, 1.0):
        print("WARNING: f is identically 1. Using synthetic (1-f)=0.5 for diagnostic.")
        one_minus_f = 0.5 * np.ones_like(f)
    else:
        one_minus_f = 1.0 - f
    rhs_gray = one_minus_f * y

    U = gray_solver.solve(rhs_gray, T_star)

    Hu = gray_solver.apply_operator(U, T_star)

    rel_err = np.linalg.norm(Hu - rhs_gray) / (np.linalg.norm(rhs_gray) + 1e-30)

    print("\nGray-solve diagnostic (random y):")
    print("  ||H(U) - rhs|| / ||rhs|| =", f"{rel_err:.3e}")
    print("  max|H(U) - rhs|         =", f"{np.max(np.abs(Hu - rhs_gray)):.3e}")

    # Suggested pass threshold:
    # For double precision sparse solves, ~1e-10 to 1e-8 is typical depending on conditioning.
    if rel_err < 1e-8:
        print("  PASS: Gray solve is consistent with H operator.")
    else:
        print("  WARNING: Large inconsistency. Check H assembly, BCs, or absorption/time-term handling.")

    # Optional: constant-vector reaction check (H*1)
    ones = np.ones(n_cells)
    H1 = gray_solver.apply_operator(ones, T_star)
    inv_c_dt = 1.0 / (C_LIGHT * solver.dt)
    react = sigma_a_gray * (1.0 - f) + inv_c_dt
    react_err = np.linalg.norm(H1 - react) / (np.linalg.norm(react) + 1e-30)
    print("\nConstant-vector check:")
    print("  ||H(1) - (sigma(1-f)+1/cdt)|| / ||...|| =", f"{react_err:.3e}")
    print("  (Diffusion term should vanish on constant vector.)")


if __name__ == "__main__":
    main()
