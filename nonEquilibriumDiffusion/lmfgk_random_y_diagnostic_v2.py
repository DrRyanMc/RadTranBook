#!/usr/bin/env python3
"""
LMFGK gray-solve diagnostic (random y) — v2

Key fix vs v1:
  DiffusionOperatorSolver1D.apply_operator() does NOT apply boundary conditions;
  solve() DOES apply boundary conditions. So comparing apply_operator(U) to rhs
  is inconsistent for Neumann/Robin BCs.

This script applies BCs to the assembled matrix before forming H(phi) so the
operator used in the check matches the operator used in the solve.

Usage:
  python lmfgk_random_y_diagnostic_v2.py

Edit the solver import below if you've renamed the module.
"""

import numpy as np

import multigroup_diffusion_solver as mg
from diffusion_operator_solver import DiffusionOperatorSolver1D, C_LIGHT


def neumann_bc(phi, r):
    # Robin form: A*phi + B*dphi/dn = C. Neumann zero-gradient => A=0, B=1, C=0.
    return 0.0, 1.0, 0.0


def apply_operator_with_bcs(op: DiffusionOperatorSolver1D, phi: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    Compute (A_bc) * phi where A_bc is the assembled matrix AFTER applying BC modifications,
    matching what solve() uses internally.
    """
    A, D_faces, _ = op.assemble_matrix(T, phi)
    rhs_dummy = np.zeros_like(phi)
    A_bc = op.apply_boundary_conditions(A, rhs_dummy, phi, T, D_faces)
    return A_bc.dot(phi)


def main():
    np.random.seed(0)

    n_groups = 10
    n_cells = 30

    # IMPORTANT: keep r_min slightly > 0 to avoid any accidental 1/r terms if geometry branches leak
    r_min, r_max = 1.0e-6, 5.0
    dt = 0.5
    geometry = "planar"

    energy_edges = np.array([1.00e-04, 3.16e-04, 1.00e-03, 3.16e-03, 1.00e-02,
                             3.16e-02, 1.00e-01, 3.16e-01, 1.00e+00, 3.16e+00, 1.00e+01])

    # Constant opacity everywhere (true gray-frequency limit)
    def sigma_gray(T, r):
        return 10.0

    sigma_funcs = [lambda T, r, g=g: sigma_gray(T, r) for g in range(n_groups)]
    diff_funcs  = [lambda T, r, g=g: 1.0 / (3.0 * sigma_gray(T, r)) for g in range(n_groups)]
    chi = np.ones(n_groups) / n_groups

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

    # Uniform temperature
    T0 = 0.5
    solver.T = T0 * np.ones(n_cells)
    solver.T_old = solver.T.copy()

    # Fleck factor
    if hasattr(solver, "compute_fleck_factor"):
        solver.fleck_factor = solver.compute_fleck_factor(solver.T_old)

    f = getattr(solver, "fleck_factor", None)
    if f is None:
        raise RuntimeError("solver.fleck_factor is not set. Run one solver.step() first or ensure fleck_factor is computed.")

    print("USING SOLVER FILE:", mg.__file__)
    print("n_cells =", n_cells, "dt =", dt, "c =", C_LIGHT)
    print("Fleck factor f: min/max =", float(np.min(f)), "/", float(np.max(f)))

    # Linearization temperature
    T_star = solver.T_old.copy()

    # Build the same gray coefficients as LMFGK uses
    lambda_tilde = solver.compute_gray_weights(T_star, verbose=False)
    sigma_a_gray, D_gray = solver.compute_gray_operator_coefficients(T_star, lambda_tilde, verbose=False)

    r_centers = solver.solvers[0].r_centers

    def gray_diffusion_coeff(T, r):
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
        # Return ONLY <sigma_a>(1-f). DiffusionOperatorSolver1D itself adds 1/(c*dt).
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

    # Homogeneous BCs (C=0) to match LMFGK gray solve
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
    # ---- BC sanity: what does the solver actually return? ----
    print("\nBC triples returned by gray_solver:")
    print("  left :", gray_solver.left_bc_func(1.0, gray_solver.r_faces[0]))
    print("  right:", gray_solver.right_bc_func(1.0, gray_solver.r_faces[-1]))

    # ---- Matrix magnitude sanity ----
    A_raw, D_faces, _ = gray_solver.assemble_matrix(T_star, np.ones(n_cells))
    rhs0 = np.zeros(n_cells)
    A_bc = gray_solver.apply_boundary_conditions(A_raw, rhs0, np.ones(n_cells), T_star, D_faces)

    print("  max|A_raw| =", np.max(np.abs(A_raw.data)))
    print("  max|A_bc|  =", np.max(np.abs(A_bc.data)))
    print("  max|rhs0|  =", np.max(np.abs(rhs0)))

    print("gray_solver.geometry =", gray_solver.geometry)
    print("first face r =", gray_solver.r_faces[0], "first center r =", gray_solver.r_centers[0])
    
    A_raw, D_faces, _ = gray_solver.assemble_matrix(T_star, np.ones(n_cells))
    imax = np.argmax(np.abs(A_raw.data))
    print("max entry:", A_raw.data[imax])
    print("row/col of max:", A_raw.nonzero()[0][imax], A_raw.nonzero()[1][imax])
    print("min/max D_faces:", np.min(D_faces), np.max(D_faces))
    # ---------------------------
    # Diagnostic: random y
    # ---------------------------
    y = np.random.randn(n_cells)
    if np.allclose(f, 1.0):
        print("WARNING: f is identically 1. Using synthetic (1-f)=0.5 for diagnostic.")
        one_minus_f = 0.5 * np.ones_like(f)
    else:
        one_minus_f = 1.0 - f

    rhs = one_minus_f * y

    U = gray_solver.solve(rhs, T_star)
    HU = apply_operator_with_bcs(gray_solver, U, T_star)

    # Sanity: finite values?
    if not (np.isfinite(U).all() and np.isfinite(HU).all() and np.isfinite(rhs).all()):
        print("\nWARNING: Non-finite values detected.")
        print("  finite(U)? ", np.isfinite(U).all())
        print("  finite(HU)?", np.isfinite(HU).all())
        print("  finite(rhs)?", np.isfinite(rhs).all())

    rel_err = np.linalg.norm(HU - rhs) / (np.linalg.norm(rhs) + 1e-30)
    print("\nGray-solve diagnostic (random y), BC-consistent:")
    print("  ||H(U) - rhs|| / ||rhs|| =", f"{rel_err:.3e}")
    print("  max|H(U) - rhs|         =", f"{np.max(np.abs(HU - rhs)):.3e}")

    # ---------------------------
    # Constant-vector check: H(1) = reaction*1 for homogeneous Neumann (diffusion term vanishes)
    # ---------------------------
    ones = np.ones(n_cells)
    H1 = apply_operator_with_bcs(gray_solver, ones, T_star)

    inv_c_dt = 1.0 / (C_LIGHT * solver.dt)
    reaction = sigma_a_gray * (1.0 - f) + inv_c_dt

    react_err = np.linalg.norm(H1 - reaction) / (np.linalg.norm(reaction) + 1e-30)
    print("\nConstant-vector check, BC-consistent:")
    print("  ||H(1) - (sigma(1-f)+1/cdt)|| / ||...|| =", f"{react_err:.3e}")

    # Print extrema to help debugging if needed
    print("  min/max H(1):", float(np.min(H1)), "/", float(np.max(H1)))
    print("  min/max reaction:", float(np.min(reaction)), "/", float(np.max(reaction)))


if __name__ == "__main__":
    main()
