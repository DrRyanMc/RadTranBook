#!/usr/bin/env python3
"""
LMFGK Diagnostic: Uniform-T Gray-Limit Iteration Comparison

Runs a single Newton iteration (max_newton_iter=1) twice on the same setup:
  (1) WITHOUT LMFGK preconditioner
  (2) WITH    LMFGK preconditioner

The setup is designed to be as "gray of gray" as possible:
  - All groups identical
  - Constant sigma_a everywhere (no r or T dependence)
  - Uniform initial temperature T_init = constant => Fleck factor tends to be uniform too
  - Neumann (zero-flux) boundaries

It prints:
  - Fleck factor min/max
  - GMRES iterations (no-prec vs prec)
  - GMRES info codes

Edit the import at the top if you've renamed the solver module.
"""

import numpy as np

# ---- IMPORT YOUR SOLVER MODULE HERE ----
# If you renamed the file, change this to match your module name.
import multigroup_diffusion_solver as mg

from diffusion_operator_solver import A_RAD


def neumann_bc(phi, r):
    # A dphi/dr + B phi = C ; Neumann zero-flux corresponds to A=0,B=1,C=0 in this codebase's convention
    return 0.0, 1.0, 0.0


def build_solver(n_groups=10, n_cells=30, r_min=0.0, r_max=5.0, dt=0.5, geometry="planar"):
    # Energy edges (keV) required by the solver; values don't matter for this gray-limit test.
    energy_edges = np.array([1.00e-04, 3.16e-04, 1.00e-03, 3.16e-03, 1.00e-02,
                             3.16e-02, 1.00e-01, 3.16e-01, 1.00e+00, 3.16e+00, 1.00e+01])

    # Constant gray opacity for all groups everywhere
    SIGMA0 = 10.0  # cm^-1 (pick something moderate)

    def sigma_gray(T, r):
        return SIGMA0

    sigma_funcs = [lambda T, r, g=g: sigma_gray(T, r) for g in range(n_groups)]
    diff_funcs  = [lambda T, r, g=g: 1.0 / (3.0 * sigma_gray(T, r)) for g in range(n_groups)]

    # Equal emission fractions
    chi = np.ones(n_groups) / n_groups

    # Material props (used for Fleck factor)
    rho = 1.0
    cv  = 0.1

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


def reset_state_uniform_T(solver, T0=0.5):
    n = solver.n_cells
    solver.T = T0 * np.ones(n)
    solver.T_old = solver.T.copy()

    solver.E_r = A_RAD * solver.T**4
    solver.E_r_old = solver.E_r.copy()

    solver.kappa = np.zeros(n)
    solver.kappa_old = np.zeros(n)


def main():
    print("USING SOLVER FILE:", mg.__file__)
    print("LMFGK Uniform-T Gray-Limit Iteration Comparison")
    print("=" * 78)

    # Parameters
    n_groups = 10
    n_cells = 30
    dt = 0.5
    T0 = 0.5

    gmres_tol = 1e-10
    gmres_maxiter = 1000   # allow convergence
    max_newton_iter = 1    # one Newton step (linearization check)
    verbose_prec = True    # show pr_norm trace for the preconditioned solve

    # Build solver
    solver = build_solver(n_groups=n_groups, n_cells=n_cells, dt=dt)

    # ------------------- WITHOUT preconditioner -------------------
    reset_state_uniform_T(solver, T0=T0)

    info1 = solver.step(
        max_newton_iter=max_newton_iter,
        newton_tol=1e-12,
        gmres_tol=gmres_tol,
        gmres_maxiter=gmres_maxiter,
        use_preconditioner=False,
        verbose=False
    )

    f1 = getattr(solver, "fleck_factor", None)
    if f1 is None:
        print("WARNING: solver.fleck_factor not found after no-prec run.")
        fmin1 = fmax1 = float("nan")
    else:
        fmin1 = float(np.min(f1))
        fmax1 = float(np.max(f1))

    it_no = info1["gmres_info"].get("iterations", None)
    info_code_no = info1["gmres_info"].get("info", None)

    # ------------------- WITH preconditioner -------------------
    reset_state_uniform_T(solver, T0=T0)

    info2 = solver.step(
        max_newton_iter=max_newton_iter,
        newton_tol=1e-12,
        gmres_tol=gmres_tol,
        gmres_maxiter=gmres_maxiter,
        use_preconditioner=True,
        verbose=verbose_prec
    )

    f2 = getattr(solver, "fleck_factor", None)
    if f2 is None:
        print("WARNING: solver.fleck_factor not found after prec run.")
        fmin2 = fmax2 = float("nan")
    else:
        fmin2 = float(np.min(f2))
        fmax2 = float(np.max(f2))

    it_pre = info2["gmres_info"].get("iterations", None)
    info_code_pre = info2["gmres_info"].get("info", None)

    # ------------------- Report -------------------
    print("\n" + "=" * 78)
    print("RESULTS (Uniform-T, constant sigma, gray-limit)")
    print("=" * 78)
    print(f"dt = {dt} ns, T0 = {T0} keV, groups = {n_groups}, cells = {n_cells}")
    print(f"Fleck factor (no-prec run): min/max = {fmin1:.6e} / {fmax1:.6e}")
    print(f"Fleck factor (prec run):    min/max = {fmin2:.6e} / {fmax2:.6e}")
    print()
    print(f"GMRES (no preconditioner): iterations = {it_no}, info = {info_code_no}")
    print(f"GMRES (LMFGK precondition): iterations = {it_pre}, info = {info_code_pre}")
    if isinstance(it_no, int) and it_no > 0 and isinstance(it_pre, int):
        reduction = 100.0 * (it_no - it_pre) / it_no
        print(f"Iteration reduction: {reduction:.1f}%")
    print()
    print("Note: info==0 means converged. If info>0, GMRES hit a limit and did not fully converge.")
    print("=" * 78)


if __name__ == "__main__":
    main()
