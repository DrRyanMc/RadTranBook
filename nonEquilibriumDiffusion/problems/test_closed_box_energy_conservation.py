#!/usr/bin/env python3
"""
Simple closed-box conservation test for MultigroupDiffusionSolver1D.

Goal:
- Verify whether total energy is conserved when Newton iterations converge.
- Closed system: 1 cell, reflecting boundaries, no external sources.

If the formulation is exactly conservative, relative drift in
E_total = E_rad + E_mat should remain near roundoff after converged steps.
"""

import os
import sys
import numpy as np

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
NON_EQ_ROOT = os.path.dirname(THIS_DIR)
sys.path.insert(0, NON_EQ_ROOT)

from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD


def run_closed_box_case(
    n_groups=1,
    n_steps=20,
    dt=1.0e-4,
    sigma_a=1.0,
    rho_cv=0.01,
    t_mat0=0.4,
    t_rad0=0.5,
    newton_tol=1e-12,
    gmres_tol=1e-12,
    max_newton_iter=80,
    gmres_maxiter=2000,
):
    if n_groups == 1:
        energy_edges = np.array([1.0e-4, 40.0], dtype=float)
    else:
        energy_edges = np.logspace(np.log10(1.0e-4), np.log10(40.0), n_groups + 1)

    def sigma_func(T, r):
        return sigma_a + 0.0 * T

    def diff_func(T, r):
        return C_LIGHT / (3.0 * sigma_a)

    zero_flux_bc = lambda phi, r: (0.0, 1.0, 0.0)

    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=0.0,
        r_max=1.0,
        n_cells=1,
        energy_edges=energy_edges,
        geometry="planar",
        dt=dt,
        diffusion_coeff_funcs=[diff_func] * n_groups,
        absorption_coeff_funcs=[sigma_func] * n_groups,
        left_bc_funcs=[zero_flux_bc] * n_groups,
        right_bc_funcs=[zero_flux_bc] * n_groups,
        rho=1.0,
        cv=rho_cv,
    )

    e_rad0 = A_RAD * t_rad0**4
    solver.T = np.array([t_mat0], dtype=float)
    solver.T_old = solver.T.copy()
    solver.E_r = np.array([e_rad0], dtype=float)
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(1, dtype=float)
    solver.kappa_old = np.zeros(1, dtype=float)
    solver.phi_g_fraction[:, 0] = 1.0 / n_groups
    solver.phi_g_stored[:, 0] = (solver.E_r[0] * C_LIGHT) / n_groups

    e_total0 = float(rho_cv * solver.T[0] + solver.E_r[0])
    rel_drifts = [0.0]
    converged_all = True

    for step in range(1, n_steps + 1):
        info = solver.step(
            max_newton_iter=max_newton_iter,
            newton_tol=newton_tol,
            gmres_tol=gmres_tol,
            gmres_maxiter=gmres_maxiter,
            use_preconditioner=True,
            max_relative_change=1.0,
            verbose=False,
        )
        solver.advance_time()

        if not info["converged"]:
            converged_all = False

        e_total = float(rho_cv * solver.T[0] + solver.E_r[0])
        rel_drift = (e_total - e_total0) / max(e_total0, 1e-300)
        rel_drifts.append(rel_drift)

    rel_drifts = np.array(rel_drifts)
    return {
        "n_groups": n_groups,
        "converged_all": converged_all,
        "rel_final": float(rel_drifts[-1]),
        "rel_min": float(np.min(rel_drifts)),
        "rel_max": float(np.max(rel_drifts)),
    }


def main():
    print("=" * 88)
    print("Closed-Box Energy Conservation Test")
    print("=" * 88)

    cases = [1, 12]
    results = []
    for ng in cases:
        res = run_closed_box_case(n_groups=ng)
        results.append(res)
        print(
            f"groups={res['n_groups']:>2d}  converged_all={res['converged_all']}  "
            f"rel_final={res['rel_final']:.6e}  rel_min={res['rel_min']:.6e}  rel_max={res['rel_max']:.6e}"
        )

    tol = 1e-8
    ok = all(r["converged_all"] and abs(r["rel_final"]) < tol and abs(r["rel_min"]) < tol for r in results)

    if ok:
        print("PASS: Closed-box energy conserved within tolerance after converged Newton solves.")
        raise SystemExit(0)

    print("FAIL: Closed-box energy drift detected despite converged Newton solves.")
    raise SystemExit(1)


if __name__ == "__main__":
    main()
