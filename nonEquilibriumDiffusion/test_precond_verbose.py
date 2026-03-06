#!/usr/bin/env python3
"""Test preconditioner with verbose output."""

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D
from diffusion_operator_solver import A_RAD

n_groups = 10
n_cells = 30
edges = np.array([0.0001,0.000316,0.001,0.00316,0.01,0.0316,0.1,0.316,1.0,3.16,10.0])

def sigma(T, r, e0, e1):
    hm = 2.0 * e0 * e1 / (e0 + e1)
    base = 10.0 if r < 2.5 else 100.0
    return base / np.sqrt(T) * np.power(hm, -3.0)

sfunc = [lambda T,r,e0=edges[g],e1=edges[g+1]: sigma(T,r,e0,e1) for g in range(n_groups)]
dfunc = [lambda T,r,e0=edges[g],e1=edges[g+1]: 1.0/(3.0*sigma(T,r,e0,e1)) for g in range(n_groups)]

def bc(phi,r):
    return (0.0,1.0,0.0)

solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups, r_min=0.0, r_max=5.0, n_cells=n_cells,
    energy_edges=edges, geometry='planar', dt=0.1,
    diffusion_coeff_funcs=dfunc, absorption_coeff_funcs=sfunc,
    left_bc_funcs=[bc]*n_groups, right_bc_funcs=[bc]*n_groups,
    emission_fractions=np.ones(n_groups)/n_groups, rho=1.0, cv=0.1, quiet=True
)

r = solver.r_centers
T0 = 0.1 + 0.9*np.exp(-r/2.0)

# Check preconditioner with verbose=True
print("=" * 70)
print("WITHOUT PRECONDITIONER")
print("=" * 70)
solver.T = T0.copy()
solver.T_old = T0.copy()
solver.E_r = A_RAD*T0**4
solver.E_r_old = solver.E_r.copy()
solver.kappa = np.zeros(n_cells)
solver.kappa_old = np.zeros(n_cells)

info = solver.step(max_newton_iter=1, newton_tol=1e-8, gmres_tol=1e-8,
                  gmres_maxiter=500, use_preconditioner=False, verbose=True)
gi = info['gmres_info']
print(f"\n==> Iters: {gi['iterations']}, True residual: {gi['final_true_resid']:.3e}\n")

print("\n" + "=" * 70)
print("WITH PRECONDITIONER (verbose)")
print("=" * 70)
solver.T = T0.copy()
solver.T_old = T0.copy()
solver.E_r = A_RAD*T0**4
solver.E_r_old = solver.E_r.copy()
solver.kappa = np.zeros(n_cells)
solver.kappa_old = np.zeros(n_cells)

info = solver.step(max_newton_iter=1, newton_tol=1e-8, gmres_tol=1e-8,
                  gmres_maxiter=500, use_preconditioner=True, verbose=True)
gi = info['gmres_info']
print(f"\n==> Iters: {gi['iterations']}, True residual: {gi['final_true_resid']:.3e}\n")
