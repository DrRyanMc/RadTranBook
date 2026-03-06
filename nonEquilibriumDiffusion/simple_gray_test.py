#!/usr/bin/env python3
"""Simple test of gray solver."""

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

solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups, r_min=0.0, r_max=5.0, n_cells=n_cells,
    energy_edges=edges, geometry='planar', dt=0.1,
    diffusion_coeff_funcs=dfunc, absorption_coeff_funcs=sfunc,
    left_bc_funcs=[lambda p,r: (0.0,1.0,0.0) for _ in range(n_groups)],
    right_bc_funcs=[lambda p,r: (0.0,1.0,0.0) for _ in range(n_groups)],
    emission_fractions=np.ones(n_groups)/n_groups, rho=1.0, cv=0.1, quiet=True
)

r = solver.r_centers
T0 = 0.1 + 0.9*np.exp(-r/2.0)
solver.T = T0.copy()
solver.T_old = T0.copy()
solver.E_r = A_RAD*T0**4

# Prepare for kappa solve
print("Setting up...")
solver.update_absorption_coefficients(T0)
print("Creating preconditioner (this creates gray solver)...")
C_op = solver.create_lmfg_preconditioner(T0, verbose=True)
print("Done!")

# Try a simple matrix-vector product
print("\nTesting C·e_0...")
e0 = np.zeros(n_cells)
e0[0] = 1.0
Ce0 = C_op.matvec(e0)
print(f"Success! C·e_0[0] = {Ce0[0]:.6f}")
