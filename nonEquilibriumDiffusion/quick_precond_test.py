#!/usr/bin/env python3
"""Simple diagnostic of preconditioner."""

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
solver.E_r = A_RAD*T0**4

# Simple test
print("Building preconditioner...")
C_op = solver.create_lmfg_preconditioner(T0, verbose=False)

# Test on a random vector
print("\nTesting preconditioner:")
x = np.ones(n_cells)
Cx = C_op.matvec(x)
print(f"||C·x|| / ||x|| = {np.linalg.norm(Cx) / np.linalg.norm(x):.6f}")

# Check if C ≈ I
I_error = np.linalg.norm(Cx - x) / np.linalg.norm(x)
print(f"||C·x - x|| / ||x|| = {I_error:.6e}")
print(f"  (If ~0, then C ≈ I, which is bad)")
print()

# Check diagonal of C
ei = np.zeros(n_cells)
ei[0] = 1.0
Cei = C_op.matvec(ei)
print(f"C[0,0] = {Cei[0]:.6f}")
print(f"(Should be > 1 for effective preconditioner, ~1 means C ≈ I)")
