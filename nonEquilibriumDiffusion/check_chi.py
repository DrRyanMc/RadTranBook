#!/usr/bin/env python3
"""Check emission fractions."""

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

solver.update_absorption_coefficients(T0)
chi_local = solver.compute_local_emission_fractions(T0)

print("Emission fractions χ_g at mid-domain (r[15]):")
print()
print("  g  |  χ_g[15]")
print("-" * 20)
for g in range(n_groups):
    print(f"  {g}  |  {chi_local[g, 15]:.6f}")

print()
print(f"Sum of χ_g = {np.sum(chi_local[:, 15]):.6f}")
print()
print("Are emission fractions uniform across groups?")
min_chi = np.min(chi_local)
max_chi = np.max(chi_local)
print(f"  Min: {min_chi:.3e}, Max: {max_chi:.3e}")
print(f"  Range: {max_chi / max(min_chi, 1e-100):.2f}x")
