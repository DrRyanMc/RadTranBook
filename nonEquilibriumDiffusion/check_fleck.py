#!/usr/bin/env python3
"""Check Fleck factor values."""

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D
from diffusion_operator_solver import A_RAD, C_LIGHT

n_groups = 10  
n_cells = 30
edges = np.array([0.0001, 0.000316, 0.001, 0.00316, 0.01, 0.0316, 0.1, 0.316, 1.0, 3.16, 10.0])

def sigma(T, r, e0, e1):
    hm = 2.0 * e0 * e1 / (e0 + e1)
    base = 10.0 if r < 2.5 else 100.0
    return base / np.sqrt(T) * np.power(hm, -3.0)

sfunc = [lambda T, r, e0=edges[g], e1=edges[g+1]: sigma(T, r, e0, e1) for g in range(n_groups)]
dfunc = [lambda T, r, e0=edges[g], e1=edges[g+1]: 1.0 / (3.0 * sigma(T, r, e0, e1)) for g in range(n_groups)]

solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups, r_min=0.0, r_max=5.0, n_cells=n_cells,
    energy_edges=edges, geometry='planar', dt=0.1,
    diffusion_coeff_funcs=dfunc, absorption_coeff_funcs=sfunc,
    left_bc_funcs=[lambda p, r: (0.0, 1.0, 0.0) for _ in range(n_groups)],
    right_bc_funcs=[lambda p, r: (0.0, 1.0, 0.0) for _ in range(n_groups)],
    emission_fractions=np.ones(n_groups)/n_groups, rho=1.0, cv=0.1, quiet=True
)

r = solver.r_centers
T0 = 0.1 + 0.9 * np.exp(-r / 2.0)
solver.T = T0.copy()
solver.E_r = A_RAD * T0**4

print("Fleck factor (f) and (1-f):")
for i in [0, 10, 20, 29]:
    f = solver.compute_fleck_factor(i)
    print(f"  i={i:2d}: T={T0[i]:.4f} keV,  f={f:.6f},  (1-f)={1-f:.6f}")
print()

print("Absorption coefficients σ_a (cm^-1):")
for g in [0, 5, 9]:
    sigma_vals = sfunc[g](T0, r)
    print(f"  Group {g}: min={sigma_vals.min():.4e}, max={sigma_vals.max():.4e}")
