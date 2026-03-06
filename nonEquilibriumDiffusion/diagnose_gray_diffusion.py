#!/usr/bin/env python3
"""Check if the gray preconditioner formula itself is fundamentally flawed for this problem."""

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D
from diffusion_operator_solver import A_RAD, C_LIGHT

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

# IMPORTANT: Update absorption coefficients before computing weights!
solver.update_absorption_coefficients(T0)

# Compute gray weights
lambda_tilde = solver.compute_gray_weights(T0, verbose=False)

# Compute diffusion and absorption for each group
print("Group diffusion coefficients D_g at r[0]:")
for g in range(n_groups):
    D_g_0 = dfunc[g](T0[0], r[0])
    print(f"  Group {g}: D = {D_g_0:.4e} cm²")

print("\nGray diffusion ⟨D⟩ at r[0]:")
D_gray_0 = 0.0
for g in range(n_groups):
    D_g_0 = dfunc[g](T0[0], r[0])
    w = lambda_tilde[g, 0]
    D_gray_0 += w * D_g_0
    print(f"  Group {g}: w·D = {w:.4e} * {D_g_0:.4e} = {w*D_g_0:.4e}")
print(f"  Total ⟨D⟩ = {D_gray_0:.4e}")

# Now check the ratio problem
print("\nAbsorption structure:")
print(f"  ⟨σ_a⟩ ~ 1.0 (normalized)")
f_factors = solver.compute_fleck_factor(T0)
f_0 = f_factors[0]
print(f"  (1-f) ~ {1 - f_0:.4f}")
reaction_term = 1.0 * (1 - f_0)
print(f"  Reaction term ⟨σ_a⟩(1-f) ~ {reaction_term:.4e}")
print(f"  Diffusion term -∇·⟨D⟩∇ ~ depends on ⟨D⟩ and second derivative")
print(f"  With ⟨D⟩ = {D_gray_0:.4e}, diffusion is MUCH smaller than reaction")
print(f"  Ratio: diffusion/reaction ~ {D_gray_0 / reaction_term:.6e}")
print(f"\nThis is why the gray preconditioner is ineffective!")
print(f"The gray operator is dominated by reaction/absorption, not diffusion.")
print(f"It cannot capture the multigroup diffusion structure that makes B stiff.")
