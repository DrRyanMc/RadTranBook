#!/usr/bin/env python3
"""Diagnose preconditioner effectiveness -- check if C is actually helping."""

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
solver.T = T0.copy()
solver.T_old = T0.copy()
solver.E_r = A_RAD*T0**4
solver.E_r_old = solver.E_r.copy()
solver.kappa = np.zeros(n_cells)
solver.kappa_old = np.zeros(n_cells)

# First compute RHS and B without preconditioner
xi_g_list = []
for g in range(n_groups):
    xi = solver.compute_xi_g(g, T0)
    xi_g_list.append(xi)

rhs = solver.compute_rhs_for_kappa(T0, xi_g_list)
print(f"RHS norm: {np.linalg.norm(rhs):.3e}")

# Now test the preconditioner
print("\nCreating preconditioner...")
C_op = solver.create_lmfg_preconditioner(T0, verbose=True)

print("\nTesting preconditioner effectiveness:")
print("-" * 70)

# Create B operator for comparison
from scipy.sparse.linalg import LinearOperator

def B_matvec(kappa_vec):
    return solver.apply_operator_B(kappa_vec, T0, xi_g_list)

B_op = LinearOperator((n_cells, n_cells), matvec=B_matvec)

# Test: CB vs B  diagonal
print("\nDiagonal of B vs |C·rhs|/||rhs||:")
x_test = np.zeros(n_cells)
x_test[0] = 1.0
Bx = B_op.matvec(x_test)
Cx = C_op.matvec(x_test)
CBx = C_op.matvec(B_op.matvec(x_test))

print(f"  B[0,0] = {Bx[0]:.6e}")
print(f"  C[0,0] = {Cx[0]:.6e}")
print(f"  (C·B)[0,0] = {CBx[0]:.6e}")

print("\nCondition numbers (estimated):")
# Estimate condition number via power method
def estimate_spectral_radius(A_op, n_iter=20):
    v = np.random.randn(n_cells)
    for _ in range(n_iter):
        v = A_op.matvec(v)
        v /= np.linalg.norm(v)
    lam_max = np.dot(v, A_op.matvec(v))
    return lam_max

try:
    rho_B = estimate_spectral_radius(B_op, n_iter=5)
    print(f"  Spectral radius of B: {rho_B:.6f}")
except:
    print("  (Could not estimate condition number)")

print("\nTest complete!")
