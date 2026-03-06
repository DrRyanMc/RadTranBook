#!/usr/bin/env python3
"""Analyze spectral properties of B and CB to understand preconditioner effectiveness."""

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D
from diffusion_operator_solver import A_RAD
from scipy.sparse.linalg import LinearOperator
import matplotlib.pyplot as plt

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

# Build RHS and operators
xi_g_list = []
for g in range(n_groups):
    xi = solver.compute_xi_g(g, T0)
    xi_g_list.append(xi)

rhs = solver.compute_rhs_for_kappa(T0, xi_g_list)

def B_matvec(kappa_vec):
    return solver.apply_operator_B(kappa_vec, T0, xi_g_list)

B_op = LinearOperator((n_cells, n_cells), matvec=B_matvec)

# Build preconditioner
C_op = solver.create_lmfg_preconditioner(T0, verbose=False)

# Sample B and C matrices for small problems
print("Sampling matrix entries...")
B_samples = []
C_samples = []
CB_samples = []

for i in range(n_cells):
    ei = np.zeros(n_cells)
    ei[i] = 1.0
    B_samples.append(B_op.matvec(ei))
    C_samples.append(C_op.matvec(ei))
    CB_samples.append(C_op.matvec(B_op.matvec(ei)))

B_mat = np.array(B_samples).T
C_mat = np.array(C_samples).T
CB_mat = np.array(CB_samples).T

# Compute condition numbers
print(f"B condition number: {np.linalg.cond(B_mat):.3e}")
print(f"CB condition number: {np.linalg.cond(CB_mat):.3e}")
print(f"Ratio CB/B: {np.linalg.cond(CB_mat) / np.linalg.cond(B_mat):.3f}")
print()

# Check if C and B are nearly proportional
print("Checking C vs B proportionality:")
print(f"  ||C - aB|| for best scalar a:")
# Find best a to minimize ||C - aB||
a_opt = np.trace(B_mat.T @ C_mat) / np.trace(B_mat.T @ B_mat)
print(f"    a_opt = {a_opt:.6f}")
print(f"    ||C - {a_opt:.3f}·B|| = {np.linalg.norm(C_mat - a_opt*B_mat) / np.linalg.norm(C_mat):.3e}")
print()

# Check determinants
print("Operator properties:")
print(f"  det(B) = {np.linalg.det(B_mat):.6e}")
print(f"  det(C) = {np.linalg.det(C_mat):.6e}")
print(f"  det(CB) = {np.linalg.det(CB_mat):.6e}")
print()

print("Diagonal structure:")
print(f"  B diagonal: {np.diag(B_mat)[:5]}")
print(f"  C diagonal: {np.diag(C_mat)[:5]}")
print(f"  CB diagonal: {np.diag(CB_mat)[:5]}")
