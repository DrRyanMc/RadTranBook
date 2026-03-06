#!/usr/bin/env python3
"""Test if preconditioner is actually doing anything."""

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D
from diffusion_operator_solver import A_RAD
from scipy.sparse.linalg import LinearOperator

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
solver.update_absorption_coefficients(T0)
solver.chi_local = solver.compute_local_emission_fractions(T0)
solver.chi = np.mean(solver.chi_local, axis=1)
solver.chi /= (np.sum(solver.chi) + 1e-300)

# Compute fleck factor
f = solver.compute_fleck_factor(T0)

# Build operators
xi_g_list = []
for g in range(n_groups):
    xi = solver.compute_xi_g(g, T0)
    xi_g_list.append(xi)

rhs = solver.compute_rhs_for_kappa(T0, xi_g_list)

def B_matvec(kappa_vec):
    return solver.apply_operator_B(kappa_vec, T0, xi_g_list)

B_op = LinearOperator((n_cells, n_cells), matvec=B_matvec)

# Create preconditioner
C_op = solver.create_lmfg_preconditioner(T0, verbose=False)

print("Testing preconditioner effect on matrix:")
print("-" * 70)

# Sample a few matrix-vector products
n_samples = min(5, n_cells)
print("\nSampling matrix-vector products:")
print("  i   |  ||B·e_i||  |  ||C·B·e_i||  |  Reduction")
print("-" * 55)

for i in range(n_samples):
    ei = np.zeros(n_cells)
    ei[i] = 1.0
    
    Bei = B_op.matvec(ei)
    CBei = C_op.matvec(Bei)
    
    norm_B = np.linalg.norm(Bei)
    norm_CB = np.linalg.norm(CBei)
    
    if norm_B > 1e-15:
        ratio = norm_CB / norm_B
    else:
        ratio = 0
    
    print(f"  {i}   |  {norm_B:.4e}  |  {norm_CB:.4e}  |  {ratio:.4f}")

# Check if C improves the conditioning
print("\nCheck: Is C amplifying or damping vectors?")
x_test = np.ones(n_cells)
Cx_test = C_op.matvec(x_test)
BCx_test = B_op.matvec(Cx_test)

print(f"  ||C·ones||/||ones|| = {np.linalg.norm(Cx_test) / np.linalg.norm(x_test):.6f}")
print(f"  ||B·C·ones||/||B·ones|| = {np.linalg.norm(BCx_test) / np.linalg.norm(B_op.matvec(x_test)):.6f}")

# Test: Solve a small matrix completely
print("\nSmall matrix test (construct dense matrix from 10 samples):")
n_test = 10
B_dense = np.zeros((n_test, n_test))
C_dense = np.zeros((n_test, n_test))
CB_dense = np.zeros((n_test, n_test))

for j in range(n_test):
    ej = np.zeros(n_cells)
    ej[j] = 1.0
    
    B_dense[:, j] = B_op.matvec(ej)[:n_test]
    C_dense[:, j] = C_op.matvec(ej)[:n_test]
    CB_dense[:, j] = C_op.matvec(B_op.matvec(ej))[:n_test]

cond_B = np.linalg.cond(B_dense)
cond_CB = np.linalg.cond(CB_dense)

print(f"  cond(B) ≈ {cond_B:.3e}")
print(f"  cond(CB) ≈ {cond_CB:.3e}")
print(f"  Ratio = {cond_CB / cond_B:.4f}")
if cond_CB < cond_B:
    print(f"  ✓ Preconditioner IMPROVES conditioning by {cond_B / cond_CB:.2f}x")
else:
    print(f"  ✗ Preconditioner WORSENS conditioning by {cond_CB / cond_B:.2f}x")
