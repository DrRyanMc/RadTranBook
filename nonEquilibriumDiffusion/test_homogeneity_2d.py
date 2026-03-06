#!/usr/bin/env python3
"""Test 2D Robin BC homogeneity"""

import numpy as np
from diffusion_operator_solver import DiffusionOperatorSolver2D, C_LIGHT

def robin_bc_left(phi, pos, t):
    """Robin BC: A*phi + B*dphi/dr = C"""
    return 0.5, 2.0, 1.0  # A, B, C

def robin_bc_other(phi, pos, t):
    return 0.0, 1.0, 0.0  # Neumann (reflecting)

# Create solver with Robin BCs
solver = DiffusionOperatorSolver2D(
    x_min=0.0,
    x_max=1.0,
    nx_cells=10,
    y_min=0.0,
    y_max=0.1,
    ny_cells=3,
    geometry='cartesian',
    diffusion_coeff_func=lambda T, x, y, *args: 1.0,
    absorption_coeff_func=lambda T, x, y: 1.0,
    dt=1.0,
    left_bc_func=robin_bc_left,
    right_bc_func=robin_bc_other,
    bottom_bc_func=robin_bc_other,
    top_bc_func=robin_bc_other
)

# Test homogeneity
n_total = solver.n_total
T = np.ones(n_total)
alpha = 3.7

# Create two test vectors
x1 = np.random.rand(n_total)
x2 = alpha * x1

# Assemble matrix with same temperature
A1, D_x1, D_y1, diag1 = solver.assemble_matrix(T, x1)
rhs1 = np.zeros(n_total)
A1 = solver.apply_boundary_conditions(A1, rhs1, x1, T, D_x1, D_y1)

A2, D_x2, D_y2, diag2 = solver.assemble_matrix(T, x2)
rhs2 = np.zeros(n_total)
A2 = solver.apply_boundary_conditions(A2, rhs2, x2, T, D_x2, D_y2)

# For homogeneity, we need A to be independent of phi
# Check if matrix elements are the same
print("2D Homogeneity Test with Robin BCs")
print("="*60)
print(f"Grid: {solver.nx_cells} × {solver.ny_cells} = {n_total} cells")
print(f"Matrix difference: ||A1 - A2||_F = {np.linalg.norm((A1 - A2).toarray()):.3e}")
print(f"Matrix norm: ||A1||_F = {np.linalg.norm(A1.toarray()):.3e}")
rel_diff = np.linalg.norm((A1 - A2).toarray()) / np.linalg.norm(A1.toarray())
print(f"Relative difference: {rel_diff:.3e}")

# Check boundary contribution directly for left boundary cells
print("\nLeft boundary diagonal elements (i=0, varying j):")
for j in range(solver.ny_cells):
    idx = solver._index_2d_to_1d(0, j)
    print(f"  j={j}: A1[{idx},{idx}] = {A1[idx,idx]:.10e}, A2[{idx},{idx}] = {A2[idx,idx]:.10e}, diff = {abs(A1[idx,idx] - A2[idx,idx]):.3e}")

# Also test with phi-dependent diffusion coefficient
print("\n" + "="*60)
print("Test with phi-dependent diffusion (should still be homogeneous)")
print("="*60)

def phi_dep_diffusion(T, x, y, phi_left, phi_right, *args):
    """Diffusion that depends on phi (flux-limited)"""
    phi_avg = 0.5 * (abs(phi_left) + abs(phi_right)) + 1e-10
    return 1.0 / (1.0 + phi_avg)  # Example: D decreases with phi

solver2 = DiffusionOperatorSolver2D(
    x_min=0.0,
    x_max=1.0,
    nx_cells=10,
    y_min=0.0,
    y_max=0.1,
    ny_cells=3,
    geometry='cartesian',
    diffusion_coeff_func=phi_dep_diffusion,
    absorption_coeff_func=lambda T, x, y: 1.0,
    dt=1.0,
    left_bc_func=robin_bc_left,
    right_bc_func=robin_bc_other,
    bottom_bc_func=robin_bc_other,
    top_bc_func=robin_bc_other
)

A3, D_x3, D_y3, diag3 = solver2.assemble_matrix(T, x1)
rhs3 = np.zeros(n_total)
A3 = solver2.apply_boundary_conditions(A3, rhs3, x1, T, D_x3, D_y3)

A4, D_x4, D_y4, diag4 = solver2.assemble_matrix(T, x2)
rhs4 = np.zeros(n_total)
A4 = solver2.apply_boundary_conditions(A4, rhs4, x2, T, D_x4, D_y4)

print(f"Matrix difference: ||A3 - A4||_F = {np.linalg.norm((A3 - A4).toarray()):.3e}")
print(f"Matrix norm: ||A3||_F = {np.linalg.norm(A3.toarray()):.3e}")
rel_diff2 = np.linalg.norm((A3 - A4).toarray()) / np.linalg.norm(A3.toarray())
print(f"Relative difference: {rel_diff2:.3e}")
print("\nNote: With flux-limited diffusion, homogeneity is expected to fail")
print("because the operator A depends on phi (nonlinear problem).")
