#!/usr/bin/env python3
"""Test 1D Robin BC homogeneity"""

import numpy as np
from diffusion_operator_solver import DiffusionOperatorSolver1D, C_LIGHT

def robin_bc_left(phi, r):
    """Robin BC: A*phi + B*dphi/dr = C"""
    return 0.5, 2.0, 1.0  # A, B, C

def robin_bc_right(phi, r):
    return 0.0, 1.0, 0.0  # Neumann (reflecting)

# Create solver with Robin BCs
solver = DiffusionOperatorSolver1D(
    r_min=0.0,
    r_max=1.0,
    n_cells=50,
    geometry='planar',
    diffusion_coeff_func=lambda T, r, *args: 1.0,
    absorption_coeff_func=lambda T, r: 1.0,
    dt=1.0,
    left_bc_func=robin_bc_left,
    right_bc_func=robin_bc_right
)

# Test homogeneity
T = np.ones(50)
alpha = 3.7

# Create two test vectors
x1 = np.random.rand(50)
x2 = alpha * x1

# Assemble matrix with same temperature
A1, D_faces1, diag1 = solver.assemble_matrix(T, x1)
A1 = solver.apply_boundary_conditions(A1, np.zeros(50), x1, T, D_faces1)

A2, D_faces2, diag2 = solver.assemble_matrix(T, x2)
A2 = solver.apply_boundary_conditions(A2, np.zeros(50), x2, T, D_faces2)

# For homogeneity, we need A to be independent of phi
# Check if matrix elements are the same
print("1D Homogeneity Test with Robin BCs")
print("="*60)
print(f"Matrix difference: ||A1 - A2||_F = {np.linalg.norm((A1 - A2).toarray()):.3e}")
print(f"Matrix norm: ||A1||_F = {np.linalg.norm(A1.toarray()):.3e}")
rel_diff = np.linalg.norm((A1 - A2).toarray()) / np.linalg.norm(A1.toarray())
print(f"Relative difference: {rel_diff:.3e}")

# Check boundary contribution directly
print("\nBoundary diagonal elements:")
print(f"  A1[0,0] = {A1[0,0]:.10e}")
print(f"  A2[0,0] = {A2[0,0]:.10e}")
print(f"  Difference: {abs(A1[0,0] - A2[0,0]):.3e}")
