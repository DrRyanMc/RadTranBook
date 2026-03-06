#!/usr/bin/env python3
"""Test 2D Robin BC homogeneity with temperature-dependent diffusion"""

import numpy as np
from diffusion_operator_solver import DiffusionOperatorSolver2D, C_LIGHT

# Temperature-dependent diffusion (like in the multigroup test)
def temp_dependent_diffusion(T, x, y, *args):
    """D(T) = c / (3*sigma(T)) where sigma ∝ T^(-0.5)"""
    T_safe = max(T, 1e-2)
    sigma = 100.0 * T_safe**(-0.5)  # Example opacity
    return C_LIGHT / (3.0 * sigma)

# Marshak BC with fixed boundary temperature
T_bc = 0.05
D_bc = temp_dependent_diffusion(T_bc, 0, 0)

def robin_bc_left(phi, pos, t):
    """Marshak BC: A*phi + B*dphi/dr = C, with B = 2*D(T_bc)"""
    return 0.5, 2.0 * D_bc, 1.0  # Fixed B_bc = 2*D(T_bc)

def robin_bc_other(phi, pos, t):
    return 0.0, 1.0, 0.0  # Neumann

# Create solver
solver = DiffusionOperatorSolver2D(
    x_min=0.0, x_max=1.0, nx_cells=10,
    y_min=0.0, y_max=0.1, ny_cells=3,
    geometry='cartesian',
    diffusion_coeff_func=temp_dependent_diffusion,
    absorption_coeff_func=lambda T, x, y: 1.0,
    dt=1.0,
    left_bc_func=robin_bc_left,
    right_bc_func=robin_bc_other,
    bottom_bc_func=robin_bc_other,
    top_bc_func=robin_bc_other
)

n_total = solver.n_total
alpha = 3.7
x1 = np.random.rand(n_total)
x2 = alpha * x1

print("2D Homogeneity Test with Temperature-Dependent Diffusion")
print("="*70)

# Test 1: Uniform temperature T = T_bc (should be homogeneous)
print("\nTest 1: Uniform temperature T = T_bc = 0.05 keV")
T_uniform = np.full(n_total, T_bc)

A1, D_x1, D_y1, diag1 = solver.assemble_matrix(T_uniform, x1)
rhs1 = np.zeros(n_total)
A1 = solver.apply_boundary_conditions(A1, rhs1, x1, T_uniform, D_x1, D_y1)

A2, D_x2, D_y2, diag2 = solver.assemble_matrix(T_uniform, x2)
rhs2 = np.zeros(n_total)
A2 = solver.apply_boundary_conditions(A2, rhs2, x2, T_uniform, D_x2, D_y2)

diff1 = np.linalg.norm((A1 - A2).toarray())
norm1 = np.linalg.norm(A1.toarray())
print(f"  Matrix difference: ||A1 - A2||_F = {diff1:.3e}")
print(f"  Relative difference: {diff1/norm1:.3e}")

# Test 2: Non-uniform temperature (T varies spatially)
print("\nTest 2: Non-uniform temperature (T varies from 0.02 to 0.08 keV)")
T_nonuniform = np.linspace(0.02, 0.08, n_total)

A3, D_x3, D_y3, diag3 = solver.assemble_matrix(T_nonuniform, x1)
rhs3 = np.zeros(n_total)
A3 = solver.apply_boundary_conditions(A3, rhs3, x1, T_nonuniform, D_x3, D_y3)

A4, D_x4, D_y4, diag4 = solver.assemble_matrix(T_nonuniform, x2)
rhs4 = np.zeros(n_total)
A4 = solver.apply_boundary_conditions(A4, rhs4, x2, T_nonuniform, D_x4, D_y4)

diff2 = np.linalg.norm((A3 - A4).toarray())
norm2 = np.linalg.norm(A3.toarray())
print(f"  Matrix difference: ||A3 - A4||_F = {diff2:.3e}")
print(f"  Relative difference: {diff2/norm2:.3e}")

print("\nConclusion:")
print("  - If relative difference is ~1e-14: Perfect homogeneity (the fix works)")
print("  - If relative difference is ~1e-4: Homogeneity broken (inconsistent BC/interior)")
