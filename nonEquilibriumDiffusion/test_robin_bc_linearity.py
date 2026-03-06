#!/usr/bin/env python3
"""
Test Robin BC linearity in diffusion operator
"""
import numpy as np
from diffusion_operator_solver import DiffusionOperatorSolver2D, C_LIGHT

# Create a simple 2D problem with Robin BCs
def robin_bc_left(phi, pos, t):
    """Robin BC: A·φ + B·∇φ = C"""
    A = 0.5
    B = 2.0  # Some diffusion-like coefficient
    C = 1.0  # Inhomo term
    return A, B, C

def robin_bc_zero(phi, pos, t):
    """Zero flux: 0·φ + 1·∇φ = 0"""
    return 0.0, 1.0, 0.0

# Create solver
nx, ny = 10, 10
solver = DiffusionOperatorSolver2D(
    x_min=0.0, x_max=1.0, nx_cells=nx,
    y_min=0.0, y_max=1.0, ny_cells=ny,
    geometry='cartesian',
    diffusion_coeff_func=lambda T, x, y: 1.0,
    absorption_coeff_func=lambda T, x, y: 1.0,
    dt=1.0,
    left_bc_func=robin_bc_left,
    right_bc_func=robin_bc_zero,
    bottom_bc_func=robin_bc_zero,
    top_bc_func=robin_bc_zero
)

# Test linearity of the operator A
print("Testing linearity of diffusion operator with Robin BCs")
print("="*60)

# Random test vectors
np.random.seed(42)
phi_x = np.random.randn(nx, ny)
phi_y = np.random.randn(nx, ny)
a, b = 2.3, -1.7

# Temperature field (constant for simplicity)
T = np.ones((nx, ny))

# Apply operator to individual vectors
A_phi_x = solver.apply_operator(phi_x, T)
A_phi_y = solver.apply_operator(phi_y, T)

# Apply operator to linear combination
phi_combo = a * phi_x + b * phi_y
A_phi_combo = solver.apply_operator(phi_combo, T)

# Expected linear combination
A_linear = a * A_phi_x + b * A_phi_y

# Check linearity error
lin_error = np.linalg.norm(A_phi_combo - A_linear) / (np.linalg.norm(A_phi_combo) + 1e-30)

print(f"||A(a*x + b*y)||     = {np.linalg.norm(A_phi_combo):.6e}")
print(f"||a*A(x) + b*A(y)||  = {np.linalg.norm(A_linear):.6e}")
print(f"Linearity error      = {lin_error:.6e}")
print()

if lin_error < 1e-12:
    print("✓ Operator is LINEAR (error < 1e-12)")
else:
    print(f"✗ Operator is NON-LINEAR (error = {lin_error:.3e})")
    print("  This indicates a problem with BC implementation!")
