#!/usr/bin/env python3
"""
Understand what the original (theta=1) implementation is doing

The discretized equation for implicit Euler is:
(E_r^{n+1} - E_r^n)/dt = ∇·(D∇E_r^{n+1}) + source terms

This is linearized using Newton's method in a special form.
Let me trace through what happens for a simple case.
"""

import numpy as np
from oneDFV import RadiationDiffusionSolver, A_RAD

def constant_opacity(T):
    return 1.0

def constant_cv(T):
    return 1.0

def material_energy(T):
    return constant_cv(T) * T

def zero_bc(Er, r):
    return 1.0, 0.0, 0.0

# Create solver with theta=1.0 (original implicit Euler)
solver = RadiationDiffusionSolver(
    r_min=0.0, r_max=1.0, n_cells=3, d=0,
    dt=0.1, theta=1.0,
    max_newton_iter=1, newton_tol=1e-10,
    rosseland_opacity_func=constant_opacity,
    specific_heat_func=constant_cv,
    material_energy_func=material_energy,
    left_bc_func=zero_bc,
    right_bc_func=zero_bc
)

# Set uniform initial condition
Er_init = 1.0
solver.set_initial_condition(Er_init)

print("="*70)
print("UNDERSTANDING ORIGINAL IMPLICIT EULER (theta=1)")
print("="*70)
print(f"\nInitial condition: Er = {Er_init} everywhere")
print(f"Time step: dt = {solver.dt}")
print(f"Number of cells: {solver.n_cells}")
print()

# Manually call assemble_system to see what it does
Er_k = solver.Er.copy()
Er_prev = solver.Er_old.copy()

print("Calling assemble_system with Er_k = Er_prev (first Newton iteration)")
A_tri, rhs = solver.assemble_system(Er_k, Er_prev)

print("\nMatrix A (tridiagonal):")
for i in range(solver.n_cells):
    row = np.zeros(solver.n_cells)
    row[i] = A_tri[1, i]  # diagonal
    if i > 0:
        row[i-1] = A_tri[0, i]  # sub-diagonal
    if i < solver.n_cells - 1:
        row[i+1] = A_tri[2, i]  # super-diagonal
    print(f"  Row {i}: {row}")

print(f"\nRHS vector b: {rhs}")

# The system is A * phi = b where phi is the Newton correction
# So E_r^{new} = E_r^{k} + phi, or just E_r^{new} = phi in the linearized form used here

print("\nSolving A * E_r^{n+1} = b...")
from oneDFV import solve_tridiagonal
Er_new = solve_tridiagonal(A_tri, rhs)
print(f"Solution: E_r^{{n+1}} = {Er_new}")
print(f"Change: ΔE_r = {Er_new - Er_k}")

print("\n" + "="*70)
print("KEY OBSERVATIONS:")
print("="*70)
print("1. The matrix A includes diffusion operator + alpha term")
print("2. The RHS b includes alpha*Er_k - u(Er_k) + Qhat(Er_prev)")
print("3. For theta method, we need to weight the operator terms")
print("="*70)
