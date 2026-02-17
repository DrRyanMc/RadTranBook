#!/usr/bin/env python3
"""
Debug the theta method implementation by examining what happens in detail
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
from oneDFV import RadiationDiffusionSolver, A_RAD

def constant_opacity(T):
    return 1.0

def constant_cv(T):
    return 1.0

def material_energy(T):
    return constant_cv(T) * T

def left_bc(Er, r):
    T_bc = 1.0
    Er_bc = A_RAD * T_bc**4
    return 1.0, 0.0, Er_bc

def right_bc(Er, r):
    return 1.0, -1.0, 0.0

# Create a simple solver
solver = RadiationDiffusionSolver(
    r_min=0.0, r_max=1.0, n_cells=5, d=0,
    dt=0.01, theta=0.5,
    max_newton_iter=3, newton_tol=1e-10,
    rosseland_opacity_func=constant_opacity,
    specific_heat_func=constant_cv,
    material_energy_func=material_energy,
    left_bc_func=left_bc,
    right_bc_func=right_bc
)

T_init = 0.1
Er_init = A_RAD * T_init**4
solver.set_initial_condition(Er_init)

print("Initial E_r:", solver.Er)
print()

# Manually call assemble_system to see what's happening
Er_k = solver.Er.copy()
Er_prev = solver.Er_old.copy()

print(f"Theta = {solver.theta}")
print(f"dt = {solver.dt}")
print()

# Call assemble_system multiple times with different Er_k to simulate Newton iterations
for iteration in range(3):
    print(f"=== Newton Iteration {iteration} ===")
    print(f"Er_k = {Er_k}")
    print(f"Er_prev (should not change) = {Er_prev}")
    print()
    
    A_tri, rhs = solver.assemble_system(Er_k, Er_prev)
    
    print(f"Matrix diagonal (sample): {A_tri[1, :3]}")
    print(f"RHS (sample): {rhs[:3]}")
    print()
    
    # Perturb Er_k for next iteration to simulate Newton
    Er_k = Er_k * 1.01
    
print("\n" + "="*70)
print("Key observations:")
print("1. Does Er_prev stay constant? (It should!)")
print("2. Does the RHS change each iteration? (It shouldn't change much if explicit part is constant)")
print("="*70)
