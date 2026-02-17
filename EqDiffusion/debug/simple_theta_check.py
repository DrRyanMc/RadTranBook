#!/usr/bin/env python3
"""
Simple check: manually verify the theta method discretization
by checking that we recover the correct equation.
"""

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

print("Testing if theta=1 and theta=0.5 give same result at t=0")
print("(They should be identical for the first step from a steady initial condition)")
print()

for theta in [1.0, 0.5]:
    solver = RadiationDiffusionSolver(
        r_min=0.0, r_max=1.0, n_cells=10, d=0,
        dt=0.001, theta=theta,
        max_newton_iter=10, newton_tol=1e-12,
        rosseland_opacity_func=constant_opacity,
        specific_heat_func=constant_cv,
        material_energy_func=material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    
    # Start from a uniform state
    T_init = 0.5
    Er_init = A_RAD * T_init**4
    solver.set_initial_condition(Er_init)
    
    print(f"Theta = {theta}:")
    print(f"  Initial E_r: {solver.Er[5]:.6e}")
    
    # Take one small step
    solver.time_step(n_steps=1, verbose=False)
    
    r, Er = solver.get_solution()
    print(f"  After 1 step E_r: {Er[5]:.6e}")
    print(f"  Change: {Er[5] - Er_init:.6e}")
    print()

print("\nIf theta implementation is correct:")
print("- Both methods should show energy increasing (hot left BC)")
print("- The changes should be similar in magnitude")
print("- theta=0.5 might show slightly different change due to averaging")
