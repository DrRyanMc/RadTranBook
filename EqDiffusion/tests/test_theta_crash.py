#!/usr/bin/env python3
"""
Simple test to understand why theta=0.5 crashes
"""

import numpy as np
from oneDFV import RadiationDiffusionSolver, A_RAD, temperature_from_Er

# Marshak wave parameters  
SIGMA_R = 300.0
C_V = 0.3

def marshak_opacity(Er):
    T = temperature_from_Er(Er)
    T_min = 0.001
    if T < T_min:
        T = T_min
    return SIGMA_R / T**3

def marshak_cv(T):
    return C_V

def marshak_material_energy(T):
    return C_V * T

def marshak_left_bc(Er, r):
    T_bc = 1.0
    Er_bc = A_RAD * T_bc**4
    return 1.0, 0.0, Er_bc

def marshak_right_bc(Er, r):
    return 1.0, -1.0, 0.0

# Start simple
for theta in [1.0, 0.9, 0.75, 0.5]:
    print(f"\n{'='*70}")
    print(f"Testing theta = {theta}")
    print('='*70)
    
    solver = RadiationDiffusionSolver(
        r_min=0.0, r_max=0.3, n_cells=50, d=0,
        dt=0.01, theta=theta,
        max_newton_iter=5, newton_tol=1e-6,
        rosseland_opacity_func=marshak_opacity,
        specific_heat_func=marshak_cv,
        material_energy_func=marshak_material_energy,
        left_bc_func=marshak_left_bc,
        right_bc_func=marshak_right_bc
    )
    
    T_init = 0.01
    solver.set_initial_condition(A_RAD * T_init**4)
    
    try:
        # Take a few steps
        for step in range(10):
            solver.time_step(n_steps=1, verbose=False)
            r, Er = solver.get_solution()
            T = np.array([temperature_from_Er(er) for er in Er])
            print(f"  Step {step+1}: T_max = {T.max():.4f}, T[0] = {T[0]:.4f}, T[10] = {T[10]:.4f}")
            
            # Check for problems
            if np.any(~np.isfinite(T)):
                print(f"  ✗ Non-finite values detected!")
                break
            if np.any(T < 0):
                print(f"  ✗ Negative temperatures!")
                break
                
        print(f"✓ Completed 10 steps successfully")
        
    except Exception as e:
        print(f"✗ FAILED after {step} steps: {e}")

print("\n" + "="*70)
