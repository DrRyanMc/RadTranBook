#!/usr/bin/env python3
"""
Verify that with theta < 1, the explicit diffusion contribution
stays constant during Newton iterations.
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
    T_bc = 1.5
    Er_bc = A_RAD * T_bc**4
    return 1.0, 0.0, Er_bc

def right_bc(Er, r):
    return 1.0, -1.0, 0.0

# Create solver with theta = 0.5 and multiple Newton iterations
solver = RadiationDiffusionSolver(
    r_min=0.0, r_max=1.0, n_cells=10, d=0,
    dt=0.1, theta=0.5,  # Larger time step to require more iterations
    max_newton_iter=10, newton_tol=1e-10,
    rosseland_opacity_func=constant_opacity,
    specific_heat_func=constant_cv,
    material_energy_func=material_energy,
    left_bc_func=left_bc,
    right_bc_func=right_bc
)

# Non-uniform initial condition to require multiple Newton iterations
def initial_profile(r):
    return A_RAD * (0.1 + 0.05 * np.sin(np.pi * r))**4  # Colder initial condition

solver.set_initial_condition(initial_profile)

# Instrument to track explicit diffusion during Newton iterations
original_assemble = solver.assemble_system
explicit_history = []

def track_explicit(Er_k, Er_prev):
    # Recompute explicit diffusion to verify it's constant
    if solver.theta < 1.0:
        n_cells = len(Er_prev)
        n_faces = len(solver.r_faces)
        D_cells_prev = np.array([solver.get_diffusion_coefficient(Er_prev[i]) for i in range(n_cells)])
        D_faces_prev = np.zeros(n_faces)
        
        for i in range(1, n_faces - 1):
            from oneDFV import temperature_from_Er
            T_left = temperature_from_Er(Er_prev[i-1])
            T_right = temperature_from_Er(Er_prev[i])
            T_face = 0.5 * (T_left + T_right)
            Er_face = A_RAD * T_face**4
            D_faces_prev[i] = solver.get_diffusion_coefficient(Er_face)
        
        D_faces_prev[0] = D_cells_prev[0]
        D_faces_prev[-1] = D_cells_prev[-1]
        
        explicit_diffusion = np.zeros(n_cells)
        for i in range(n_cells):
            V_i = solver.V_cells[i]
            A_left = solver.A_faces[i]
            A_right = solver.A_faces[i + 1]
            
            if i == 0:
                dr_left = solver.r_centers[i] - solver.r_faces[i]
            else:
                dr_left = solver.r_centers[i] - solver.r_centers[i - 1]
                
            if i == n_cells - 1:
                dr_right = solver.r_faces[i + 1] - solver.r_centers[i]
            else:
                dr_right = solver.r_centers[i + 1] - solver.r_centers[i]
            
            contrib_from_left = 0.0
            contrib_from_right = 0.0
            contrib_center = 0.0
            
            if i > 0:
                coeff_left = A_left * D_faces_prev[i] / (dr_left * V_i)
                contrib_center += coeff_left * Er_prev[i]
                contrib_from_left = -coeff_left * Er_prev[i-1]
                
            if i < n_cells - 1:
                coeff_right = A_right * D_faces_prev[i+1] / (dr_right * V_i)
                contrib_center += coeff_right * Er_prev[i]
                contrib_from_right = -coeff_right * Er_prev[i+1]
            
            explicit_diffusion[i] = contrib_center + contrib_from_left + contrib_from_right
        
        explicit_history.append(explicit_diffusion.copy())
    
    return original_assemble(Er_k, Er_prev)

solver.assemble_system = track_explicit

print("="*70)
print("VERIFYING EXPLICIT DIFFUSION IS CONSTANT DURING NEWTON ITERATIONS")
print("="*70)
print(f"Theta = {solver.theta}")
print(f"Max Newton iterations = {solver.max_newton_iter}")
print()

# Take one time step (this will do multiple Newton iterations)
solver.time_step(n_steps=1, verbose=False)

print(f"Number of Newton iterations performed: {len(explicit_history)}")
print()

if len(explicit_history) > 1:
    print("Explicit diffusion vector at each Newton iteration:")
    for iter_num, explicit in enumerate(explicit_history):
        print(f"  Iteration {iter_num}: [", end="")
        print(", ".join(f"{x:.6f}" for x in explicit[:3]), end="")
        print(", ...]")
    
    print("\nChecking if explicit diffusion is constant:")
    max_variation = 0.0
    for i in range(1, len(explicit_history)):
        variation = np.max(np.abs(explicit_history[i] - explicit_history[0]))
        max_variation = max(max_variation, variation)
        print(f"  Iteration {i} vs 0: max variation = {variation:.2e}")
    
    print()
    if max_variation < 1e-12:
        print("✓ PASSED: Explicit diffusion is constant (variation < 1e-12)")
    else:
        print(f"✗ FAILED: Explicit diffusion varies by {max_variation:.2e}")
        print("This indicates Er_prev is being modified during Newton iterations")
else:
    print("⚠ Only one iteration - cannot verify constancy")
    print("Try with a more challenging initial condition or looser tolerance")

print("="*70)
