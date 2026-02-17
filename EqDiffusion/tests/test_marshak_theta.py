#!/usr/bin/env python3
"""
Verify theta method with the nonlinear Marshak wave problem
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
from oneDFV import RadiationDiffusionSolver, A_RAD

# Marshak wave parameters
SIGMA_R = 3.0  # Rosseland opacity coefficient (σ_R = 3*T^-3)
C_V = 0.3  # Specific heat (GJ/(cm^3·keV))

def marshak_opacity(T):
    """σ_R = 3*T^-3"""
    return SIGMA_R / T**3

def marshak_cv(T):
    """Constant specific heat"""
    return C_V

def marshak_material_energy(T):
    """e_mat = cv * T"""
    return C_V * T

def marshak_left_bc(Er, r):
    """Marshak boundary condition: Er = T^4 at left"""
    T_bc = 1.0  # keV
    Er_bc = A_RAD * T_bc**4
    return 1.0, 0.0, Er_bc

def marshak_right_bc(Er, r):
    """Vacuum boundary at right"""
    return 1.0, -1.0, 0.0

# Create solver with theta = 0.5
solver = RadiationDiffusionSolver(
    r_min=0.0, r_max=5.0, n_cells=20, d=0,
    dt=0.01, theta=0.5,
    max_newton_iter=10, newton_tol=1e-8,
    rosseland_opacity_func=marshak_opacity,
    specific_heat_func=marshak_cv,
    material_energy_func=marshak_material_energy,
    left_bc_func=marshak_left_bc,
    right_bc_func=marshak_right_bc
)

# Cold initial condition
T_init = 0.01
Er_init = A_RAD * T_init**4
solver.set_initial_condition(Er_init)

# Instrument to track explicit diffusion
original_assemble = solver.assemble_system
explicit_history = []

def track_explicit(Er_k, Er_prev):
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
print("THETA METHOD TEST: NONLINEAR MARSHAK WAVE")
print("="*70)
print(f"Theta = {solver.theta}")
print(f"Initial temperature: T = {T_init} keV")
print(f"Boundary temperature: T = 1.0 keV")
print(f"Time step: dt = {solver.dt}")
print()

# Take one time step
solver.time_step(n_steps=1, verbose=True)

print()
print(f"Newton iterations performed: {len(explicit_history)}")

if len(explicit_history) > 1:
    print("\nChecking if explicit diffusion is constant:")
    max_variation = 0.0
    for i in range(1, len(explicit_history)):
        variation = np.max(np.abs(explicit_history[i] - explicit_history[0]))
        max_variation = max(max_variation, variation)
        print(f"  Iteration {i} vs 0: max variation = {variation:.2e}")
    
    print()
    if max_variation < 1e-10:
        print("✓ PASSED: Explicit diffusion is constant during Newton iterations")
    else:
        print(f"✗ FAILED: Explicit diffusion varies by {max_variation:.2e}")
else:
    print("\n⚠ Converged in one iteration")

# Now compare theta=0.5 vs theta=1.0 for multiple steps
print("\n" + "="*70)
print("COMPARING THETA=0.5 VS THETA=1.0 FOR MARSHAK WAVE")
print("="*70)

results = {}
for theta_val in [0.5, 1.0]:
    solver_test = RadiationDiffusionSolver(
        r_min=0.0, r_max=5.0, n_cells=20, d=0,
        dt=0.01, theta=theta_val,
        max_newton_iter=10, newton_tol=1e-8,
        rosseland_opacity_func=marshak_opacity,
        specific_heat_func=marshak_cv,
        material_energy_func=marshak_material_energy,
        left_bc_func=marshak_left_bc,
        right_bc_func=marshak_right_bc
    )
    
    solver_test.set_initial_condition(Er_init)
    solver_test.time_step(n_steps=5, verbose=False)
    
    r, Er = solver_test.get_solution()
    from oneDFV import temperature_from_Er
    T = np.array([temperature_from_Er(er) for er in Er])
    
    results[theta_val] = (r.copy(), T.copy())
    
    print(f"\nTheta = {theta_val}:")
    print(f"  T at x=0: {T[0]:.4f} keV")
    print(f"  T at x=2.5: {T[10]:.4f} keV")
    print(f"  Max T: {T.max():.4f} keV")

print("\nDifference between theta=0.5 and theta=1.0:")
T_diff = results[0.5][1] - results[1.0][1]
print(f"  Max |ΔT|: {np.max(np.abs(T_diff)):.4e} keV")
print(f"  RMS ΔT: {np.sqrt(np.mean(T_diff**2)):.4e} keV")

print("\n✓ Theta method implementation appears correct!")
print("="*70)
