#!/usr/bin/env python3
"""
Debug TR-BDF2 diffusion rate
Compare TR-BDF2 vs Implicit Euler step-by-step to diagnose why diffusion is too slow
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
import matplotlib.pyplot as plt
from oneDFV import RadiationDiffusionSolver, temperature_from_Er, A_RAD, C_LIGHT

# Setup
n_cells = 200
r_max = 2.0
dt = 0.001
sigma_R = 100.0
D = C_LIGHT / (3.0 * sigma_R)
k_coupling = 1.0
x0 = 1.0
sigma0 = 0.15
amplitude = A_RAD * 1.0**4

def constant_opacity(T):
    return sigma_R

def cubic_cv(T):
    return 4 * k_coupling * A_RAD * T**3

def linear_material_energy(T):
    return k_coupling * A_RAD * T**4

def left_bc(r, t):
    return (0.0, 1.0, 0.0)

def right_bc(r, t):
    return (0.0, 1.0, 0.0)

def gaussian_Er(r):
    return amplitude * np.exp(-(r - x0)**2 / (2 * sigma0**2))

def analytical_solution(r, t):
    D_eff = D / (1.0 + k_coupling)
    sigma_t_sq = sigma0**2 + 2 * D_eff * t
    sigma_t = np.sqrt(sigma_t_sq)
    amplitude_t = amplitude * sigma0 / sigma_t
    return amplitude_t * np.exp(-(r - x0)**2 / (2 * sigma_t_sq))

print("="*70)
print("TR-BDF2 vs Implicit Euler Diffusion Comparison")
print("="*70)

# Test different numbers of steps
test_cases = [
    (10, 0.001),   # 10 steps of dt=0.001
    (50, 0.001),   # 50 steps of dt=0.001
    (100, 0.001),  # 100 steps of dt=0.001
    (500, 0.001),  # 500 steps of dt=0.001
]

for n_steps, dt_test in test_cases:
    t_final = n_steps * dt_test
    
    print(f"\nTest: n_steps={n_steps}, dt={dt_test}, t_final={t_final}")
    print("-"*70)
    
    # Analytical solution at final time
    r_test = np.linspace(0.1, r_max, n_cells)
    Er_analytical = analytical_solution(r_test, t_final)
    T_analytical = temperature_from_Er(Er_analytical)
    
    # TR-BDF2
    solver_trbdf2 = RadiationDiffusionSolver(
        n_cells=n_cells, r_max=r_max, dt=dt_test, theta=0.5, d=0,
        rosseland_opacity_func=constant_opacity,
        specific_heat_func=cubic_cv,
        material_energy_func=linear_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    solver_trbdf2.set_initial_condition(gaussian_Er)
    solver_trbdf2.time_step_trbdf2(n_steps=n_steps, verbose=False)
    r, Er_trbdf2 = solver_trbdf2.get_solution()
    T_trbdf2 = temperature_from_Er(Er_trbdf2)
    
    # Implicit Euler
    solver_ie = RadiationDiffusionSolver(
        n_cells=n_cells, r_max=r_max, dt=dt_test, theta=1.0, d=0,
        rosseland_opacity_func=constant_opacity,
        specific_heat_func=cubic_cv,
        material_energy_func=linear_material_energy,
        left_bc_func=left_bc,
        right_bc_func=right_bc
    )
    solver_ie.set_initial_condition(gaussian_Er)
    solver_ie.time_step(n_steps=n_steps, verbose=False)
    r, Er_ie = solver_ie.get_solution()
    T_ie = temperature_from_Er(Er_ie)
    
    # Errors
    error_trbdf2 = np.sqrt(np.mean((Er_trbdf2 - Er_analytical)**2))
    rel_error_trbdf2 = error_trbdf2 / np.sqrt(np.mean(Er_analytical**2))
    
    error_ie = np.sqrt(np.mean((Er_ie - Er_analytical)**2))
    rel_error_ie = error_ie / np.sqrt(np.mean(Er_analytical**2))
    
    # Peak comparison
    print(f"  Analytical:  Peak T = {T_analytical.max():.4f} keV, Peak Er = {Er_analytical.max():.6e}")
    print(f"  TR-BDF2:     Peak T = {T_trbdf2.max():.4f} keV, Peak Er = {Er_trbdf2.max():.6e}")
    print(f"               Error = {rel_error_trbdf2*100:.4f}%, Peak diff = {(T_trbdf2.max()-T_analytical.max())*1000:.2f} eV")
    print(f"  Impl. Euler: Peak T = {T_ie.max():.4f} keV, Peak Er = {Er_ie.max():.6e}")
    print(f"               Error = {rel_error_ie*100:.4f}%, Peak diff = {(T_ie.max()-T_analytical.max())*1000:.2f} eV")
    
    # Diffusion rate comparison
    peak_decay_analytical = 1.0 - T_analytical.max() / 1.0
    peak_decay_trbdf2 = 1.0 - T_trbdf2.max() / 1.0
    peak_decay_ie = 1.0 - T_ie.max() / 1.0
    
    print(f"  Peak decay:")
    print(f"    Analytical:  {peak_decay_analytical*100:.2f}%")
    print(f"    TR-BDF2:     {peak_decay_trbdf2*100:.2f}% (should be {peak_decay_analytical*100:.2f}%)")
    print(f"    Impl Euler:  {peak_decay_ie*100:.2f}%")

print("\n" + "="*70)
print("Summary:")
print("="*70)
print("If TR-BDF2 peak decay is consistently less than analytical,")
print("it means the diffusion is too weak.")
print("Implicit Euler should also be less (first-order accurate),")
print("but TR-BDF2 should be closer (second-order accurate).")
