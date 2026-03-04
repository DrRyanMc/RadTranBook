#!/usr/bin/env python3
"""
Test flux limiter functionality in multigroup solver

Compares standard diffusion (no limiting) with flux-limited diffusion
for a simple problem.
"""
import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multigroup_diffusion_solver import (
    MultigroupDiffusionSolver1D, 
    C_LIGHT, A_RAD,
    flux_limiter_standard,
    flux_limiter_levermore_pomraning,
    flux_limiter_larsen,
    flux_limiter_sum,
    flux_limiter_max
)

print("="*80)
print("FLUX LIMITER TEST")
print("="*80)

# Problem parameters
n_groups = 1  # Single group
n_cells = 100
x_max = 10.0
dt = 0.01
sigma_a = 1.0  # Absorption opacity
sigma_R = 1.0  # Rosseland opacity

# Energy edges (arbitrary for gray)
energy_edges = np.array([0.0, 10.0])

# Material properties (constant cv for simplicity)
rho = 1.0
cv = 0.1

# Diffusion and absorption coefficients
def absorption_coeff(T, r):
    return sigma_a

def rosseland_opacity(T, r):
    return sigma_R

# Boundary conditions: hot left, cold right
T_hot = 1.0
T_cold = 0.01

def left_bc(phi, r):
    """Dirichlet BC at hot temperature"""
    phi_bc = A_RAD * C_LIGHT * T_hot**4
    return 1.0, 0.0, phi_bc

def right_bc(phi, r):
    """Dirichlet BC at cold temperature"""  
    phi_bc = A_RAD * C_LIGHT * T_cold**4
    return 1.0, 0.0, phi_bc

print(f"\nProblem setup:")
print(f"  Domain: [0, {x_max}] cm with {n_cells} cells")
print(f"  Left BC: T = {T_hot} keV")
print(f"  Right BC: T = {T_cold} keV")
print(f"  Timestep: dt = {dt} ns")

# Test different flux limiters
limiters = {
    'Standard (no limiting)': None,
    'Levermore-Pomraning': flux_limiter_levermore_pomraning,
    'Larsen (n=2)': flux_limiter_larsen,
    'Sum': flux_limiter_sum,
    'Max': flux_limiter_max,
}

results = {}
colors = ['black', 'blue', 'red', 'green', 'orange']

print(f"\n{'='*80}")
print("Running simulations with different flux limiters...")
print(f"{'='*80}")

for idx, (limiter_name, limiter_func) in enumerate(limiters.items()):
    print(f"\n{limiter_name}:")
    print("-" * 80)
    
    # Create solver
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=0.0,
        r_max=x_max,
        n_cells=n_cells,
        energy_edges=energy_edges,
        geometry='planar',
        dt=dt,
        diffusion_coeff_funcs=None,  # Will be wrapped by flux limiter
        absorption_coeff_funcs=[absorption_coeff],
        left_bc_funcs=[left_bc],
        right_bc_funcs=[right_bc],
        flux_limiter_funcs=limiter_func,  # Apply flux limiter
        rosseland_opacity_funcs=[rosseland_opacity],
        rho=rho,
        cv=cv
    )
    
    # Initial conditions: linear temperature profile
    T_init = T_cold + (T_hot - T_cold) * (1.0 - solver.r_centers / x_max)
    solver.T = T_init
    solver.T_old = solver.T.copy()
    solver.E_r = A_RAD * T_init**4
    solver.E_r_old = solver.E_r.copy()
    
    # Run for several timesteps
    n_steps = 10
    for step in range(n_steps):
        info = solver.step(max_newton_iter=10, newton_tol=1e-8, verbose=False)
        solver.advance_time()
        
        if step == 0 or step == n_steps - 1:
            converged = "✓" if info['converged'] else "✗"
            print(f"  Step {step+1}: T_max={solver.T.max():.4f} keV, Newton={info['newton_iter']} {converged}")
    
    # Store results
    results[limiter_name] = {
        'r': solver.r_centers.copy(),
        'T': solver.T.copy(),
        'E_r': solver.E_r.copy(),
        'color': colors[idx]
    }

print(f"\n{'='*80}")
print("Creating comparison plot...")
print(f"{'='*80}")

# Plot comparison
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for limiter_name, data in results.items():
    ax1.plot(data['r'], data['T'], label=limiter_name, 
            color=data['color'], linewidth=2, alpha=0.8)
    ax2.plot(data['r'], data['E_r'], label=limiter_name,
            color=data['color'], linewidth=2, alpha=0.8)

ax1.set_xlabel('Position (cm)', fontsize=12)
ax1.set_ylabel('Temperature (keV)', fontsize=12)
ax1.set_title('Temperature Profile', fontsize=14)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

ax2.set_xlabel('Position (cm)', fontsize=12)
ax2.set_ylabel('Radiation Energy Density (GJ/cm³)', fontsize=12)
ax2.set_title('Radiation Energy', fontsize=14)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('flux_limiter_comparison.pdf')
print("Saved plot: flux_limiter_comparison.pdf")

# Summary
print(f"\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")
print(f"Successfully tested {len(limiters)} flux limiters:")
for limiter_name in limiters.keys():
    T_final = results[limiter_name]['T']
    print(f"  {limiter_name}: T_max = {T_final.max():.4f} keV, T_min = {T_final.min():.4f} keV")
print(f"\nFlux limiters reduce diffusion in optically thin regions,")
print(f"showing visible differences from standard diffusion.")
print("="*80)
