"""
Quick test to verify flux limiters produce different results after bug fix.
Run just 2 limiters for τ=1.0 and compare.
"""
import sys
import os
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from multigroup_diffusion_solver import (MultigroupDiffusionSolver1D,
                                         flux_limiter_levermore_pomraning,
                                         flux_limiter_sum)

# Physical constants
C_LIGHT = 2.998e1  # cm/ns
A_RAD = 0.01372    # GJ/(cm³·keV⁴)

# Problem setup (Su-Olson)
n_groups = 3
n_cells = 100  # Fewer cells for speed
x_max = 12.0  # cm
mean_free_time = 1.0 / C_LIGHT
dt = 0.01 * mean_free_time
n_steps = 100  # Run to τ=1.0

# Energy edges
energy_edges = np.linspace(0.0, 100.0, n_groups + 1)

# Material functions
def rosseland_opacity(T, r):
    return 1.0

def absorption_coeff(T, r):
    return 1.0

def material_energy_func(T):
    return A_RAD * T**4

def inverse_material_energy_func(e):
    return (e / A_RAD)**0.25

def specific_heat_func(T):
    return 4.0 * A_RAD * T**3

def left_bc(phi, r):
    return (0.0, 1.0, 0.0)

def right_bc(phi, r):
    return (1.0, 0.0, 0.0)

source_region = 0.5
source_magnitude = 0.5

results = {}

for limiter_name, limiter_func in [('Levermore-Pomraning', flux_limiter_levermore_pomraning),
                                    ('Sum', flux_limiter_sum)]:
    print(f"Running with {limiter_name}...")
    
    solver = MultigroupDiffusionSolver1D(
        n_groups=n_groups,
        r_min=0.0,
        r_max=x_max,
        n_cells=n_cells,
        energy_edges=energy_edges,
        geometry='planar',
        dt=dt,
        diffusion_coeff_funcs=None,
        absorption_coeff_funcs=[absorption_coeff] * n_groups,
        left_bc_funcs=[left_bc] * n_groups,
        right_bc_funcs=[right_bc] * n_groups,
        source_funcs=[lambda r, t, g=g: source_magnitude / n_groups if r < source_region else 0.0 
                      for g in range(n_groups)],
        emission_fractions=np.ones(n_groups) / n_groups,
        material_energy_func=material_energy_func,
        inverse_material_energy_func=inverse_material_energy_func,
        cv=specific_heat_func,
        flux_limiter_funcs=[limiter_func] * n_groups,
        rosseland_opacity_funcs=[rosseland_opacity] * n_groups
    )
    
    # Initial conditions
    T_init = 0.001
    solver.T = np.ones(n_cells) * T_init
    solver.T_old = solver.T.copy()
    solver.E_r = np.ones(n_cells) * A_RAD * T_init**4
    solver.E_r_old = solver.E_r.copy()
    solver.kappa = np.zeros(n_cells)
    solver.kappa_old = np.zeros(n_cells)
    
    for step in range(n_steps):
        info = solver.step(
            max_newton_iter=10,
            newton_tol=1e-6,
            gmres_tol=1e-6,
            gmres_maxiter=200,
            verbose=False
        )
        solver.advance_time()
        
        if (step + 1) % 20 == 0:
            print(f"  Step {step+1}/{n_steps}: t={solver.t/mean_free_time:.2f}τ")
    
    results[limiter_name] = {
        'E_r': solver.E_r.copy(),
        'T': solver.T.copy(),
        'r': solver.r_centers.copy()
    }
    print(f"  Final:  E_r_max={solver.E_r.max():.6e}, T_max={solver.T.max():.6f} keV")

# Compare results
print("\n" + "="*80)
print("COMPARISON:")
print("="*80)
E_r_diff = np.abs(results['Levermore-Pomraning']['E_r'] - results['Sum']['E_r'])
T_diff = np.abs(results['Levermore-Pomraning']['T'] - results['Sum']['T'])

print(f"Max |ΔE_r| = {E_r_diff.max():.6e}")
print(f"Max |ΔT|   = {T_diff.max():.6e}")
print(f"Relative E_r diff = {E_r_diff.max() / results['Sum']['E_r'].max():.6e}")
print(f"Relative T diff   = {T_diff.max() / results['Sum']['T'].max():.6e}")

if E_r_diff.max() > 1e-10:
    print("\n✓ SUCCESS: Flux limiters produce DIFFERENT results!")
else:
    print("\n✗ FAILURE: Flux limiters still produce IDENTICAL results")
