"""
Quick test to see if phi develops spatial gradients in Su-Olson problem.
Run just one limiter for a few timesteps and examine phi structure.
"""
import sys
import os
import numpy as np

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

from multigroup_diffusion_solver import (MultigroupDiffusionSolver1D,
                                         flux_limiter_levermore_pomraning)

# Physical constants
C_LIGHT = 2.998e1  # cm/ns
A_RAD = 0.01372    # GJ/(cm³·keV⁴)

# Problem setup (Su-Olson)
n_groups = 3
n_cells = 400
x_max = 12.0  # cm
mean_free_time = 1.0 / C_LIGHT  # 1/(c·σ) with σ=1
dt = 0.01 * mean_free_time
n_steps = 20  # Just run 20 steps

# Energy edges
energy_edges = np.linspace(0.0, 100.0, n_groups + 1)

# Material functions
def rosseland_opacity(T, r):
    return 1.0  # cm⁻¹

def absorption_coeff(T, r):
    return 1.0  # cm⁻¹ * keV/GJ

def material_energy_func(T):
    return A_RAD * T**4

def inverse_material_energy_func(e):
    return (e / A_RAD)**0.25

def specific_heat_func(T):
    return 4.0 * A_RAD * T**3

# Boundary conditions
def left_bc(phi, r):
    return (0.0, 1.0, 0.0)  # Reflecting

def right_bc(phi, r):
    return (1.0, 0.0, 0.0)  # Vacuum

# Source (radiation source in left 0.5 cm)
source_region = 0.5  # cm
source_magnitude = 0.5  # Arbitrary units

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
    flux_limiter_funcs=[flux_limiter_levermore_pomraning] * n_groups,
    rosseland_opacity_funcs=[rosseland_opacity] * n_groups
)

# Initial conditions
T_init = 0.001  # keV
solver.T = np.ones(n_cells) * T_init
solver.T_old = solver.T.copy()
solver.E_r = np.ones(n_cells) * A_RAD * T_init**4
solver.E_r_old = solver.E_r.copy()
solver.kappa = np.zeros(n_cells)
solver.kappa_old = np.zeros(n_cells)

print("Running Su-Olson for 20 steps to check phi gradient development...")
print("="*80)

for step in range(n_steps):
    # Take Newton step
    info = solver.step(
        max_newton_iter=10,
        newton_tol=1e-6,
        gmres_tol=1e-6,
        gmres_maxiter=200,
        verbose=False
    )
    
    # Advance time
    solver.advance_time()
    
    # Analyze phi_g_stored for group 0
    phi_g = solver.phi_g_stored[0, :]
    
    # Compute gradients
    phi_min = phi_g.min()
    phi_max = phi_g.max()
    phi_range = phi_max - phi_min
    phi_mean = phi_g.mean()
    
    # Compute R at faces
    R_values = []
    for i in range(1, n_cells):
        phi_left = phi_g[i-1]
        phi_right = phi_g[i]
        dx = solver.r_centers[i] - solver.r_centers[i-1]
        grad_phi = abs(phi_right - phi_left) / dx
        phi_avg = 0.5 * (phi_left + phi_right)
        sigma_R = 1.0
        if phi_avg > 1e-20:
            R = grad_phi / (sigma_R * phi_avg)
            R_values.append(R)
    
    R_array = np.array(R_values) if R_values else np.array([0.0])
    
    print(f"Step {step+1:3d}: t={solver.t/mean_free_time:.3f}τ")
    print(f"  phi[0]: min={phi_min:.3e}, max={phi_max:.3e}, range={phi_range:.3e}, mean={phi_mean:.3e}")
    print(f"  R: min={R_array.min():.3e}, max={R_array.max():.3e}, mean={R_array.mean():.3e}, median={np.median(R_array):.3e}")
    print(f"  # R > 0.1: {np.sum(R_array > 0.1)},  # R > 1.0: {np.sum(R_array > 1.0)}")
    print()

print("="*80)
print("Analysis complete!")
