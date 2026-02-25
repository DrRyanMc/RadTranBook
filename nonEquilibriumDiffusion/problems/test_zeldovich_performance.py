#!/usr/bin/env python3
"""
Quick Zel'dovich performance test - just run a few steps to measure speed
"""

import sys
import os
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import numpy as np
from twoDFV import (
    NonEquilibriumRadiationDiffusionSolver2D,
    A_RAD,
    C_LIGHT,
    RHO
)

# =============================================================================
# ZELDOVICH WAVE MATERIAL PROPERTIES
# =============================================================================

def zeldovich_rosseland_opacity(T, x, y):
    """σ_R = 300 * T^-3"""
    n = 3
    T_min = 0.001
    T_safe = np.maximum(T, T_min)
    return 300.0 * T_safe**(-n)

def zeldovich_planck_opacity(T, x, y):
    """σ_P = 300 * T^-3"""
    return zeldovich_rosseland_opacity(T, x, y)

def zeldovich_specific_heat(T, x, y):
    """c_v = 3e-6 GJ/(cm^3·keV)"""
    cv_volumetric = 3e-6
    return cv_volumetric / RHO

def zeldovich_material_energy(T, x, y):
    """e = ρ·c_v·T"""
    cv_volumetric = 3e-6
    return RHO * cv_volumetric * T

def zeldovich_inverse_material_energy(e, x, y):
    """T from e"""
    cv_volumetric = 3e-6
    return e / (RHO * cv_volumetric)

def bc_reflecting(phi, pos, t, boundary='left', geometry='cylindrical'):
    """Reflecting boundary: zero flux (∇φ · n = 0)"""
    return 0.0, 1.0, 0.0

# =============================================================================
# MAIN TEST
# =============================================================================

print("="*80)
print("Zel'dovich Wave Performance Test")
print("="*80)

# Domain setup
r_min = 0.0
r_max = 3.0
z_min = 0.0
z_max = 3.0

# Grid resolution
nr_cells = 50
nz_cells = 50

# Time stepping
dt = 1e-4
n_test_steps = 3

# Boundary conditions
boundary_funcs = {
    'left': bc_reflecting,
    'right': bc_reflecting,
    'bottom': bc_reflecting,
    'top': bc_reflecting
}

# Create solver
print(f"\nInitializing solver: {nr_cells} × {nz_cells} = {nr_cells*nz_cells} cells")
solver = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=r_min,
    x_max=r_max,
    nx_cells=nr_cells,
    y_min=z_min,
    y_max=z_max,
    ny_cells=nz_cells,
    geometry='cylindrical',
    dt=dt,
    max_newton_iter=100,
    newton_tol=1e-6,
    rosseland_opacity_func=zeldovich_rosseland_opacity,
    planck_opacity_func=zeldovich_planck_opacity,
    specific_heat_func=zeldovich_specific_heat,
    material_energy_func=zeldovich_material_energy,
    inverse_material_energy_func=zeldovich_inverse_material_energy,
    boundary_funcs=boundary_funcs,
    theta=1.0
)

# Simple initial condition
T_cold = 0.1  # keV
T_hot = 1.0   # keV

T_2d = np.zeros((nr_cells, nz_cells))
phi_2d = np.zeros((nr_cells, nz_cells))

for i in range(nr_cells):
    for j in range(nz_cells):
        r = solver.x_centers[i]
        z = solver.y_centers[j]
        rho = np.sqrt(r**2 + z**2)
        
        if rho < 0.3:
            T_2d[i, j] = T_hot
        else:
            T_2d[i, j] = T_cold
        
        phi_2d[i, j] = A_RAD * C_LIGHT * T_2d[i, j]**4

solver.set_initial_condition(phi_init=phi_2d, T_init=T_2d)

print(f"Initial condition: T ∈ [{T_2d.min():.2f}, {T_2d.max():.2f}] keV")
print(f"\nRunning {n_test_steps} time steps with dt={dt:.1e} ns...")
print("="*80)

# Run test timesteps
step_times = []
for step in range(n_test_steps):
    t_start = time.time()
    
    phi_prev = solver.phi.copy()
    T_prev = solver.T.copy()
    solver.phi, solver.T = solver.newton_step(phi_prev, T_prev, source=None, verbose=True)
    solver.phi_old = phi_prev
    solver.T_old = T_prev
    solver.current_time += dt
    
    t_end = time.time()
    step_time = t_end - t_start
    step_times.append(step_time)
    
    print(f"  Step {step+1} completed in {step_time:.3f} seconds")
    print()

print("="*80)
print("Performance Summary:")
print(f"  Average time per step: {np.mean(step_times):.3f} ± {np.std(step_times):.3f} seconds")
print(f"  Grid size: {nr_cells*nz_cells} cells")
print(f"  Time per cell per step: {1000*np.mean(step_times)/(nr_cells*nz_cells):.2f} ms")
print("="*80)
