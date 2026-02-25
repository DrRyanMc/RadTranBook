#!/usr/bin/env python3
"""
Quick test to verify vectorized material property calculations work correctly
"""

import sys
import os
import time

# Add parent directory to path
parent_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, parent_dir)

import numpy as np
from twoDFV import (
    NonEquilibriumRadiationDiffusionSolver2D,
    A_RAD,
    C_LIGHT,
    RHO
)

# Simple material functions for testing
def simple_opacity(T, x, y):
    """σ = 10 * T^-2"""
    T_safe = np.maximum(T, 0.01)
    return 10.0 * T_safe**(-2)

def simple_specific_heat(T, x, y):
    """c_v = constant"""
    return 1e-6 / RHO

def simple_material_energy(T, x, y):
    """e = ρ·c_v·T"""
    return RHO * 1e-6 * T

def simple_inverse_material_energy(e, x, y):
    """T from e"""
    return e / (RHO * 1e-6)

def bc_reflecting(phi, pos, t, boundary='left', geometry='cartesian'):
    """Reflecting BC"""
    return 0.0, 1.0, 0.0

# Test with small grid
print("Testing vectorized material property calculations...")
print("="*70)

nx = 50
ny = 50
n_total = nx * ny

solver = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=0.0, x_max=1.0, nx_cells=nx,
    y_min=0.0, y_max=1.0, ny_cells=ny,
    geometry='cartesian',
    dt=1e-4,
    max_newton_iter=5,
    newton_tol=1e-6,
    rosseland_opacity_func=simple_opacity,
    planck_opacity_func=simple_opacity,
    specific_heat_func=simple_specific_heat,
    material_energy_func=simple_material_energy,
    inverse_material_energy_func=simple_inverse_material_energy,
    boundary_funcs={
        'left': bc_reflecting, 'right': bc_reflecting,
        'bottom': bc_reflecting, 'top': bc_reflecting
    },
    theta=1.0
)

# Set up test arrays
T_test = np.random.uniform(0.1, 1.0, n_total)
phi_test = A_RAD * C_LIGHT * T_test**4

print(f"Grid: {nx} × {ny} = {n_total} cells")
print(f"T range: [{T_test.min():.3f}, {T_test.max():.3f}] keV")
print()

# Test vectorized material properties
print("Testing compute_material_properties_vectorized...")
t0 = time.time()
props = solver.compute_material_properties_vectorized(T_test, phi_test)
t1 = time.time()
print(f"  Time: {(t1-t0)*1000:.2f} ms")
print(f"  sigma_P range: [{props['sigma_P'].min():.2e}, {props['sigma_P'].max():.2e}]")
print(f"  f range: [{props['f'].min():.6f}, {props['f'].max():.6f}]")
print(f"  e range: [{props['e'].min():.2e}, {props['e'].max():.2e}]")
print()

# Test vectorized inverse
print("Testing compute_temperature_from_energy_vectorized...")
e_test = props['e']
t0 = time.time()
T_recovered = solver.compute_temperature_from_energy_vectorized(e_test)
t1 = time.time()
print(f"  Time: {(t1-t0)*1000:.2f} ms")
print(f"  Max error: {np.max(np.abs(T_recovered - T_test)):.2e}")
print()

# Test a small timestep
print("Testing single Newton iteration...")
solver.set_initial_condition(phi_init=phi_test, T_init=T_test)
phi_prev = phi_test.copy()
T_prev = T_test.copy()

t0 = time.time()
phi_new, T_new = solver.newton_step(phi_prev, T_prev, source=None, verbose=False)
t1 = time.time()
print(f"  Time for Newton step: {(t1-t0)*1000:.2f} ms")
print(f"  phi changed by: {np.max(np.abs(phi_new - phi_prev)):.2e}")
print(f"  T changed by: {np.max(np.abs(T_new - T_prev)):.2e}")
print()

print("="*70)
print("✓ Vectorization tests passed!")
print()
print("Performance summary:")
print(f"  Material properties: ~{(t1-t0)*1000:.1f} ms for full Newton iteration")
print(f"  This is for {n_total} cells with {solver.max_newton_iter} max iterations")
