#!/usr/bin/env python3
"""
Test timing breakdown for Zel'dovich wave problem
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

# Material properties
def zeldovich_rosseland_opacity(T, x, y):
    """σ_R = 300 * T^-3"""
    T_safe = np.maximum(T, 0.001)
    return 300.0 * T_safe**(-3)

def zeldovich_planck_opacity(T, x, y):
    return zeldovich_rosseland_opacity(T, x, y)

def zeldovich_specific_heat(T, x, y):
    """c_v = 3e-6 GJ/(cm^3·keV)"""
    return 3e-6 / RHO

def zeldovich_material_energy(T, x, y):
    """e = ρ·c_v·T"""
    return RHO * 3e-6 * T

def zeldovich_inverse_material_energy(e, x, y):
    """T from e"""
    return e / (RHO * 3e-6)

def bc_reflecting(phi, pos, t, boundary='left', geometry='cylindrical'):
    return 0.0, 1.0, 0.0

print("="*80)
print("TIMING BREAKDOWN TEST - Zel'dovich Wave")
print("="*80)

# Domain
r_min, r_max = 0.0, 3.0
z_min, z_max = 0.0, 3.0

# Grid
nr, nz = 50, 50
print(f"\nGrid: {nr} × {nz} = {nr*nz} cells")

# Time stepping
dt = 1e-4

# Boundary conditions
boundary_funcs = {
    'left': bc_reflecting,
    'right': bc_reflecting,
    'bottom': bc_reflecting,
    'top': bc_reflecting
}

# Create solver
solver = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=r_min, x_max=r_max, nx_cells=nr,
    y_min=z_min, y_max=z_max, ny_cells=nz,
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

# Initial condition: hot spot at origin
T_cold, T_hot = 0.1, 1.0
T_2d = np.zeros((nr, nz))
phi_2d = np.zeros((nr, nz))

for i in range(nr):
    for j in range(nz):
        r = solver.x_centers[i]
        z = solver.y_centers[j]
        rho = np.sqrt(r**2 + z**2)
        
        if rho < 0.3:
            T_2d[i, j] = T_hot
        else:
            T_2d[i, j] = T_cold
        
        phi_2d[i, j] = A_RAD * C_LIGHT * T_2d[i, j]**4

solver.set_initial_condition(phi_init=phi_2d, T_init=T_2d)

print(f"Initial condition: T ∈ [{T_2d.min():.2f}, {T_2d.max():.2f}] keV\n")

# Run with timing enabled
print("="*80)
print("RUNNING TIMESTEP WITH DETAILED TIMING")
print("="*80)

t_start_total = time.time()
solver.time_step(n_steps=1, source=None, verbose=True, timing=True)
t_end_total = time.time()

print("\n" + "="*80)
print(f"Total wallclock time: {t_end_total - t_start_total:.3f}s")
print("="*80)
