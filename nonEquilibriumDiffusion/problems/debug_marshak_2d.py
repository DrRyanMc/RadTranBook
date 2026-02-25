#!/usr/bin/env python3
"""
Debug script to understand the directional asymmetry issue
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from twoDFV import NonEquilibriumRadiationDiffusionSolver2D

# Physical constants
C_LIGHT = 2.998e1
A_RAD = 0.01372
RHO = 1.0

# Simple constant properties for debugging
def const_opacity(T):
    return 1.0

def const_heat(T):
    return 1.0 / RHO

def const_energy(T):
    return RHO * T

def inv_const_energy(e):
    return e / RHO

# Boundary conditions
def bc_dirichlet_1kev(phi, pos, boundary='left', geometry='cartesian'):
    T_bc = 1.0
    phi_bc = C_LIGHT * A_RAD * T_bc**4
    return 1.0, 0.0, phi_bc

def bc_zero_flux(phi, pos, boundary='left', geometry='cartesian'):
    return 0.0, 1.0, 0.0

print("="*80)
print("DEBUG: Simple 2D diffusion test - X vs Y direction")
print("="*80)

# Test 1: X-direction wave
print("\n--- TEST 1: X-direction wave ---")
boundary_funcs_x = {
    'left': bc_dirichlet_1kev,
    'right': bc_zero_flux,
    'bottom': bc_zero_flux,
    'top': bc_zero_flux
}

solver_x = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=0.0, x_max=1.0, nx_cells=10,
    y_min=0.0, y_max=0.2, ny_cells=3,
    geometry='cartesian',
    dt=0.1,
    theta=1.0,
    rosseland_opacity_func=const_opacity,
    planck_opacity_func=const_opacity,
    specific_heat_func=const_heat,
    material_energy_func=const_energy,
    inverse_material_energy_func=inv_const_energy,
    boundary_funcs=boundary_funcs_x
)

T_init = 0.1
phi_init = C_LIGHT * A_RAD * T_init**4
solver_x.set_initial_condition(phi_init=phi_init, T_init=T_init)

print(f"Initial: T_min={solver_x.T.min():.4f}, T_max={solver_x.T.max():.4f}")

# Take one step
solver_x.time_step(n_steps=1, verbose=True)

T_2d_x = solver_x.get_T_2d()
print(f"After 1 step: T_min={solver_x.T.min():.4f}, T_max={solver_x.T.max():.4f}")
print(f"T_2d_x shape: {T_2d_x.shape}")
print(f"T along x (middle y): {T_2d_x[:, 1]}")

# Test 2: Y-direction wave
print("\n--- TEST 2: Y-direction wave ---")
boundary_funcs_y = {
    'left': bc_zero_flux,
    'right': bc_zero_flux,
    'bottom': bc_dirichlet_1kev,
    'top': bc_zero_flux
}

solver_y = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=0.0, x_max=0.2, nx_cells=3,
    y_min=0.0, y_max=1.0, ny_cells=10,
    geometry='cartesian',
    dt=0.1,
    theta=1.0,
    rosseland_opacity_func=const_opacity,
    planck_opacity_func=const_opacity,
    specific_heat_func=const_heat,
    material_energy_func=const_energy,
    inverse_material_energy_func=inv_const_energy,
    boundary_funcs=boundary_funcs_y
)

solver_y.set_initial_condition(phi_init=phi_init, T_init=T_init)

print(f"Initial: T_min={solver_y.T.min():.4f}, T_max={solver_y.T.max():.4f}")

# Take one step
solver_y.time_step(n_steps=1, verbose=True)

T_2d_y = solver_y.get_T_2d()
print(f"After 1 step: T_min={solver_y.T.min():.4f}, T_max={solver_y.T.max():.4f}")
print(f"T_2d_y shape: {T_2d_y.shape}")
print(f"T along y (middle x): {T_2d_y[1, :]}")

# Compare
print("\n--- COMPARISON ---")
T_x_profile = T_2d_x[:, 1]  # Middle y
T_y_profile = T_2d_y[1, :]  # Middle x

# Interpolate to same grid for comparison
positions = np.linspace(0, 1, 10)
T_x_interp = np.interp(positions, solver_x.x_centers, T_x_profile)
T_y_interp = np.interp(positions, solver_y.y_centers, T_y_profile)

diff = T_x_interp - T_y_interp
print(f"Max difference: {np.max(np.abs(diff)):.6f}")
print(f"Mean difference: {np.mean(np.abs(diff)):.6f}")

# Plot
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.plot(solver_x.x_centers, T_x_profile, 'b-o', label='X-direction')
ax.set_xlabel('x')
ax.set_ylabel('T')
ax.set_title('X-direction wave')
ax.grid(True)
ax.legend()

ax = axes[1]
ax.plot(solver_y.y_centers, T_y_profile, 'r-s', label='Y-direction')
ax.set_xlabel('y')
ax.set_ylabel('T')
ax.set_title('Y-direction wave')
ax.grid(True)
ax.legend()

ax = axes[2]
ax.plot(positions, T_x_interp, 'b-o', label='X', alpha=0.7)
ax.plot(positions, T_y_interp, 'r-s', label='Y', alpha=0.7)
ax.plot(positions, diff, 'k--', label='Difference', linewidth=2)
ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
ax.set_xlabel('Position')
ax.set_ylabel('T')
ax.set_title('Comparison')
ax.grid(True)
ax.legend()

plt.tight_layout()
plt.savefig('debug_xy_comparison.png', dpi=150)
print("\nPlot saved as 'debug_xy_comparison.png'")

print("\n" + "="*80)
