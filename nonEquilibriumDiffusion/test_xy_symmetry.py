"""
Detailed comparison: X-direction vs Y-direction for simple diffusion
"""
import sys
sys.path.append('..')
import numpy as np
from twoDFV import NonEquilibriumRadiationDiffusionSolver2D

# Physical constants
A_RAD = 0.01372
C_LIGHT = 29.9792458
RHO = 1.0

print("="*70)
print("Testing X vs Y symmetry for homogeneous diffusion")
print("="*70)

# Simple constant material properties
def rosseland_opacity(T, x, y):
    return 1.0  # cm^-1 (constant)

def planck_opacity(T, x, y):
    return 1.0

def specific_heat(T, x, y):
    return 0.01 / RHO

def material_energy(T, x, y):
    return 0.01 * T

def inverse_material_energy(e, x, y):
    return e / 0.01

# Hot boundary on one side, zero flux on others
def bc_hot_x(phi, pos, t, boundary='left', geometry='cartesian'):
    T_bc = 1.0
    phi_bc = C_LIGHT * A_RAD * T_bc**4
    return 1.0, 0.0, phi_bc

def bc_hot_y(phi, pos, t, boundary='bottom', geometry='cartesian'):
    T_bc = 1.0
    phi_bc = C_LIGHT * A_RAD * T_bc**4
    return 1.0, 0.0, phi_bc

def bc_zero_flux(phi, pos, t, boundary='right', geometry='cartesian'):
    return 0.0, 1.0, 0.0

# Test 1: Wave in X-direction (10x1 grid)
print("\n--- Test 1: X-direction wave (10×1 grid) ---")
boundary_funcs_x = {
    'left': bc_hot_x,
    'right': bc_zero_flux,
    'bottom': bc_zero_flux,
    'top': bc_zero_flux
}

solver_x = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=0.0, x_max=1.0, nx_cells=10,
    y_min=0.0, y_max=0.1, ny_cells=1,
    geometry='cartesian', dt=0.01,
    rosseland_opacity_func=rosseland_opacity,
    planck_opacity_func=planck_opacity,
    specific_heat_func=specific_heat,
    material_energy_func=material_energy,
    inverse_material_energy_func=inverse_material_energy,
    boundary_funcs=boundary_funcs_x,
    theta=1.0
)

T_init = 0.01
phi_init = A_RAD * T_init**4 * C_LIGHT
solver_x.set_initial_condition(phi_init=phi_init, T_init=T_init)

solver_x.time_step(n_steps=1, verbose=False)
T_x = solver_x.get_T_2d()[:, 0]  # Extract 1D profile

print(f"After 1 time step:")
print(f"  T range: [{np.min(T_x):.6f}, {np.max(T_x):.6f}] keV")

# Test 2: Wave in Y-direction (1x10 grid)
print("\n--- Test 2: Y-direction wave (1×10 grid) ---")
boundary_funcs_y = {
    'left': bc_zero_flux,
    'right': bc_zero_flux,
    'bottom': bc_hot_y,
    'top': bc_zero_flux
}

solver_y = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=0.0, x_max=0.1, nx_cells=1,
    y_min=0.0, y_max=1.0, ny_cells=10,
    geometry='cartesian', dt=0.01,
    rosseland_opacity_func=rosseland_opacity,
    planck_opacity_func=planck_opacity,
    specific_heat_func=specific_heat,
    material_energy_func=material_energy,
    inverse_material_energy_func=inverse_material_energy,
    boundary_funcs=boundary_funcs_y,
    theta=1.0
)

solver_y.set_initial_condition(phi_init=phi_init, T_init=T_init)

solver_y.time_step(n_steps=1, verbose=False)
T_y = solver_y.get_T_2d()[0, :]  # Extract 1D profile

print(f"After 1 time step:")
print(f"  T range: [{np.min(T_y):.6f}, {np.max(T_y):.6f}] keV")

# Compare
print("\n--- Comparison ---")
print(f"X-direction T: {T_x}")
print(f"Y-direction T: {T_y}")
print(f"\nDifference: {T_x - T_y}")
print(f"Max absolute diff: {np.max(np.abs(T_x - T_y)):.6e}")
print(f"Max relative diff: {np.max(np.abs((T_x - T_y)/T_x)):.6e}")

if np.max(np.abs(T_x - T_y)) < 1e-10:
    print("\n✓ X and Y directions match! (symmetric)")
else:
    print("\n✗ WARNING: X and Y directions differ! (asymmetric)")

print("="*70)
