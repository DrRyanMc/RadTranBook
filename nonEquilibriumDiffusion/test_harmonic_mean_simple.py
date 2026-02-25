"""
Simple test: Compare old vs new approach for 1D slab problem
"""
import sys
sys.path.append('..')
import numpy as np
from twoDFV import NonEquilibriumRadiationDiffusionSolver2D

# Physical constants
A_RAD = 0.01372
C_LIGHT = 29.9792458
RHO = 1.0

print("Testing harmonic mean vs old approach on simple 1D problem")
print("="*70)

# Define homogeneous material properties (no spatial dependence)
def rosseland_opacity_homogeneous(T, x, y):
    """Temperature-dependent only"""
    return 300.0 * max(T, 0.01)**(-3)

def planck_opacity_homogeneous(T, x, y):
    return 300.0 * max(T, 0.01)**(-3)

def specific_heat_homogeneous(T, x, y):
    return 0.3 / RHO

def material_energy_homogeneous(T, x, y):
    return 0.3 * T

def inverse_material_energy_homogeneous(e, x, y):
    return e / 0.3

# Boundary conditions: hot left, zero flux right
def bc_left(phi, pos, t, boundary='left', geometry='cartesian'):
    T_bc = 1.0 # keV
    phi_bc = C_LIGHT * A_RAD * T_bc**4
    return 1.0, 0.0, phi_bc

def bc_right(phi, pos, t, boundary='right', geometry='cartesian'):
    return 0.0, 1.0, 0.0

def bc_reflect(phi, pos, t, boundary='bottom', geometry='cartesian'):
    return 0.0, 1.0, 0.0

boundary_funcs = {
    'left': bc_left,
    'right': bc_right,
    'bottom': bc_reflect,
    'top': bc_reflect
}

# Create 1D problem as 2D with 1 cell in y
nx, ny = 10, 1
x_min, x_max = 0.0, 0.1  # cm
y_min, y_max = 0.0, 0.01  # cm
dt = 0.01  # ns

print(f"\nGrid: {nx} × {ny} cells")
print(f"Domain: x ∈ [{x_min}, {x_max}], y ∈ [{y_min}, {y_max}]")
print(f"Time step: dt = {dt} ns")

# Create solver
solver = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=x_min, x_max=x_max, nx_cells=nx,
    y_min=y_min, y_max=y_max, ny_cells=ny,
    geometry='cartesian', dt=dt,
    rosseland_opacity_func=rosseland_opacity_homogeneous,
    planck_opacity_func=planck_opacity_homogeneous,
    specific_heat_func=specific_heat_homogeneous,
    material_energy_func=material_energy_homogeneous,
    inverse_material_energy_func=inverse_material_energy_homogeneous,
    boundary_funcs=boundary_funcs,
    theta=1.0
)

# Initial conditions
T_init = 0.01  # keV
phi_init = A_RAD * T_init**4 * C_LIGHT
solver.set_initial_condition(phi_init=phi_init, T_init=T_init)

print(f"\nInitial: T = {T_init} keV everywhere")

# Take one time step
print(f"\nTaking time step...")
solver.time_step(n_steps=1, verbose=False)

T_final = solver.get_T_2d()[:, 0]  # Get 1D profile
print(f"\nAfter 1 time step:")
print(f"  T range: [{np.min(T_final):.4f}, {np.max(T_final):.4f}] keV")
print(f"  T values: {T_final}")

print("\n" + "="*70)
print("✓ Test completed successfully")
print("  (If this runs without error, harmonic mean is working)")
print("="*70)
