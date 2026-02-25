#!/usr/bin/env python3
"""
Test Marshak wave by initializing with self-similar solution at t=0.1 ns
and running to t=1.0 ns to verify the wave propagates correctly.

If there's a problem with the boundary conditions or solver, the wave
will not move from its initial position.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from twoDFV import NonEquilibriumRadiationDiffusionSolver2D

# Physical constants
C_LIGHT = 2.998e1  # cm/ns
A_RAD = 0.01372    # GJ/(cm³·keV⁴)
RHO = 1.0          # g/cm³

# =============================================================================
# MARSHAK WAVE MATERIAL PROPERTIES
# =============================================================================

def marshak_opacity(T):
    """Temperature-dependent opacity: σ = 300 * T^-3"""
    n = 3
    T_min = 0.01
    T_safe = np.maximum(T, T_min)
    return 300.0 * T_safe**(-n)

def marshak_rosseland_opacity(T, x, y):
    """Rosseland opacity"""
    return marshak_opacity(T)

def marshak_planck_opacity(T, x, y):
    """Planck opacity"""
    return marshak_opacity(T)

def marshak_specific_heat(T, x, y):
    """Specific heat: c_v = 0.3 GJ/(cm^3·keV)"""
    cv_volumetric = 0.3
    return cv_volumetric / RHO

def marshak_material_energy(T, x, y):
    """Material energy density"""
    cv_volumetric = 0.3
    return cv_volumetric * T

def marshak_inverse_material_energy(e, x, y):
    """Inverse: Given energy density, return temperature"""
    cv_volumetric = 0.3
    return e / cv_volumetric

# =============================================================================
# BOUNDARY CONDITIONS
# =============================================================================

def bc_blackbody_incoming(phi, pos, t, boundary='left', geometry='cartesian'):
    """Blackbody boundary condition at T = 1.0 keV"""
    T_bc = 1.0
    phi_bc = C_LIGHT * A_RAD * T_bc**4
    return 1.0, 0.0, phi_bc

def bc_zero_flux(phi, pos, t, boundary='left', geometry='cartesian'):
    """Zero flux (reflecting) boundary condition"""
    return 0.0, 1.0, 0.0

# =============================================================================
# SELF-SIMILAR SOLUTION
# =============================================================================

def self_similar_solution(r, t):
    """
    Self-similar solution for Marshak wave with n=3
    
    Parameters:
    -----------
    r : array
        Position (cm)
    t : float
        Time (ns)
    
    Returns:
    --------
    T : array
        Temperature (keV)
    """
    # Self-similar solution parameters for n=3
    xi_max = 1.11305
    omega = 0.05989
    K_const = 8*A_RAD*C_LIGHT/((4+3)*3*300*RHO*0.3)
    
    # Self-similar variable
    xi = r / np.sqrt(K_const * t)
    
    # Temperature profile (with minimum background temperature)
    T_min = 0.01  # Minimum temperature (keV)
    T = np.zeros_like(r)
    mask = xi < xi_max
    T[mask] = np.power((1 - xi[mask]/xi_max)*(1+omega*xi[mask]/xi_max), 1/6)
    T[~mask] = T_min  # Set background temperature beyond wave front
    
    return T

# =============================================================================
# MAIN TEST
# =============================================================================

print("=" * 80)
print("MARSHAK WAVE SELF-SIMILAR SOLUTION TEST")
print("=" * 80)
print()
print("This test initializes the Marshak wave with the self-similar solution")
print("at t = 0.1 ns and runs to t = 1.0 ns.")
print()
print("If the solver is working correctly, the solution should match the")
print("self-similar solution at t = 1.0 ns.")
print("=" * 80)
print()

# Grid parameters
nx = 100
ny = 1
x_min, x_max = 0.0, 0.5  # cm
y_min, y_max = 0.0, 0.1  # cm (thin strip)

# Time parameters
t_init = 0.1  # Initial time (ns)
t_final = 5.0  # Final time (ns)
dt = 0.02  # Time step (ns)

# Create solver
print(f"Initializing solver with {nx} × {ny} = {nx*ny} cells...")

# Boundary conditions
boundary_funcs = {
    'left': bc_blackbody_incoming,
    'right': bc_zero_flux,
    'bottom': bc_zero_flux,
    'top': bc_zero_flux
}

solver = NonEquilibriumRadiationDiffusionSolver2D(
    nx_cells=nx, ny_cells=ny,
    x_min=x_min, x_max=x_max,
    y_min=y_min, y_max=y_max,
    rosseland_opacity_func=marshak_rosseland_opacity,
    planck_opacity_func=marshak_planck_opacity,
    specific_heat_func=marshak_specific_heat,
    material_energy_func=marshak_material_energy,
    inverse_material_energy_func=marshak_inverse_material_energy,
    boundary_funcs=boundary_funcs,
    dt=dt,
    theta=1.0
)

# Initialize with self-similar solution at t = t_init
print(f"\nInitializing with self-similar solution at t = {t_init} ns...")

# Create callable initial conditions
def T_init_func(x, y):
    return self_similar_solution(np.array([x]), t_init)[0]

def phi_init_func(x, y):
    T = self_similar_solution(np.array([x]), t_init)[0]
    return C_LIGHT * A_RAD * T**4

solver.set_initial_condition(phi_init=phi_init_func, T_init=T_init_func)
solver.current_time = t_init

# Get initial conditions for plotting
x_centers = solver.x_centers
T_init_array = self_similar_solution(x_centers, t_init)
phi_init_array = C_LIGHT * A_RAD * T_init_array**4

print(f"  Initial T range: [{T_init_array.min():.4f}, {T_init_array.max():.4f}] keV")
print(f"  Initial phi range: [{phi_init_array.min():.6e}, {phi_init_array.max():.6e}] GJ/cm²")

# Store initial profile
T_initial = solver.get_T_2d()[:, 0].copy()

# Run to t_final
print(f"\nRunning from t = {t_init} ns to t = {t_final} ns...")
n_steps = 0
while solver.current_time < t_final:
    solver.time_step(n_steps=1, verbose=False)
    n_steps += 1
    
    if n_steps % 10 == 0:
        print(f"  Step {n_steps}, t = {solver.current_time:.4f} ns")

print(f"\nCompleted {n_steps} steps")
print(f"Final time: {solver.current_time:.4f} ns")

# Get final solution
T_final = solver.get_T_2d()[:, 0]

# Compute self-similar solution at t_final
T_ss_initial = self_similar_solution(x_centers, t_init)
T_ss_final = self_similar_solution(x_centers, t_final)

# Compute errors
error_abs = np.abs(T_final - T_ss_final)
error_rel = error_abs / (T_ss_final + 1e-10)
max_error = np.max(error_abs)
max_rel_error = np.max(error_rel)

print(f"\nFinal T range: [{T_final.min():.4f}, {T_final.max():.4f}] keV")
print(f"Self-similar T range: [{T_ss_final.min():.4f}, {T_ss_final.max():.4f}] keV")
print(f"\nMaximum absolute error: {max_error:.4e} keV")
print(f"Maximum relative error: {max_rel_error:.4e}")

# =============================================================================
# PLOTTING
# =============================================================================

print("\nGenerating comparison plot...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Initial condition
ax = axes[0, 0]
ax.plot(x_centers, T_ss_initial, 'k-', linewidth=2, label='Self-similar t=0.1 ns')
ax.plot(x_centers, T_initial, 'r--', linewidth=1.5, label='Initial condition')
ax.set_xlabel('Position x (cm)', fontsize=11)
ax.set_ylabel('T (keV)', fontsize=11)
ax.set_title(f'Initial Condition at t = {t_init} ns', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Plot 2: Final solution comparison
ax = axes[0, 1]
ax.plot(x_centers, T_ss_final, 'k-', linewidth=2, label='Self-similar t=1.0 ns')
ax.plot(x_centers, T_final, 'b--', linewidth=1.5, label='Computed t=1.0 ns')
ax.set_xlabel('Position x (cm)', fontsize=11)
ax.set_ylabel('T (keV)', fontsize=11)
ax.set_title(f'Final Solution at t = {t_final} ns', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)

# Plot 3: Evolution
ax = axes[1, 0]
ax.plot(x_centers, T_ss_initial, 'k-', linewidth=2, alpha=0.5, label=f'Self-similar t={t_init} ns')
ax.plot(x_centers, T_initial, 'r--', linewidth=1.5, alpha=0.7, label=f'Initial (computed) t={t_init} ns')
ax.plot(x_centers, T_ss_final, 'k-', linewidth=2, label=f'Self-similar t={t_final} ns')
ax.plot(x_centers, T_final, 'b--', linewidth=1.5, label=f'Final (computed) t={t_final} ns')
ax.set_xlabel('Position x (cm)', fontsize=11)
ax.set_ylabel('T (keV)', fontsize=11)
ax.set_title('Evolution: Initial → Final', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=9)

# Plot 4: Error
ax = axes[1, 1]
ax.semilogy(x_centers, error_abs + 1e-10, 'r-', linewidth=2, label='Absolute error')
ax.set_xlabel('Position x (cm)', fontsize=11)
ax.set_ylabel('|T_computed - T_selfsimilar| (keV)', fontsize=11)
ax.set_title('Error at Final Time', fontsize=12, fontweight='bold')
ax.grid(True, alpha=0.3, which='both')
ax.legend(fontsize=10)

plt.tight_layout()
plt.savefig('test_marshak_selfsimilar.png', dpi=150, bbox_inches='tight')
print("Plot saved as: test_marshak_selfsimilar.png")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("TEST SUMMARY")
print("=" * 80)

# Check if wave moved
wave_front_initial = np.max(np.where(T_initial > 0.1)[0]) if np.any(T_initial > 0.1) else 0
wave_front_final = np.max(np.where(T_final > 0.1)[0]) if np.any(T_final > 0.1) else 0
wave_front_expected = np.max(np.where(T_ss_final > 0.1)[0]) if np.any(T_ss_final > 0.1) else 0

print(f"Wave front position (index where T > 0.1 keV):")
print(f"  Initial (t={t_init} ns):  cell {wave_front_initial} (x = {x_centers[wave_front_initial]:.4f} cm)")
print(f"  Final (t={t_final} ns):    cell {wave_front_final} (x = {x_centers[wave_front_final]:.4f} cm)")
print(f"  Expected (self-similar): cell {wave_front_expected} (x = {x_centers[wave_front_expected]:.4f} cm)")
print()

if wave_front_final > wave_front_initial + 5:
    print("✓ PASS: Wave has propagated forward")
else:
    print("✗ FAIL: Wave has NOT moved significantly")
    print("  This suggests a problem with the boundary conditions or solver.")

if max_rel_error < 0.05:
    print("✓ PASS: Solution matches self-similar solution (< 5% error)")
elif max_rel_error < 0.2:
    print("⚠ WARNING: Solution has moderate error (5-20%)")
else:
    print("✗ FAIL: Solution does not match self-similar solution (> 20% error)")

print("=" * 80)
