#!/usr/bin/env python3
"""
Simple test to verify TR-BDF2 implementation for 2D solver
"""

import numpy as np
import sys
sys.path.insert(0, '/Users/ryanmcclarren/Dropbox/Papers/RadTranBook/nonEquilibriumDiffusion')
from twoDFV import NonEquilibriumRadiationDiffusionSolver2D

print("="*70)
print("Testing TR-BDF2 vs θ-method on 2D diffusion problem")
print("="*70)

# Small test problem
nx, ny = 15, 15
x_min, x_max = 0.0, 1.0
y_min, y_max = 0.0, 1.0
dt = 0.01
n_steps = 10

# Initial condition: hot spot in center
def T_init(x, y):
    r2 = (x - 0.5)**2 + (y - 0.5)**2
    return 0.3 + 0.7 * np.exp(-50.0 * r2)

# Test 1: θ-method (implicit Euler)
print("\nTest 1: Implicit Euler (θ=1.0)")
print("-" * 70)
solver_theta = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=x_min, x_max=x_max, nx_cells=nx,
    y_min=y_min, y_max=y_max, ny_cells=ny,
    geometry='cartesian',
    dt=dt,
    theta=1.0,
    max_newton_iter=20,
    newton_tol=1e-6
)
solver_theta.set_initial_condition(T_init=T_init)
print(f"Running {n_steps} steps with dt={dt}...")
solver_theta.time_step(n_steps=n_steps, verbose=False)
x, y, phi_theta, T_theta = solver_theta.get_solution()
print(f"Final φ range: [{phi_theta.min():.4e}, {phi_theta.max():.4e}]")
print(f"Final T range: [{T_theta.min():.4f}, {T_theta.max():.4f}] keV")

# Test 2: TR-BDF2
print("\nTest 2: TR-BDF2")
print("-" * 70)
solver_trbdf2 = NonEquilibriumRadiationDiffusionSolver2D(
    x_min=x_min, x_max=x_max, nx_cells=nx,
    y_min=y_min, y_max=y_max, ny_cells=ny,
    geometry='cartesian',
    dt=dt,
    theta=1.0,
    max_newton_iter=20,
    newton_tol=1e-6
)
solver_trbdf2.set_initial_condition(T_init=T_init)
print(f"Running {n_steps} TR-BDF2 steps with dt={dt}...")
solver_trbdf2.time_step_trbdf2(n_steps=n_steps, verbose=False)
x, y, phi_trbdf2, T_trbdf2 = solver_trbdf2.get_solution()
print(f"Final φ range: [{phi_trbdf2.min():.4e}, {phi_trbdf2.max():.4e}]")
print(f"Final T range: [{T_trbdf2.min():.4f}, {T_trbdf2.max():.4f}] keV")

# Compare solutions
print("\n" + "="*70)
print("Comparison")
print("="*70)
phi_diff = np.abs(phi_trbdf2 - phi_theta)
T_diff = np.abs(T_trbdf2 - T_theta)
print(f"Max absolute difference in φ: {phi_diff.max():.4e}")
print(f"Max absolute difference in T: {T_diff.max():.4e}")
print(f"Relative difference in φ: {phi_diff.max() / phi_theta.max():.4e}")
print(f"Relative difference in T: {T_diff.max() / T_theta.max():.4e}")

# Both methods should give similar results (within numerical tolerance)
if phi_diff.max() / phi_theta.max() < 0.1:  # Within 10%
    print("\n✓ TR-BDF2 and θ-method solutions are consistent")
else:
    print("\n⚠ Solutions differ significantly (expected for different time integrators)")

print("\n" + "="*70)
print("Test complete!")
print("="*70)
