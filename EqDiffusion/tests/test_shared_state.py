#!/usr/bin/env python3
"""
Test if modifying BDF2 assembly affects TR stage due to shared state
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
from oneDFV import RadiationDiffusionSolver, temperature_from_Er, A_RAD

n_cells = 50
r_max = 2.0
dt = 0.001
gamma = 2.0 - np.sqrt(2.0)

def constant_opacity(T): return 100.0
def cubic_cv(T): return 4 * 1.0 * A_RAD * T**3
def linear_material_energy(T): return 1.0 * A_RAD * T**4
def left_bc(Er, r): return (0.0, 1.0, 0.0)
def right_bc(Er, r): return (0.0, 1.0, 0.0)

print("="*70)
print("Test 1: TR stage BEFORE calling any BDF2 methods")
print("="*70)

solver1 = RadiationDiffusionSolver(
    n_cells=n_cells, r_max=r_max, dt=dt, theta=0.5, d=0,
    rosseland_opacity_func=constant_opacity,
    specific_heat_func=cubic_cv,
    material_energy_func=linear_material_energy,
    left_bc_func=left_bc,
    right_bc_func=right_bc
)

amplitude = A_RAD * 1.0**4
x0 = 1.0
sigma0 = 0.15
gaussian_Er = lambda r: amplitude * np.exp(-(r - x0)**2 / (2 * sigma0**2))
solver1.set_initial_condition(gaussian_Er)

Er_n = solver1.Er.copy()
solver1.dt = gamma * dt
solver1.theta = 0.5

print(f"About to call newton_step for TR stage...")
print(f"  solver1.dt = {solver1.dt}")
print(f"  solver1.theta = {solver1.theta}")

try:
    Er_intermediate = solver1.newton_step(Er_n, verbose=False)
    print(f"✓ TR stage succeeded!")
    print(f"  Er_intermediate max = {Er_intermediate.max():.6e}")
except Exception as e:
    print(f"✗ TR stage FAILED: {e}")

print()
print("="*70)
print("Test 2: Create NEW solver instance and test TR stage")
print("="*70)

solver2 = RadiationDiffusionSolver(
    n_cells=n_cells, r_max=r_max, dt=dt, theta=0.5, d=0,
    rosseland_opacity_func=constant_opacity,
    specific_heat_func=cubic_cv,
    material_energy_func=linear_material_energy,
    left_bc_func=left_bc,
    right_bc_func=right_bc
)
solver2.set_initial_condition(gaussian_Er)

Er_n2 = solver2.Er.copy()
solver2.dt = gamma * dt
solver2.theta = 0.5

print(f"About to call newton_step for TR stage on solver2...")
try:
    Er_intermediate2 = solver2.newton_step(Er_n2, verbose=False)
    print(f"✓ TR stage succeeded on solver2!")
    print(f"  Er_intermediate max = {Er_intermediate2.max():.6e}")
except Exception as e:
    print(f"✗ TR stage FAILED on solver2: {e}")

print()
print("="*70)
print("Test 3: Call assemble_system_bdf2 then test TR stage on SAME instance")
print("="*70)

# Reset solver1 for TR stage again
solver1.Er = Er_n.copy()
solver1.dt = gamma * dt
solver1.theta = 0.5

# But first, call assemble_system_bdf2 (don't actually use the result)
print("Calling assemble_system_bdf2 (just to see if it has side effects)...")
solver1.dt = dt  # Set to full dt for BDF2
dummy_A, dummy_rhs = solver1.assemble_system_bdf2(Er_n, Er_n, Er_n, gamma)
print(f"  assemble_system_bdf2 returned matrix shape {dummy_A.shape}")

# Now try TR stage again
solver1.dt = gamma * dt  # Reset to TR dt
solver1.theta = 0.5

print(f"About to call newton_step for TR stage AFTER calling assemble_system_bdf2...")
print(f"  solver1.dt = {solver1.dt}")
print(f"  solver1.theta = {solver1.theta}")

try:
    Er_intermediate3 = solver1.newton_step(Er_n, verbose=False)
    print(f"✓ TR stage still works after calling assemble_system_bdf2!")
    print(f"  Er_intermediate max = {Er_intermediate3.max():.6e}")
except Exception as e:
    print(f"✗ TR stage FAILED after calling assemble_system_bdf2: {e}")

print()
print("="*70)
print("CONCLUSION")
print("="*70)
print()
print("If all three tests pass, then assemble_system_bdf2 doesn't have")
print("problematic side effects. The issue with test_double_trapezoidal.py")
print("must be something else (possibly related to test structure or")
print("how instances are created in that specific test).")
