#!/usr/bin/env python3
"""
Debug exactly where the NaN appears with small k and fine mesh
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
from oneDFV import RadiationDiffusionSolver, temperature_from_Er, A_RAD, C_LIGHT

n_cells = 200
r_max = 2.0
dt = 0.001
sigma_R = 100.0
k_coupling = 0.01
x0 = 1.0
sigma0 = 0.15
amplitude = A_RAD * 1.0**4
gamma = 2.0 - np.sqrt(2.0)

def constant_opacity(T): return sigma_R
def cubic_cv(T): return 4 * k_coupling * A_RAD * T**3
def linear_material_energy(T): return k_coupling * A_RAD * T**4
def left_bc(r, t): return (0.0, 1.0, 0.0)
def right_bc(r, t): return (0.0, 1.0, 0.0)
def gaussian_Er(r): return amplitude * np.exp(-(r - x0)**2 / (2 * sigma0**2))

solver = RadiationDiffusionSolver(
    n_cells=n_cells, r_max=r_max, dt=dt, theta=0.5, d=0,
    rosseland_opacity_func=constant_opacity,
    specific_heat_func=cubic_cv,
    material_energy_func=linear_material_energy,
    left_bc_func=left_bc,
    right_bc_func=right_bc
)
solver.set_initial_condition(gaussian_Er)

print("="*70)
print("Debugging NaN generation with k=0.01, n_cells=200")
print("="*70)
print()

# Step 0 - should work
print("Step 0:")
Er_n = solver.Er.copy()
print(f"  Initial: min={Er_n.min():.6e}, max={Er_n.max():.6e}")

# TR stage
solver.dt = gamma * dt
solver.theta = 0.5
Er_intermediate = solver.newton_step(Er_n, verbose=False)
print(f"  After TR: min={Er_intermediate.min():.6e}, max={Er_intermediate.max():.6e}")

# BDF2 stage
solver.dt = dt
A_bdf2, rhs_bdf2 = solver.assemble_system_bdf2(Er_intermediate, Er_n, Er_intermediate, gamma)

print(f"  BDF2 matrix diagonal: min={A_bdf2[1,:].min():.6e}, max={A_bdf2[1,:].max():.6e}")
print(f"  BDF2 RHS: min={rhs_bdf2.min():.6e}, max={rhs_bdf2.max():.6e}")
print(f"  Matrix has NaN: {np.any(~np.isfinite(A_bdf2))}")
print(f"  RHS has NaN: {np.any(~np.isfinite(rhs_bdf2))}")

if not np.any(~np.isfinite(A_bdf2)) and not np.any(~np.isfinite(rhs_bdf2)):
    Er_new = solver.newton_step_bdf2(Er_n, Er_intermediate, gamma, verbose=False)
    print(f"  After BDF2: min={Er_new.min():.6e}, max={Er_new.max():.6e}")
    solver.Er = Er_new.copy()
    solver.Er_old = Er_n.copy()
    print("  ✓ Step 0 succeeded")
else:
    print("  ✗ Step 0 already has NaN!")
    import sys; sys.exit(1)

print()

# Step 1 - should work
print("Step 1:")
Er_n = solver.Er.copy()
print(f"  Initial: min={Er_n.min():.6e}, max={Er_n.max():.6e}")

solver.dt = gamma * dt
solver.theta = 0.5
Er_intermediate = solver.newton_step(Er_n, verbose=False)
print(f"  After TR: min={Er_intermediate.min():.6e}, max={Er_intermediate.max():.6e}")

solver.dt = dt
A_bdf2, rhs_bdf2 = solver.assemble_system_bdf2(Er_intermediate, Er_n, Er_intermediate, gamma)

print(f"  BDF2 matrix diagonal: min={A_bdf2[1,:].min():.6e}, max={A_bdf2[1,:].max():.6e}")
print(f"  BDF2 RHS: min={rhs_bdf2.min():.6e}, max={rhs_bdf2.max():.6e}")
print(f"  Matrix has NaN: {np.any(~np.isfinite(A_bdf2))}")
print(f"  RHS has NaN: {np.any(~np.isfinite(rhs_bdf2))}")

if not np.any(~np.isfinite(A_bdf2)) and not np.any(~np.isfinite(rhs_bdf2)):
    Er_new = solver.newton_step_bdf2(Er_n, Er_intermediate, gamma, verbose=False)
    print(f"  After BDF2: min={Er_new.min():.6e}, max={Er_new.max():.6e}")
    
    # Check for small negative values
    if np.any(Er_new < 0):
        neg_idx = np.where(Er_new < 0)[0]
        print(f"  ⚠ WARNING: {len(neg_idx)} negative Er values!")
        print(f"    Min Er = {Er_new.min():.6e}")
        print(f"    Locations: {neg_idx[:10]}")
    
    solver.Er = Er_new.copy()
    solver.Er_old = Er_n.copy()
    print("  ✓ Step 1 succeeded")
else:
    print("  ✗ Step 1 has NaN!")
    import sys; sys.exit(1)

print()

# Step 2 - this is where it typically fails
print("Step 2:")
Er_n = solver.Er.copy()
print(f"  Initial: min={Er_n.min():.6e}, max={Er_n.max():.6e}")

# Check if any negative values
if np.any(Er_n < 0):
    print(f"  ⚠ Starting with negative Er values!")
    neg_idx = np.where(Er_n < 0)[0]
    print(f"    Number of negative: {len(neg_idx)}")
    print(f"    Min Er = {Er_n.min():.6e}")

solver.dt = gamma * dt
solver.theta = 0.5

try:
    Er_intermediate = solver.newton_step(Er_n, verbose=False)
    print(f"  After TR: min={Er_intermediate.min():.6e}, max={Er_intermediate.max():.6e}")
except Exception as e:
    print(f"  ✗ TR stage failed: {e}")
    import sys; sys.exit(1)

solver.dt = dt
A_bdf2, rhs_bdf2 = solver.assemble_system_bdf2(Er_intermediate, Er_n, Er_intermediate, gamma)

print(f"  BDF2 matrix diagonal: min={A_bdf2[1,:].min():.6e}, max={A_bdf2[1,:].max():.6e}")
print(f"  BDF2 RHS: min={rhs_bdf2.min():.6e}, max={rhs_bdf2.max():.6e}")
print(f"  Matrix has NaN: {np.any(~np.isfinite(A_bdf2))}")
print(f"  RHS has NaN: {np.any(~np.isfinite(rhs_bdf2))}")

if np.any(~np.isfinite(A_bdf2)):
    nan_idx = np.where(~np.isfinite(A_bdf2[1,:]))[0]
    print(f"  NaN in matrix diagonal at indices: {nan_idx[:10]}")
    if len(nan_idx) > 0:
        idx = nan_idx[0]
        print(f"\n  Investigating first NaN at index {idx}:")
        print(f"    Er_intermediate[{idx}] = {Er_intermediate[idx]}")
        print(f"    Er_n[{idx}] = {Er_n[idx]}")
        
        # Check what caused NaN
        T_test = temperature_from_Er(Er_intermediate[idx])
        print(f"    T = {T_test}")
        
        try:
            dudEr = solver.get_dudEr(Er_intermediate[idx])
            print(f"    dudEr = {dudEr}")
        except:
            print(f"    dudEr calculation failed!")

print()
print("="*70)
print("The problem appears to be that small negative Er values")
print("can arise, and then temperature_from_Er(negative) produces NaN.")
print("="*70)
