#!/usr/bin/env python3
"""
Check matrix conditioning in BDF2 vs other methods with small k
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / 'utils'))
import numpy as np
from oneDFV import RadiationDiffusionSolver, temperature_from_Er, A_RAD, C_LIGHT, solve_tridiagonal, apply_tridiagonal

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
print("Matrix Conditioning Analysis: k=0.01, n_cells=200")
print("="*70)
print()

# Take one step to get Er_intermediate
Er_n = solver.Er.copy()
solver.dt = gamma * dt
solver.theta = 0.5
Er_intermediate = solver.newton_step(Er_n, verbose=False)

print("After first TR stage:")
print(f"  Er range: [{Er_intermediate.min():.6e}, {Er_intermediate.max():.6e}]")
print()

# Get BDF2 matrix
solver.dt = dt
A_bdf2, rhs_bdf2 = solver.assemble_system_bdf2(Er_intermediate, Er_n, Er_intermediate, gamma)

print("BDF2 Matrix:")
print(f"  Diagonal range: [{A_bdf2[1,:].min():.6e}, {A_bdf2[1,:].max():.6e}]")
print(f"  Sub-diagonal range: [{A_bdf2[0,1:].min():.6e}, {A_bdf2[0,1:].max():.6e}]")
print(f"  Super-diagonal range: [{A_bdf2[2,:-1].min():.6e}, {A_bdf2[2,:-1].max():.6e}]")
print()

# Check diagonal dominance
diag_dom_violations = 0
for i in range(n_cells):
    diag = abs(A_bdf2[1, i])
    off_diag = 0.0
    if i > 0:
        off_diag += abs(A_bdf2[0, i])
    if i < n_cells - 1:
        off_diag += abs(A_bdf2[2, i])
    
    if diag < off_diag:
        diag_dom_violations += 1

print(f"Diagonal dominance: {diag_dom_violations} violations out of {n_cells} cells")
print()

# Compare to implicit Euler matrix for same state
solver.dt = dt
solver.theta = 1.0
A_ie, rhs_ie = solver.assemble_system(Er_intermediate, Er_n, theta=1.0)

print("Implicit Euler Matrix (for comparison):")
print(f"  Diagonal range: [{A_ie[1,:].min():.6e}, {A_ie[1,:].max():.6e}]")
print(f"  Sub-diagonal range: [{A_ie[0,1:].min():.6e}, {A_ie[0,1:].max():.6e}]")
print(f"  Super-diagonal range: [{A_ie[2,:-1].min():.6e}, {A_ie[2,:-1].max():.6e}]")
print()

# Ratio of diagonals
diag_ratio = A_bdf2[1,:] / A_ie[1,:]
print(f"BDF2/IE diagonal ratio: [{diag_ratio.min():.3f}, {diag_ratio.max():.3f}]")
print(f"  Expected ratio ≈ c_0 = {(2-gamma)/(1-gamma):.3f}")
print()

# Try solving both systems
print("Attempting to solve both systems:")
print()

print("1. Implicit Euler:")
try:
    Er_ie = solve_tridiagonal(A_ie, rhs_ie)
    print(f"   ✓ Solved successfully")
    print(f"   Solution range: [{Er_ie.min():.6e}, {Er_ie.max():.6e}]")
    has_nan_ie = np.any(~np.isfinite(Er_ie))
    has_neg_ie = np.any(Er_ie < 0)
    print(f"   Has NaN: {has_nan_ie}, Has negative: {has_neg_ie}")
except Exception as e:
    print(f"   ✗ Failed: {e}")

print()

print("2. BDF2:")
try:
    Er_bdf2 = solve_tridiagonal(A_bdf2, rhs_bdf2)
    print(f"   ✓ Solved successfully")
    print(f"   Solution range: [{Er_bdf2.min():.6e}, {Er_bdf2.max():.6e}]")
    has_nan_bdf2 = np.any(~np.isfinite(Er_bdf2))
    has_neg_bdf2 = np.any(Er_bdf2 < 0)
    print(f"   Has NaN: {has_nan_bdf2}, Has negative: {has_neg_bdf2}")
    
    if has_neg_bdf2:
        neg_idx = np.where(Er_bdf2 < 0)[0]
        print(f"   {len(neg_idx)} negative values")
        print(f"   Most negative: {Er_bdf2.min():.6e} at index {Er_bdf2.argmin()}")
        
except Exception as e:
    print(f"   ✗ Failed: {e}")

print()
print("="*70)
print("ANALYSIS")
print("="*70)
print()
print("If BDF2 produces NaN while IE doesn't, the problem is specific")
print("to the BDF2 matrix structure (larger diagonal terms with c_0≈3.4)")
print("combined with small k_coefficient.")
