#!/usr/bin/env python3
"""
Check if the matrix assembly has correct signs for diffusion
"""

import numpy as np
from oneDFV import RadiationDiffusionSolver, temperature_from_Er, A_RAD, apply_tridiagonal

# Simple test problem
n_cells = 50
r_max = 2.0
dt = 0.01

def constant_opacity(T): return 100.0
def cubic_cv(T): return 4 * 1.0 * A_RAD * T**3
def linear_material_energy(T): return 1.0 * A_RAD * T**4
def left_bc(Er, r): return (0.0, 1.0, 0.0)
def right_bc(Er, r): return (0.0, 1.0, 0.0)

# Test with theta method
solver_theta = RadiationDiffusionSolver(
    n_cells=n_cells, r_max=r_max, dt=dt, theta=0.5, d=0,
    rosseland_opacity_func=constant_opacity,
    specific_heat_func=cubic_cv,
    material_energy_func=linear_material_energy,
    left_bc_func=left_bc,
    right_bc_func=right_bc
)

# Gaussian initial condition
amplitude = A_RAD * 1.0**4
x0 = 1.0
sigma0 = 0.15
gaussian_Er = lambda r: amplitude * np.exp(-(r - x0)**2 / (2 * sigma0**2))
solver_theta.set_initial_condition(gaussian_Er)

# Get diffusion matrix
Er_init = solver_theta.Er.copy()
L_tri = solver_theta.assemble_diffusion_matrix(Er_init)

# Apply to initial condition
L_Er = apply_tridiagonal(L_tri, Er_init)

print("="*70)
print("Checking Sign Convention of Diffusion Matrix")
print("="*70)
print()
print(f"Initial condition: Gaussian peak at r={x0}")
print(f"Peak Er = {Er_init.max():.6e} at index {Er_init.argmax()}")
print()
print("Diffusion operator L applied to Gaussian:")
peak_idx = Er_init.argmax()
print(f"  L(Er) at peak (idx={peak_idx}): {L_Er[peak_idx]:.6e}")
print(f"  L(Er) nearby: {L_Er[peak_idx-2:peak_idx+3]}")
print()

if L_Er[peak_idx] < 0:
    print("✓ L(Er) is NEGATIVE at peak -> diffusion removes energy from peak")
    print("  This is CORRECT for du/dt = +L(u) formulation")
else:
    print("✗ L(Er) is POSITIVE at peak -> diffusion adds energy to peak")  
    print("  This is WRONG for du/dt = +L(u) formulation")

print()
print("="*70)
print("Checking Matrix Assembly in Theta Method")
print("="*70)

# Assemble system for one Newton step
Er_prev = Er_init.copy()
A_tri, rhs = solver_theta.assemble_system(Er_init, Er_prev, theta=0.5)

print()
print("Theta method matrix: A = alpha + theta*L")
print(f"  A[1, peak_idx] = {A_tri[1, peak_idx]:.6e} (diagonal at peak)")
print(f"  L[1, peak_idx] = {L_tri[1, peak_idx]:.6e} (diffusion diagonal at peak)")

alpha_peak = solver_theta.get_alpha_coefficient(Er_init[peak_idx], dt)
expected_A_diag = alpha_peak + 0.5 * L_tri[1, peak_idx]
print(f"  Expected: alpha + theta*L = {alpha_peak:.6e} + 0.5*{L_tri[1, peak_idx]:.6e}")
print(f"           = {expected_A_diag:.6e}")
print(f"  Actual: {A_tri[1, peak_idx]:.6e}")
print(f"  Match: {np.isclose(A_tri[1, peak_idx], expected_A_diag)}")

print()
print("="*70)
print("Checking Matrix Assembly in BDF2 Method")
print("="*70)

gamma = 2.0 - np.sqrt(2.0)
c_0 = (2-gamma)/(1-gamma)

# Manually assemble BDF2 matrix
A_bdf2, rhs_bdf2 = solver_theta.assemble_system_bdf2(Er_init, Er_prev, Er_init, gamma)

print()
print("BDF2 matrix: A = a_0 - L")
print(f"  A[1, peak_idx] = {A_bdf2[1, peak_idx]:.6e} (diagonal at peak)")
print(f"  L[1, peak_idx] = {L_tri[1, peak_idx]:.6e} (diffusion diagonal at peak)")

dudEr_peak = solver_theta.get_dudEr(Er_init[peak_idx])
a_0_peak = (c_0 / dt) * dudEr_peak
expected_A_bdf2_diag = a_0_peak - L_tri[1, peak_idx]
print(f"  Expected: a_0 - L = {a_0_peak:.6e} - {L_tri[1, peak_idx]:.6e}")
print(f"           = {expected_A_bdf2_diag:.6e}")
print(f"  Actual: {A_bdf2[1, peak_idx]:.6e}")
print(f"  Match: {np.isclose(A_bdf2[1, peak_idx], expected_A_bdf2_diag)}")

print()
print("="*70)
print("SUMMARY")
print("="*70)
print()
if L_Er[peak_idx] < 0:
    print("The diffusion operator L has the CORRECT sign.")
    print("For du/dt = L(u), L should be negative at peaks.")
    print()
    print("Theta method uses: A = alpha + theta*L")
    print("  -> This means theta*L is POSITIVE in the matrix")
    print("  -> This is WRONG for du/dt = L(u)!")
    print()
    print("BDF2 method uses: A = a_0 - L")
    print("  -> This means -L is NEGATIVE in the matrix")  
    print("  -> This is CORRECT for du/dt = L(u)!")
else:
    print("The diffusion operator L has the WRONG sign.")
