#!/usr/bin/env python3
"""
Debug script to verify flux limiters work in multigroup solver
"""
import sys
import os
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from multigroup_diffusion_solver import (
    MultigroupDiffusionSolver1D,
    C_LIGHT, A_RAD,
    flux_limiter_levermore_pomraning,
    flux_limiter_sum
)

# Simple test problem
n_groups = 1
n_cells = 10
x_min, x_max = 0.0, 1.0
energy_edges = np.array([0.0, 100.0])
sigma = 1.0
dt = 0.01

def absorption_coeff(T, r):
    return sigma

def rosseland_opacity(T, r):
    return sigma

print("Testing flux limiters in multigroup solver:")
print("="*60)

# Create two solvers with different flux limiters
print("\nCreating solver with Levermore-Pomraning limiter...")
solver_lp = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=x_min,
    r_max=x_max,
    n_cells=n_cells,
    energy_edges=energy_edges,
    geometry='planar',
    dt=dt,
    diffusion_coeff_funcs=None,
    absorption_coeff_funcs=[absorption_coeff],
    flux_limiter_funcs=[flux_limiter_levermore_pomraning],
    rosseland_opacity_funcs=[rosseland_opacity],
    emission_fractions=np.ones(1)
)

print("\nCreating solver with Sum limiter...")
solver_sum = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=x_min,
    r_max=x_max,
    n_cells=n_cells,
    energy_edges=energy_edges,
    geometry='planar',
    dt=dt,
    diffusion_coeff_funcs=None,
    absorption_coeff_funcs=[absorption_coeff],
    flux_limiter_funcs=[flux_limiter_sum],
    rosseland_opacity_funcs=[rosseland_opacity],
    emission_fractions=np.ones(1)
)

# Test diffusion coefficient evaluation with non-zero phi
print("\n" + "="*60)
print("Testing diffusion coefficient evaluation:")
print("="*60)

T_test = 1.0  # keV
r_test = 0.5  # cm  
phi_left = 1.0
phi_right = 2.0  # Create a gradient
dx = 0.1

# Test the diffusion coefficient functions directly
D_lp = solver_lp.solvers[0].diffusion_coeff_func(T_test, r_test, phi_left, phi_right, dx)
D_sum = solver_sum.solvers[0].diffusion_coeff_func(T_test, r_test, phi_left, phi_right, dx)

print(f"\nWith phi_left={phi_left}, phi_right={phi_right}, dx={dx}:")
print(f"  phi_avg = {0.5*(phi_left + phi_right)}")
print(f"  grad_phi = {abs(phi_right - phi_left)/dx}")
print(f"  R = grad_phi/(sigma*phi_avg) = {abs(phi_right - phi_left)/dx / (sigma * 0.5*(phi_left + phi_right)):.4f}")
print(f"\n  D (Levermore-Pomraning): {D_lp:.6f}")
print(f"  D (Sum): {D_sum:.6f}")
print(f"  Difference: {abs(D_lp - D_sum):.6e}")

if abs(D_lp - D_sum) > 1e-10:
    print("\n✓ SUCCESS: Different flux limiters produce different D values!")
else:
    print("\n✗ FAILURE: Flux limiters produce identical D values!")

# Test with larger gradient
phi_left = 0.1
phi_right = 10.0
D_lp2 = solver_lp.solvers[0].diffusion_coeff_func(T_test, r_test, phi_left, phi_right, dx)
D_sum2 = solver_sum.solvers[0].diffusion_coeff_func(T_test, r_test, phi_left, phi_right, dx)

print(f"\nWith larger gradient: phi_left={phi_left}, phi_right={phi_right}:")
print(f"  R = {abs(phi_right - phi_left)/dx / (sigma * 0.5*(phi_left + phi_right)):.4f}")
print(f"  D (Levermore-Pomraning): {D_lp2:.6f}")
print(f"  D (Sum): {D_sum2:.6f}")
print(f"  Difference: {abs(D_lp2 - D_sum2):.6e}")
