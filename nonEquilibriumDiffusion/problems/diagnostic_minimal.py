#!/usr/bin/env python3
"""
Minimal diagnostic - print B operator and RHS on first iteration
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

def constant_opacity(T, r):
    return 300.0

def constant_diffusion_coeff(T, r):
    return C_LIGHT / (3.0 * 300.0)

# Setup
n_groups = 1
n_cells = 5  # Very small for diagnostics
r_min, r_max = 0.0, 0.5
dt = 0.001

energy_edges = np.array([0.01, 10.0])

rho = 1.0
cv = 0.3 / rho

T_bc = 1.0
phi_bc = C_LIGHT * A_RAD * T_bc**4

print("="*80)
print("Diagnostic: 1-group, constant σ=300, {} cells".format(n_cells))
print("="*80)

# Define boundary condition functions
def left_bc_func(phi, r):
    """Dirichlet BC: φ = φ_bc"""
    return 1.0, 0.0, phi_bc

def right_bc_func(phi, r):
    """Neumann BC: ∇φ = 0"""
    return 0.0, 1.0, 0.0

solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=r_min,
    r_max=r_max,
    n_cells=n_cells,
    energy_edges=energy_edges,
    geometry='planar',
    dt=dt,
    diffusion_coeff_funcs=[constant_diffusion_coeff],
    absorption_coeff_funcs=[constant_opacity],
    left_bc_funcs=[left_bc_func],
    right_bc_funcs=[right_bc_func],
    rho=rho,
    cv=cv
)

# Initial condition
T_init = 0.1
solver.T = np.full(n_cells, T_init)
solver.T_old = solver.T.copy()
solver.E_r = np.full(n_cells, A_RAD * T_init**4)
solver.E_r_old = solver.E_r.copy()

print(f"\nInitial state:")
print(f"  T = {solver.T}")
print(f"  E_r = {solver.E_r}")
print(f"  φ_bc = {phi_bc:.6e}")

# Manually do first Newton iteration
T_star = solver.T_old.copy()
solver.update_absorption_coefficients(T_star)
solver.fleck_factor = solver.compute_fleck_factor(T_star)

print(f"\nσ_a = {solver.sigma_a[0, :]}")
print(f"Fleck factor f = {solver.fleck_factor}")

# Compute ξ_0
xi_g_list = [solver.compute_source_xi(0, T_star)]
print(f"\nξ_0 = {xi_g_list[0]}")

# Compute RHS
rhs = solver.compute_rhs_for_kappa(T_star, xi_g_list)
print(f"\nRHS = Σ_g σ_a·A^(-1)·ξ_g =")
print(f"  {rhs}")

# Apply B to a test vector
kappa_test = np.ones(n_cells) * 10.0
B_kappa = solver.apply_operator_B(kappa_test, T_star, xi_g_list)
print(f"\nTest: kappa = {kappa_test}")
print(f"B·kappa = {B_kappa}")

# Check if B is identity-like for 1 group
print(f"\nB·kappa / kappa = {B_kappa / kappa_test}")
print(f"  (Should be close to 1 if f is small)")
