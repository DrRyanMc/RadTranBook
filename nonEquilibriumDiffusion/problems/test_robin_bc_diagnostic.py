#!/usr/bin/env python3
"""
Diagnostic test for Robin BC with large heat capacity (linear problem)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

RHO = 1.0

def marshak_opacity(T, r):
    """σ = 300 * T^-3"""
    T_min = 0.01
    T_safe = max(T, T_min)
    return 300.0 * T_safe**(-3.0)

def marshak_diffusion_coeff(T, r):
    """D = c/(3σ)"""
    sigma_R = marshak_opacity(T, r)
    return C_LIGHT / (3.0 * sigma_R)

# Problem setup
n_groups = 1
r_min = 0.0
r_max = 0.5
n_cells = 10  # Small for diagnostics
energy_edges = np.array([0.01, 10.0])
dt = 0.001

# VERY large heat capacity to keep T constant
cv_large = 1e12 * 0.3 / RHO
print(f"Heat capacity: cv = {cv_large:.2e} GJ/(g·keV)")
print(f"This is {cv_large / (0.3):2e}x larger than normal")

# Marshak Robin BC
T_bc = 1.0
sigma_R_bc = marshak_opacity(T_bc, 0.0)
A_marshak = 0.5
B_marshak = 1.0 / (3.0 * sigma_R_bc)
C_marshak = A_RAD * C_LIGHT * T_bc**4

print(f"\nMarshak Robin BC:")
print(f"  A = {A_marshak}")
print(f"  B = {B_marshak:.6e}")
print(f"  C = {C_marshak:.6e}")

def left_bc_func(phi, r):
    return A_marshak, B_marshak, C_marshak

def right_bc_func(phi, r):
    return 0.0, 1.0, 0.0

# Create solver
solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=r_min,
    r_max=r_max,
    n_cells=n_cells,
    energy_edges=energy_edges,
    geometry='planar',
    dt=dt,
    diffusion_coeff_funcs=[marshak_diffusion_coeff],
    absorption_coeff_funcs=[marshak_opacity],
    left_bc_funcs=[left_bc_func],
    right_bc_funcs=[right_bc_func],
    rho=RHO,
    cv=cv_large
)

# Initial condition
T_init = 0.1
solver.T = np.full(n_cells, T_init)
solver.T_old = solver.T.copy()
solver.E_r = np.full(n_cells, A_RAD * T_init**4)
solver.E_r_old = solver.E_r.copy()
solver.kappa = np.zeros(n_cells)
solver.kappa_old = np.zeros(n_cells)

print(f"\nInitial state:")
print(f"  T = {T_init} keV")
print(f"  E_r = {solver.E_r[0]:.6e} GJ/cm³")

# Compute Fleck factor
f_vals = solver.fleck_factor
print(f"\nFleck factor f = {f_vals[0]:.10f} (uniform)")
print(f"  (1-f) = {1-f_vals[0]:.2e}")

# Compute xi
# xi = (1-f) * (e_star - e_n) / dt = (1-f) * Δe / Δt
# Since T is constant, Δe = 0, so xi = 0
e_init = RHO * cv_large * T_init
print(f"\ne(T_init) = ρ·cv·T = {e_init:.2e} GJ/cm³")

# Take one step with verbose output
print(f"\n{'='*80}")
print("Taking one timestep...")
print(f"{'='*80}")

info = solver.step(max_newton_iter=3, newton_tol=1e-8, gmres_tol=1e-8, gmres_maxiter=100, verbose=True)

print(f"\nAfter step 1:")
print(f"  κ_min = {solver.kappa.min():.6e}")
print(f"  κ_max = {solver.kappa.max():.6e}")
print(f"  T_min = {solver.T.min():.6e}")
print(f"  T_max = {solver.T.max():.6e}")
print(f"  E_r_min = {solver.E_r.min():.6e}")
print(f"  E_r_max = {solver.E_r.max():.6e}")

if solver.kappa.min() < 0:
    print(f"\n⚠️ WARNING: κ is negative!")
    idx_neg = np.where(solver.kappa < 0)[0]
    print(f"  Negative κ at cells: {idx_neg}")
    print(f"  κ values: {solver.kappa[idx_neg]}")
