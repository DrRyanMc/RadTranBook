#!/usr/bin/env python3
"""
Simple multigroup test - constant opacity
Test the multigroup solver with a simple problem before tackling Marshak wave
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt
from multigroup_diffusion_solver import MultigroupDiffusionSolver1D, C_LIGHT, A_RAD

# Constants
RHO = 1.0  # g/cm³

# Simple constant properties
def const_diffusion(T, r):
    return 1.0  # cm²/ns

def const_absorption(T, r):
    return 0.5  # cm⁻¹ - much more reasonable value

print("="*80)
print("Simple 2-Group Test - Constant Opacity")
print("="*80)

# Problem setup
n_groups = 2
r_min = 0.0
r_max = 10.0
n_cells = 50
dt = 0.01  # ns

# Energy group edges (keV)
energy_edges = np.array([0.01, 2.0, 10.0])

# Boundary conditions
T_bc = 1.0  # keV
phi_bc_total = C_LIGHT * A_RAD * T_bc**4

# Compute emission fractions
from multigroup_diffusion_solver import compute_emission_fractions_from_edges
chi = compute_emission_fractions_from_edges(energy_edges, T_ref=T_bc)

print(f"Emission fractions: {chi}")
print(f"Total BC flux: {phi_bc_total:.6e} GJ/cm³")

# Split boundary condition
left_bc_values = [chi[g] * phi_bc_total for g in range(n_groups)]
right_bc_values = [0.0] * n_groups

print(f"Left BC per group: {left_bc_values}")

# Create solver
solver = MultigroupDiffusionSolver1D(
    n_groups=n_groups,
    r_min=r_min,
    r_max=r_max,
    n_cells=n_cells,
    energy_edges=energy_edges,
    geometry='planar',
    dt=dt,
    diffusion_coeff_funcs=[const_diffusion] * n_groups,
    absorption_coeff_funcs=[const_absorption] * n_groups,
    left_bc='dirichlet',
    right_bc='neumann',
    left_bc_values=left_bc_values,
    right_bc_values=right_bc_values,
    rho=RHO,
    cv=0.1  # GJ/(g·keV)
)

# Initial condition - warm
T_init = 0.5  # keV
solver.T = np.full(n_cells, T_init)
solver.T_old = solver.T.copy()
solver.E_r = np.full(n_cells, A_RAD * T_init**4)
solver.E_r_old = solver.E_r.copy()
solver.kappa = np.zeros(n_cells)
solver.kappa_old = np.zeros(n_cells)

print(f"\nInitial T = {T_init} keV")
print(f"Initial E_r = {solver.E_r[0]:.6e} GJ/cm³")

# Take a few timesteps
print("\nTaking 10 timesteps:")
for step in range(10):
    info = solver.step(max_newton_iter=10, newton_tol=1e-8,
                      gmres_tol=1e-6, gmres_maxiter=200,
                      verbose=(step==0))
    
    solver.advance_time()
    
    conv_str = "✓" if info['converged'] else "✗"
    print(f"Step {step+1}: Newton iter={info['newton_iter']}, "
          f"T_max={solver.T.max():.4f}, T_min={solver.T.min():.4f}, "
          f"E_r_max={solver.E_r.max():.4e} {conv_str}")

print("\n" + "="*80)
print("Simple test completed!")
print("="*80)
